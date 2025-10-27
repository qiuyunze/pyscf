import numpy as np
from pyscf import scf
from pyscf.scf import diis as diis_mod
from utils import _pack_index, packed_to_full, full_to_packed
from fock import fock2_np as fock2
from mndod import rotate_np as rotate
from h1elec import h1elec_np as h1elec
from pyscf import lib as pyscf_lib
from pyscf.lib import logger
import time
from functools import reduce

HARTREE2EV = 27.211386245988
EV2KCALMOL = 23.060547830619029  # FORTRAN 中 fpc_9，对应eV to kcal/mol 
def density_for_GPU(C: np.ndarray, fract: float,
                    nocc_main: int, nocc_open: int,
                    scale: float, mpack: int, norbs: int, one: int,
                    out_packed: np.ndarray, iopc: int):
    """
    构造密度矩阵（packed）。这里给出一个标准 RHF/UHF close-shell 的简化版：
    - 若 scale=2.0 通常是 RHF，总占据 na2el (关闭 + 开放)；
    - 若 scale=1.0 通常是 UHF 的 alpha 或 beta。
    注意：真正的程序有 open-shell 分数占据、DSYRK/DGEMM 路径等，这里先给最小保真实现。
    """
    # 构造占据数目（整数部分），忽略 fract/open-shell 细节（你可替换为完整版本）
    nocc = nocc_main
    # C: norbs x norbs, 列为 MO
    # 密度：P = scale * C_occ C_occ^T
    Cocc = C[:, :nocc]
    P = scale * (Cocc @ Cocc.T)
    # 压成 packed
    k = 0
    for i in range(norbs):
        for j in range(i + 1):
            out_packed[k] = P[i, j]
            k += 1

def densit(C: np.ndarray, norbs: int, _norbs2: int, na2el: int, scale: float,
           na1el: int, fract: float, out_packed: np.ndarray, one: int):
    # 旧的 densit 入口：用上面的 density_for_GPU 的简版实现替代
    density_for_GPU(C, fract, na2el, na1el, scale, out_packed.size, norbs, one, out_packed, 0)

def eigenvectors_LAPACK_from_packed(packed: np.ndarray, norbs: int) -> tuple[np.ndarray, np.ndarray]:
    """把 packed 对称阵还原为全矩阵后 eigh。返回 (eigvecs, eigvals)。"""
    M = np.zeros((norbs, norbs), dtype=packed.dtype)
    k = 0
    for i in range(norbs):
        for j in range(i + 1):
            M[i, j] = packed[k]
            M[j, i] = packed[k]
            k += 1
    w, v = np.linalg.eigh(M)
    # Fortran 代码里向量排在 c(i,j) 里；我们保持相同布局：列=本征向量
    return v, w
def helect(norbs: int,
                  p: np.ndarray,  # packed, lower
                  h: np.ndarray,  # packed, lower
                  f: np.ndarray   # packed, lower
                  ) -> float:
    """
    等价 Fortran helect，但保持 packed 计算。
    下三角 packed 顺序： (0,0),(1,0),(1,1),(2,0),(2,1),(2,2),...
    """
    s = h + f  # packed
    # 对角在 packed 中的索引：i*(i+1)//2
    diag_idx = (np.cumsum(np.arange(1, norbs+1)) - 1)  # 或 np.array([i*(i+1)//2 for i in range(norbs)])
    # sum_lower = 所有下三角（含对角）一次；Fortran 想要的是：非对角一次 + 对角半次
    return float(np.dot(p, s) - 0.5 * np.dot(p[diag_idx], s[diag_idx]))

def _as_spin2(arr, mpack):
    """把输入规整成 (nspin, mpack)；支持 1D、(2,mpack) 或 (arr_a, arr_b)。"""
    if arr is None:
        return None, 1
    if isinstance(arr, (tuple, list)):
        arr = np.stack(arr, axis=0)
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(1, mpack), 1
    if arr.ndim == 2 and arr.shape[1] == mpack:
        return arr, arr.shape[0]
    raise ValueError("packed array shape must be (mpack,) or (2, mpack)")

def get_err_vec_packed(s_p, d_p, f_p, n_orb):
    """
    计算 packed 形式的 DIIS 误差向量：
      err_full = (S D F).T - (S D F)   （与 PySCF 默认一致）
    返回：
      RHF: (mpack,)；UHF: 连接 [err_a_packed, err_b_packed] -> (2*mpack,)
    """
    mpack = n_orb*(n_orb+1)//2
    f2, nspin = _as_spin2(f_p, mpack)
    d2, _     = _as_spin2(d_p, mpack)
    if s_p is None:
        Sfull = np.eye(n_orb)
        S2 = None
    else:
        S2, _ = _as_spin2(s_p, mpack)
        # S 对两自旋应相同；若给了两套，就按各自用
    errs = []
    for i in range(nspin):
        Ffull = packed_to_full(f2[i], n_orb)
        Dfull = packed_to_full(d2[i], n_orb)
        Sfull_i = Sfull if S2 is None else packed_to_full(S2[min(i, S2.shape[0]-1)], n_orb)
        sdf = Sfull_i @ Dfull @ Ffull
        err_full = sdf.T - sdf                  # 反对称部分
        err_p = full_to_packed(err_full)        # 仅打下三角
        errs.append(err_p)
    if nspin == 1:
        return errs[0]
    return np.hstack(errs)                      # UHF: 拼成一维向量


class CDIIS(pyscf_lib.diis.DIIS):
    """
    纯 packed 版本的 CDIIS：
      - s_p, d_p, f_p 都是 packed 下三角；
      - UHF 用形状 (2,mpack) 的数组（或 (arr_a, arr_b)）；
      - 返回与输入 f_p 同形的 packed Fock。
    """
    def __init__(self, mf=None, filename=None):
        super().__init__(mf, filename)
        self.rollback = 0      # 与原版一致：到达 space 时仅保留末尾若干条
        self.space = 8
        self.damp = 0.0        # 若 >0，会对 f_p 与 f_prev 做线性混合后送入 DIIS

    def update(self, s_packed, d_packed, f_packed, n_orb, *args, **kwargs):
        # 误差（packed 或拼接后的 1D）：
        xerr = get_err_vec_packed(s_packed, d_packed, f_packed, n_orb)
        logger.debug1(self, 'diis-norm(errvec)=%g', np.linalg.norm(xerr))

        f_prev = kwargs.get('f_prev', None)
        if f_prev is not None and abs(self.damp) > 1e-6:
            f_in = (1.0 - self.damp) * f_packed + self.damp * f_prev
        else:
            f_in = f_packed

        # 调用 PySCF 通用 DIIS 内核；它会把 xerr 展平成一维使用
        f_new = pyscf_lib.diis.DIIS.update(self, f_in, xerr=xerr)

        # rollback：达到空间上限后，保留最近若干条记录
        if self.rollback > 0 and len(self._bookkeep) == self.space:
            self._bookkeep = self._bookkeep[-self.rollback:]

        return f_new

    def get_num_vec(self):
        if self.rollback:
            return self._head
        else:
            return len(self._bookkeep)


# ---------- helpers (packed) ----------

def _norb_from_mpack(mpack: int) -> int:
    n = int((np.sqrt(8*mpack + 1) - 1) / 2 + 1e-12)
    if n*(n+1)//2 != mpack:
        raise ValueError(f"mpack={mpack} 不是某个 n 的 n(n+1)/2")
    return n

def _packed_weights(n: int) -> np.ndarray:
    """长度 mpack 的权重向量：对角=1，非对角=2，用于 Tr[D F] 的 packed 内积。"""
    m = n*(n+1)//2
    w = np.full(m, 2.0, dtype=float)
    diag_idx = np.cumsum(np.arange(1, n+1)) - 1
    w[diag_idx] = 1.0
    return w

def _trace_DF_packed_batch(Ds: np.ndarray, Fs: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    batched 计算 df[i,j] = Tr[D_i F_j]，其中 D/F 都是 packed，一次性向量化：
      df = Ds @ (Fs * w)^T
    Ds,Fs: (nx, mpack)
    w:     (mpack,)
    return: (nx, nx)
    """
    return Ds @ (Fs * w).T

# ---------- EDIIS 最小化（packed） ----------

def ediis_minimize_packed(es: np.ndarray,
                          Ds: np.ndarray,   # (nx, mpack)
                          Fs: np.ndarray,   # (nx, mpack)
                          n_orb: int):
    """
    与 PySCF 的 ediis_minimize 对应，但全部在 packed 空间。
    构造 df[i,j] = Tr[D_i F_j]，再按原算法 df = diag_i + diag_j - df - df^T，
    最后用 BFGS 对 cost(c)=sum(c_i^2 E_i)/sum(c_i^2) - c^T df c 做极小化，
    并返回 (etot, c) 其中 c 已被归一化为 x^2 / sum(x^2)（保证非负）。
    """
    import scipy.optimize  # 保持与原版一致的依赖

    nx, mpack = Ds.shape
    assert Fs.shape == (nx, mpack)
    w = _packed_weights(n_orb)

    # df[i,j] = Tr[D_i F_j]
    df = _trace_DF_packed_batch(Ds, Fs, w)
    diag = np.diag(df)
    df = diag[:, None] + diag[None, :] - df - df.T

    es = es.astype(float, copy=False)

    def costf(x):
        c = x**2
        c /= c.sum()
        # sum(c_i E_i) - sum_{i,j} c_i df_ij c_j
        return float(np.dot(c, es) - c @ df @ c)

    def grad(x):
        # 与原版一致的解析梯度
        x2 = x**2
        x2sum = x2.sum()
        c = x2 / x2sum
        fc = es - 2.0 * (c @ df)         # 维度 nx
        cx = np.diag(x * x2sum) - np.outer(x2, x)
        cx *= 2.0 / (x2sum**2)
        return (fc @ cx).astype(float)

    res = scipy.optimize.minimize(costf, np.ones(nx), method='BFGS', jac=grad, tol=1e-9)
    x = res.x
    c = (x**2) / (x**2).sum()
    return float(res.fun), c

# ---------- 纯 packed 版 EDIIS ----------

class EDIIS(pyscf_lib.diis.DIIS):
    """
    纯 packed 版 EDIIS。
    调用约定（与原 EDIIS.update 原型类似，但全部视为 packed）：
        update(s_ignored, d_packed, f_packed, mf_ignored, h_packed, vhf_packed, *args, **kwargs) -> fock_packed_mixed

    - d_packed: 当前密度（RHF：总密度；UHF 若要分别做，就分 α/β 各调一次）
    - f_packed: 当前 Fock（若与 h_packed + vhf_packed 不一致，将以后者为准）
    - h_packed: 一电子（core H）packed
    - vhf_packed: 两电子 J-K（只包含 J-K，不含 AO 级 level-shift 等）
    返回：线性组合后的 Fock（packed）
    """
    def __init__(self, mf=None, filename=None):
        super().__init__(mf, filename)
        self.space = 8  # 可在外部修改
        # 内部缓冲初始化推迟到第一次 update
        # self._buffer = {}  # 由父类管理

    def update(self, s_ignored, d_p, f_p, mf_ignored, h_p, vhf_p, *args, **kwargs):
        # --- 基本尺寸与缓存 ---
        d_p = np.asarray(d_p)
        f_p = np.asarray(f_p)
        h_p = np.asarray(h_p)
        vhf_p = np.asarray(vhf_p)

        if self._head >= self.space:
            self._head = 0

        if not self._buffer:
            mpack = d_p.size
            self._n_orb = _norb_from_mpack(mpack)
            shape = (self.space, mpack)
            self._buffer['dm']   = np.zeros(shape, dtype=d_p.dtype)
            self._buffer['fock'] = np.zeros(shape, dtype=f_p.dtype)
            self._buffer['etot'] = np.zeros(self.space, dtype=float)

        # --- 用 h+vhf 作为“原始 Fock”，避免把混合后的 F 再喂回 ---
        f_raw = h_p + vhf_p
        if not np.allclose(f_p, f_raw, rtol=1e-8, atol=1e-12):
            # 若传入的 f 与 h+vhf 不一致，强制改用 h+vhf（EDIIS 的基本假设）
            f_p = f_raw

        # --- 写入历史 ---
        self._buffer['dm'  ][self._head] = d_p
        self._buffer['fock'][self._head] = f_p

        # 电子能：直接用 packed helect
        e_elec = helect(self._n_orb, d_p, h_p, f_p)
        self._buffer['etot'][self._head] = float(e_elec)
        self._head += 1

        # --- 若历史不足两条，直接返回当前 F ---
        nvec = self.get_num_vec()
        if nvec < 2:
            return f_p

        # --- 计算 EDIIS 系数并线性组合（全部在 packed 空间）---
        ds = self._buffer['dm'  ][:nvec]
        fs = self._buffer['fock'][:nvec]
        es = self._buffer['etot'][:nvec]

        etot, c = ediis_minimize_packed(es, ds, fs, self._n_orb)
        logger.debug1(self, 'EDIIS (packed) Etot %s  coeff %s', etot, c)

        # 组合 Fock（packed）
        f_mix = np.einsum('i,ip->p', c, fs, optimize=True)
        return f_mix

def _extract_pair_blocks_packed(pa, pb, p, jf, jl, if_, il_):
    padi, pbdi, pdi = [], [], []

    # (j,j) 上三角
    for I in range(jf, jl + 1):
        for J in range(jf, I + 1):
            k = (I * (I + 1)) // 2 + J
            padi.append(pa[k]); pbdi.append(pb[k]); pdi.append(p[k])

    # (i,j) 交叉块 + (i,i) 上三角
    for I in range(if_, il_ + 1):
        for J in range(jf, jl + 1):
            k = (I * (I + 1)) // 2 + J
            padi.append(pa[k]); pbdi.append(pb[k]); pdi.append(p[k])
        for J in range(if_, I + 1):
            k = (I * (I + 1)) // 2 + J
            padi.append(pa[k]); pbdi.append(pb[k]); pdi.append(p[k])

    return (np.asarray(pdi,  float),
            np.asarray(padi, float),
            np.asarray(pbdi, float))

class PM6HF(scf.hf.SCF):
    """
    PM6-HF（RHF/UHF）自洽：
      - Fock/J-K：在 packed(下三角) 上用 fock2_np 构造
      - 对角化：eigenvectors_LAPACK_from_packed（packed -> full -> eigh）
      - 密度：density_for_GPU（写回 packed）
      - 混合收敛：EDIIS（packed，稳） + CDIIS（full，快），带能量回退与自适应 level-shift
    """

    # ======================== 初始化 ========================
    def __init__(self, mol):
        super().__init__(mol)
        # 自旋类型
        self.uhf = (getattr(mol, "nalpha", 0) != getattr(mol, "nbeta", 0))

        # SCF/收敛参数
        self.max_cycle        = 200
        self.conv_tol         = 1e-8 / EV2KCALMOL  # Hartree（能量阈值，内部换算到 eV）
        self.conv_tol_grad    = 1e-6    # 梯度/残差阈值（用于 commutator 或 dP_rms）

        # Level-shift 设置（Ha）
        self.level_shift      = 0.0     # 常数位移（Ha），若启用 adaptive_shift 将被覆盖
        self.adaptive_shift   = True    # 按 pl 自适应分档：-0.30/-0.08/0.0 Ha（见 kernel）
        self.plchek           = 1e-4    # 残差小到这个阈值后，冻结上一轮 shift，防止抖动
        self.damp             = 0.0     # 密度阻尼（0~0.3 建议）

        # DIIS 策略
        self.hybrid_diis      = True   # True：EDIIS -> CDIIS（默认）；False：纯 CDIIS
        self.ediis_cycles     = 10       # 前 N 轮优先 EDIIS
        self.switch_pl        = 1e-4    # 换位子残差阈值；<=阈值认为进入线性区，允许 CDIIS
        self.diis_space       = 8
        self.diis_start_cycle = 1

        # I/O / 杂项
        self.chkfile          = None
        self.verbose          = max(self.verbose, 0)

        # 缓冲
        self._n      = int(mol.norbs)
        self._mpack  = self._n * (self._n + 1) // 2
        # 这些字段由外部 PM6 分子对象提供：h, f, fb, w, wk, p, pa, pb, c, cb, eigs, eigb, enuc, atheat, ...
        if not hasattr(mol, "f"):   mol.f  = mol.h.copy()
        if self.uhf and not hasattr(mol, "fb"): mol.fb = mol.h.copy()

        # DIIS 对象
        self._cdiis_a = CDIIS(self); self._cdiis_a.space = self.diis_space
        self._ediis_a = EDIIS(self);          self._ediis_a.space = self.diis_space
        if self.uhf:
            self._cdiis_b = CDIIS(self); self._cdiis_b.space = self.diis_space
            self._ediis_b = EDIIS(self);          self._ediis_b.space = self.diis_space
        else:
            self._cdiis_b = None
            self._ediis_b = None

        # PySCF 兼容字段（按需）
        if self.uhf:
            self.mo_coeff  = (None, None)
            self.mo_energy = (None, None)
            self.mo_occ    = (None, None)
        else:
            self.mo_coeff  = None
            self.mo_energy = None
            self.mo_occ    = None
        self.e_tot = None
        self.converged = False

        # 运行时状态
        self._last_pl_a = float("inf")
        self._last_pl_b = float("inf")
        self._last_ls_a = 0.0   # 上一次采用的 level-shift（Ha）
        self._last_ls_b = 0.0
        self._force_ediis_until = 0  # 回退后，强制 EDIIS 到该迭代号为止

    # ================ 内部工具：换位子残差范数 =================
    @staticmethod
    def _comm_norm(F, D):
        """ Frobenius || F D - D F || / N ；F、D 均为 full """
        n = F.shape[0]
        return float(np.linalg.norm(F @ D - D @ F, ord='fro') / max(1, n))

    @staticmethod
    def _apply_level_shift_packed(Fp, Pp, ls_ha, n_orb):
        """在 packed 上施加 level-shift:
           F := F + ls*P    ;  diag(F) -= ls
        """
        if abs(ls_ha) < 1e-14:  # 无位移
            return
        Fp += ls_ha * Pp
        for i in range(n_orb):
            Fp[_pack_index(i, i)] -= ls_ha

    # ======================== 主 kernel ========================
    def kernel(self, diis=True, dm0=None):
        log = logger.new_logger(self, self.verbose)

        m        = self.mol
        n_orb    = self._n
        n_pack   = self._mpack
        is_uhf   = self.uhf
        nalpha   = int(getattr(m, "nalpha", 0))
        nbeta    = int(getattr(m, "nbeta", 0))
        nclose   = int(getattr(m, "nclose", nalpha))
        nopen    = int(getattr(m, "nopen",  0))

        # ---------------- 初始密度 ----------------
        if dm0 is not None:
            if is_uhf:
                m.pa[:] = full_to_packed(dm0[0])
                m.pb[:] = full_to_packed(dm0[1])
                m.p [:] = m.pa + m.pb
            else:
                m.p [:] = full_to_packed(dm0)
                m.pa[:] = 0.5 * m.p
                m.pb[:] = m.pa
        else:
            # 用 pdiag 生成初猜
            m.p[:]  = 0.0; m.pa[:] = 0.0; m.pb[:] = 0.0
            w1 = nalpha / (nalpha + 1e-6 + nbeta); w2 = 1.0 - w1
            rnd = 1.0
            for i in range(n_orb):
                j = _pack_index(i, i)
                m.p[j]  = m.pdiag[i]
                m.pa[j] = m.pdiag[i] * w1 * rnd
                rnd     = 1.0 / rnd
                m.pb[j] = m.pdiag[i] * w2 * rnd
            if not is_uhf:
                m.pa[:] = 0.5 * m.p
                m.pb[:] = m.pa

        # 历史量
        p_prev  = m.p.copy()
        pa_prev = m.pa.copy()
        pb_prev = m.pb.copy()

        e_prev_eV = None
        e_conv_thr_eV = float(self.conv_tol) * HARTREE2EV

        # ====================== SCF 迭代 ======================
        for niter in range(1, self.max_cycle + 1):
            cycl_t0 = time.perf_counter()

            # ---- 计算本轮的 level-shift（Ha），按上一轮残差自适应 ----
            if self.adaptive_shift:
                def choose_ls(pl_last, ls_last):
                    if pl_last > 10.0 * self.switch_pl:  ls = -0.30
                    elif pl_last > self.switch_pl:       ls = -0.08
                    else:                                 ls =  0.00
                    # 残差很小，冻结上一轮 shift，避免抖动
                    if pl_last < self.plchek:
                        ls = ls_last
                    return ls
                ls_a = choose_ls(self._last_pl_a, self._last_ls_a)
                ls_b = choose_ls(self._last_pl_b, self._last_ls_b) if is_uhf else 0.0
            else:
                ls_a = float(self.level_shift)
                ls_b = float(self.level_shift) if is_uhf else 0.0

            # ---- 构造 Fock（packed）+ 施加 level-shift ----
            m.f[:] = m.h
            # 先施加 LS（在 H 上），再叠加两电子
            self._apply_level_shift_packed(m.f,  m.pa, ls_a, n_orb)
            t0 = time.perf_counter()
            fock2(m.f, m.p, m.pa, m.w, m.w, m.wk if getattr(m, "wk", None) is not None else m.w,
                  m.numat, m.nfirst, m.nlast, mode=2)
            t1 = time.perf_counter()
            log.debug(f"Fock2 time: {t1 - t0:.6f} s")
            if is_uhf:
                m.fb[:] = m.h
                self._apply_level_shift_packed(m.fb, m.pb, ls_b, n_orb)
                fock2(m.fb, m.p, m.pb, m.w, m.w, m.wk if getattr(m, "wk", None) is not None else m.w,
                      m.numat, m.nfirst, m.nlast, mode=2)

            # ---- DIIS 混合（按开关：混合 or 纯 CDIIS）----
            t0 = time.perf_counter()
            tag_a = tag_b = "KEEP"
            pl = float("inf")
            if diis:
                # packed -> full（CDIIS 用 full）
                Fa_full = packed_to_full(m.f,  n_orb)
                Da_full = packed_to_full(m.pa, n_orb)
                Dt_full = packed_to_full(m.p,  n_orb)
                S_full  = np.eye(n_orb)
                if is_uhf:
                    Fb_full = packed_to_full(m.fb, n_orb)
                    Db_full = packed_to_full(m.pb, n_orb)

                # 进入线性区的判据（用当前 F 与当前密度估一个 pl）
                rnorm_a_now = self._comm_norm(Fa_full, Da_full if is_uhf else Dt_full)
                allow_cdiis = (rnorm_a_now <= self.switch_pl) or (niter > self.ediis_cycles)

                # 若上一步触发了回退，则强制 EDIIS 到指定轮次
                if niter <= self._force_ediis_until:
                    allow_cdiis = False

                if self.hybrid_diis and not allow_cdiis:
                    # —— EDIIS 期（稳定）
                    try:
                        vhf_a   = m.f - m.h
                        Fp_a_new = self._ediis_a.update(None, m.pa, m.f, None, m.h, vhf_a)
                        tag_a = "EDIIS"
                    except Exception:
                        Fp_a_new = m.f.copy()
                        tag_a = "EDIIS-FAIL->KEEP"
                    if is_uhf:
                        try:
                            vhf_b   = m.fb - m.h
                            Fp_b_new = self._ediis_b.update(None, m.pb, m.fb, None, m.h, vhf_b)
                            tag_b = "EDIIS"
                        except Exception:
                            Fp_b_new = m.fb.copy()
                            tag_b = "EDIIS-FAIL->KEEP"
                else:
                    # —— CDIIS 期（快速）
                    try:
                        Fp_a_new = self._cdiis_a.update(None, d_packed=m.pa, f_packed=m.f, n_orb=n_orb) 
                        # Fp_a_new = full_to_packed(Fa_new_full)
                        tag_a = "CDIIS"
                    except Exception:
                        self._cdiis_a.clear(); self.diis_space = max(4, self.diis_space - 2)
                        Fp_a_new = m.f.copy(); tag_a = "CDIIS-FAIL->KEEP"
                    if is_uhf:
                        try:
                            Fb_new_full = self._cdiis_b.update(S_full, Db_full, Fb_full)
                            Fp_b_new = full_to_packed(Fb_new_full)
                            tag_b = "CDIIS"
                        except Exception:
                            self._cdiis_b.clear(); self.diis_space = max(4, self.diis_space - 2)
                            Fp_b_new = m.fb.copy(); tag_b = "CDIIS-FAIL->KEEP"

                # 写回混合后的 Fock
                m.f[:] = Fp_a_new
                if is_uhf:
                    m.fb[:] = Fp_b_new
                # 计算混合后的换位子残差，用于日志/回退
                Fa_mix = packed_to_full(m.f, n_orb)
                Da_now = packed_to_full(m.pa, n_orb)
                Dt_now = packed_to_full(m.p,  n_orb)
                pla = self._comm_norm(Fa_mix, Da_now if is_uhf else Dt_now)
                if is_uhf:
                    Fb_mix = packed_to_full(m.fb, n_orb)
                    Db_now = packed_to_full(m.pb, n_orb)
                    plb = self._comm_norm(Fb_mix, Db_now)
                    pl  = max(pla, plb)
                else:
                    pl  = pla

                # 若 CDIIS 期且残差显著变差，触发回退：清空并强制 EDIIS 3 步
                if "CDIIS" in tag_a or "CDIIS" in tag_b:
                    if pl > 1.5 * max(self._last_pl_a, self._last_pl_b):
                        # 回退
                        try: self._cdiis_a.clear()
                        except Exception: pass
                        if is_uhf:
                            try: self._cdiis_b.clear()
                            except Exception: pass
                        self.diis_space = max(4, self.diis_space - 2)
                        self._force_ediis_until = niter + 3
                        tag_a += "->ROLLBACK"
                        if is_uhf: tag_b += "->ROLLBACK"
            else:
                # 不用 DIIS：直接评一个 pl
                Fa_full = packed_to_full(m.f, n_orb)
                Dt_full = packed_to_full(m.p, n_orb)
                Da_full = packed_to_full(m.pa, n_orb)
                pl = self._comm_norm(Fa_full, Da_full if is_uhf else Dt_full)
                tag_a = "NO-DIIS"; tag_b = "NO-DIIS"

            t1 = time.perf_counter()
            log.debug(f"DIIS time: {t1 - t0:.6f} s")
            # ---- 能量（eV）----
            ee_eV = helect(n_orb, m.pa, m.h, m.f)
            if is_uhf: ee_eV += helect(n_orb, m.pb, m.h, m.fb)
            else:      ee_eV *= 2.0
            e_tot_eV = ee_eV + float(getattr(m, "enuc", 0.0))

            # ---- 对角化 -> 新密度 ----
            t0 = time.perf_counter()
            C, evals = eigenvectors_LAPACK_from_packed(m.f, n_orb)
            t1 = time.perf_counter()
            log.debug(f"Eigen time: {t1 - t0:.6f} s")
            m.c[:]    = C
            m.eigs[:] = evals
            if is_uhf:
                Cb, evalsb = eigenvectors_LAPACK_from_packed(m.fb, n_orb)
                m.cb[:]   = Cb
                m.eigb[:] = evalsb

            if is_uhf:
                density_for_GPU(m.c,  getattr(m, "fract", None), m.nalpha, m.nalpha, 1.0, n_pack, n_orb, 1, m.pa, 0)
                density_for_GPU(m.cb, getattr(m, "fract", None), m.nbeta,  m.nbeta,  1.0, n_pack, n_orb, 1, m.pb, 0)
                m.p[:] = m.pa + m.pb
            else:
                na2el = nclose
                na1el = nalpha + nopen
                t0 = time.perf_counter()
                density_for_GPU(m.c, getattr(m, "fract", None), na2el, na1el, 2.0, n_pack, n_orb, 1, m.p, 0)
                t1 = time.perf_counter()
                log.debug(f"Density time: {t1 - t0:.6f} s")
                m.pa[:] = 0.5 * m.p
                m.pb[:] = m.pa

            # 可选阻尼（packed）
            if self.damp and self.damp > 0.0:
                a = float(self.damp)
                m.p[:]  = (1-a)*m.p  + a*p_prev
                if is_uhf:
                    m.pa[:] = (1-a)*m.pa + a*pa_prev
                    m.pb[:] = (1-a)*m.pb + a*pb_prev

            # 收敛判据
            de_eV  = np.inf if e_prev_eV is None else abs(e_tot_eV - e_prev_eV)
            dp     = m.p - p_prev
            dp_rms = float(np.sqrt(np.dot(dp, dp)/n_pack))
            cycl_t1 = time.perf_counter()
            if self.verbose >= 4:
                if is_uhf:
                    tag_str = f"A:{tag_a} B:{tag_b}  lsA={ls_a:.3f}Ha lsB={ls_b:.3f}Ha"
                else:
                    tag_str = f"A:{tag_a}          lsA={ls_a:.3f}Ha"
                log.info("cycle %3d: E_tot(eV)= %.12f  dE= %.3e  pl= %.3e  dP_rms= %.3e  %s  time= %.3fs",
                         niter, e_tot_eV, de_eV, pl, dp_rms, tag_str, cycl_t1 - cycl_t0)

            converged = (niter > 2) and (de_eV < e_conv_thr_eV) and (max(pl, dp_rms) < self.conv_tol_grad)
            if converged:
                # 最终结果：保存 MO / occ
                if is_uhf:
                    self.mo_coeff  = (m.c.copy(),  m.cb.copy())
                    self.mo_energy = (m.eigs[:n_orb].copy(), m.eigb[:n_orb].copy())
                    occ_a = np.zeros(n_orb); occ_a[:m.nalpha] = 1.0
                    occ_b = np.zeros(n_orb); occ_b[:m.nbeta]  = 1.0
                    self.mo_occ = (occ_a, occ_b)
                else:
                    self.mo_coeff  = m.c.copy()
                    self.mo_energy = m.eigs[:n_orb].copy()
                    occ = np.zeros(n_orb); occ[:m.nalpha] = 2.0
                    self.mo_occ = occ

                self.e_tot     = e_tot_eV
                self.converged = True

                hof_kcal = (ee_eV + float(getattr(m, "enuc", 0.0))) * EV2KCALMOL + float(getattr(m, "atheat", 0.0))
                return e_tot_eV, hof_kcal

            # —— 记录当前残差/shift，准备下一轮 —— #
            self._last_pl_a = pl if not is_uhf else self._comm_norm(
                packed_to_full(m.f, n_orb), packed_to_full(m.pa, n_orb)
            )
            if is_uhf:
                self._last_pl_b = self._comm_norm(
                    packed_to_full(m.fb, n_orb), packed_to_full(m.pb, n_orb)
                )
            self._last_ls_a = ls_a
            self._last_ls_b = ls_b

            e_prev_eV = e_tot_eV
            p_prev    = m.p.copy()
            pa_prev   = m.pa.copy()
            pb_prev   = m.pb.copy()

        # 未收敛
        self.converged = False
        self.e_tot     = e_prev_eV
        raise RuntimeError("SCF not converged")
    
    # ====== 主函数：笛卡尔梯度（对 dhc 做中心差分；不使用内部坐标） ======
    def pm6_dcart_gradient(self, prec: bool=False) -> np.ndarray:
        """
        输入：
        positions: (N,3) 坐标
        输出：
        grad: (N,3) = dE/dR（受力= -grad）
        """
        def _dhc_energy(p: np.ndarray, pa: np.ndarray, pb: np.ndarray,
                        xi: np.ndarray, nat: np.ndarray,
                        if_: int, il_: int, jf: int, jl: int, uhf: bool,
                            env=None) -> float:
            """
            等价 Fortran: subroutine dhc(..., dener, mode) 的最简分子版本（忽略 mode/id 截断）。
            输入：
            p,pa,pb:  这一对 (i,j) 的 packed 二原子密度块（一维）
            xi:       (3,2) = [R_j | R_i]
            nat:      (2,)  = [Z_j, Z_i]
            if_,il_, jf,jl: i/j 的 AO 全局范围（仅用于确定局部尺寸）
            返回：该对的总能量（电子能 + 核斥）
            """
            _w = np.zeros(2026, float)   # 为兼容接口，留空数组占位
            _wk = np.zeros(2026, float)  # 同上（分子体系下用不到）
            # 局部 AO 映射（与 Fortran 一致）
            nfirst_loc = np.zeros(2, dtype=int)
            nlast_loc  = np.zeros(2, dtype=int)
            nfirst_loc[0] = 1
            nlast_loc[0]  = il_ - if_ + 1
            nfirst_loc[1] = nlast_loc[0] + 1
            nlast_loc[1]  = nfirst_loc[1] + (jl - jf)
            linear = (nlast_loc[1] * (nlast_loc[1] + 1)) // 2
            f = np.zeros(linear, float)
            h = np.zeros(linear, float)

            ia, ic = nfirst_loc[1], nlast_loc[1]   
            ja, jc = nfirst_loc[0], nlast_loc[0]
            nj, ni = int(nat[0]), int(nat[1])
            # 一电子项
            mat = h1elec(nj, ni, xi[:,0], xi[:, 1], env=env)
            m, n = mat.shape
            shmat =  np.zeros((9, 9))
            shmat[:m, :n] = mat
            j1 = 0
            for j in range(ia, ic + 1):
                jj = (j * (j - 1)) // 2
                j1 += 1
                i1 = 0
                for i in range(ja, jc + 1):
                    jj += 1
                    i1 += 1
                    val = shmat[i1 - 1, j1 - 1]
                    h[jj - 1] = val
                    f[jj - 1] = val
            _, e2a, e1b, enuclr_loc = rotate(ni, nj,  xi[:,1], xi[:,0], _w, 0, env)   
            i2 = 0
            ja0, jc0 = ja - 1, jc - 1
            ia0, ic0 = ia - 1, ic - 1
            # --- 把 e1b 加到 (j,j) 自块：Fortran: i1=ja..jc; j1=ja..i1 ---
            for i1 in range(ja0, jc0 + 1):
                ii = i1 * (i1 + 1) // 2 + (ja0 - 1)   # 先设成“起点前一位”，循环内先自增
                for j1 in range(ja0, i1 + 1):
                    ii += 1
                    h[ii] += e1b[i2]
                    f[ii] += e1b[i2]
                    i2 += 1
            # --- 把 e2a 加到 (i,i) 自块：Fortran: i1=ia..ic; j1=ia..i1 ---
            i2 = 0
            for i1 in range(ia0, ic0 + 1):
                ii = i1 * (i1 + 1) // 2 + (ia0 - 1)
                for j1 in range(ia0, i1 + 1):
                    ii += 1
                    h[ii] += e2a[i2]
                    f[ii] += e2a[i2]
                    i2 += 1

            i_flag = -2  
            fock2(f, p, pa, _w, _w, _wk, i_flag, nfirst_loc-1, nlast_loc-1, mode=1)   # fock2 中要有deriv分支 mode=1 
            ee = helect(nlast_loc[1], pa, h, f)
            if uhf:
                f_beta = h.copy()
                fock2(f_beta, p, pb, _w, _w, _wk, i_flag, nfirst_loc-1, nlast_loc-1, mode=1)
                ee += helect(nlast_loc[1], pb, h, f_beta)
            else:
                ee *= 2.0
            return ee + enuclr_loc
        
        mol = self.mol
        env = mol.PM6env
        R = np.asarray(mol.coord, float)
        N = mol.numat
        grad = np.zeros((N, 3), float)
        h = 1e-4
        uhf = (not mol.rhf)
        ### 如何并行化
        for i in range(N):
            if_, il_ = int(mol.nfirst[i]), int(mol.nlast[i])
            for j in range(i):
                w = 0.5 if (i == j) else 1.0  
                jf, jl = int(mol.nfirst[j]), int(mol.nlast[j])
                pdi, padi, pbdi = _extract_pair_blocks_packed(mol.pa, mol.pb, mol.p, jf, jl, if_, il_)
                cdi = np.empty((3, 2), float)
                cdi[:, 0] = R[:, j]      # R_j
                cdi[:, 1] = R[:, i]      # R_i
                if not prec:
                    cdi[:, 0] += h/2
                    E_minus = _dhc_energy(pdi, padi, pbdi, cdi,
                                    np.array([mol.Z[j], mol.Z[i]], int),
                                    jf, jl, if_, il_, uhf, env=env)
                for k in range(3):
                    if prec:
                        cdi[k, 1] = cdi[k, 1] - h/2
                        E_minus = _dhc_energy(pdi, padi, pbdi, cdi,
                                    np.array([mol.Z[j], mol.Z[i]], int),
                                    jf, jl, if_, il_, uhf, env=env)
                    cdi[k, 1] += h
                    E_plus = _dhc_energy(pdi, padi, pbdi, cdi,
                                        np.array([mol.Z[j], mol.Z[i]], int),
                                    jf, jl, if_, il_, uhf, env=env)
                    cdi[k, 1] -= h/2
                    if not prec:
                        cdi[k, 1] -= h/2
                    dE_dRik = w * (E_minus - E_plus) * EV2KCALMOL / h
                    grad[i, k] += dE_dRik
                    grad[j, k] -= dE_dRik
        return -grad

    def pm6_forces(self) -> np.ndarray:
        """受力 = -梯度"""
        return - self.pm6_dcart_gradient(prec=True)