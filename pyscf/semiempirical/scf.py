import numpy as np
from pyscf import scf
from pyscf.scf import diis as diis_mod
from utils import _pack_index, packed_to_full, full_to_packed, _packed_diag_index
from fock import fock2_np as fock2
from mndod import rotate_np as rotate
from h1elec import h1elec_np as h1elec
from model_initialization import PM6Init
from pyscf import lib
from pyscf.lib import logger
import time

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

def copy_vec(dst: np.ndarray, src: np.ndarray):
    """Fortran dcopy 替代。"""
    np.copyto(dst, src)


# ====== 主类：借用 PySCF 的 DIIS，但 kernel 自己写 ======
class PM6HF(scf.hf.SCF):
    """
    自定义 PM6-HF 收敛器：
    - 支持 RHF / UHF（由 mol.nalpha == mol.nbeta 判断）
    - Fock/J-K 用你的 fock2_np 在 packed 上构造
    - 对角化用你的 eigenvectors_LAPACK_from_packed
    - 密度用你的 density_for_GPU
    - DIIS 使用 PySCF 的 DbIIS 类（在 full 矩阵上做）
    """
    def __init__(self, mol):
        super().__init__(mol)
        self.uhf = (getattr(mol, "nalpha", 0) != getattr(mol, "nbeta", 0))
        self.max_cycle = 1000
        self.diis_space = 8
        self.diis_start_cycle = 1
        self.level_shift = 0.1    # Hartree（MO 级）；>0 更稳
        self.damp = 0.0           # 0~0.5 建议
        self.conv_tol = 1e-9      # Hartree
        self.conv_tol_grad = 1e-6
        self.chkfile = None       # 禁用写 chk
        self.verbose = max(self.verbose, 0)

        # 一次性缓冲
        self._n = int(mol.norbs)
        self._mpack = self._n*(self._n+1)//2
        self._Fp  = mol.h.copy()
        if self.uhf:
            self._Fb = mol.h.copy()

        # DIIS 对象
        # self._diis_a = diis_mod.CDIIS(self)
        # self._diis_b = diis_mod.CDIIS(self) if self.uhf else None

        # 输出字段占位（PySCF 习惯）
        self.e_tot = None
        self.mo_coeff = None
        self.mo_energy = None
        self.mo_occ = None
        if self.uhf:
            self.mo_coeff = (None, None)
            self.mo_energy = (None, None)
            self.mo_occ = (None, None)
        
    # ====== 主核：自写 SCF 迭代 ======
    def kernel(self, diis=False):
        mol = self.mol
        norbs, mpack = mol.norbs, mol.mpack
        nalpha, nbeta = mol.nalpha, mol.nbeta
        nclose, nopen = mol.nclose, mol.nopen
        uhf = (not mol.rhf)

        # RHF（最稳妥）：ihomo = nalpha - 1
        ihomo = max(0, nalpha - 1)

        # UHF:
        ihomo_a = max(0, nalpha - 1)
        ihomo_b = max(0, nbeta  - 1)
        eold = 1.0e2
        dp = np.zeros_like(mol.p)
        fulscf = False
        diff = 0.0
        sellim = 0.0

        # SCF 收敛准则
        scfcrt = 1.0e-9 * EV2KCALMOL # 1.0e-10
        scfcrt = max(scfcrt, 1.0e-12)
        selcon = scfcrt
        plchek = 5e-3
        pltest = 0.05*np.sqrt(abs(selcon))
        if uhf and (nalpha!=nbeta): pltest = 1e-3

        # 初始旧密度
        rnd = 1.1 if (uhf and nalpha==nbeta) else 1.0
        w1 = nalpha/(nalpha + 1E-6 + nbeta)
        w2 = 1.0 - w1
        for i in range(norbs):                      # i: 0..norbs-1
            j = _pack_index(i, i)                   # 打包下三角 (i,i) 的 0-base 线性下标
            mol.p[j]  = mol.pdiag[i]
            mol.pa[j] = mol.p[j] * w1 * rnd
            rnd   = 1.0 / rnd                       # 在 1.1 与 ~0.909... 之间交替
            mol.pb[j] = mol.p[j] * w2 * rnd
        # 迭代循环
        niter = 0
        shift = 1.0        ## FORTRAN default 
        shiftb = 0.0
        bshift = -80.0
        ten, tenold = 10.0, 10.0
        
        if abs(bshift) > 1e-5: 
            ten = bshift
        
        shfmax = 20.0
        max_iter = 100 # 2000

        pold  = mol.p.copy()
        paold = mol.pa.copy()
        pbold = mol.pb.copy()
        # 对角历史：长度 norbs，一开始用“上一轮对角”——此处就是当前初猜的对角
        def _diag_from_packed(vec, norbs):
            return np.array([vec[i*(i+1)//2 + i] for i in range(norbs)], dtype=vec.dtype)

        p1_tot = _diag_from_packed(pold,  norbs)  # RHF 用
        p1_a   = _diag_from_packed(paold, norbs)  # α
        p1_b   = _diag_from_packed(pbold, norbs)  # β
        while niter < max_iter:
            tstart = time.time()
            niter += 1
            # ---- 构造 alpha Fock 的“一电子+shift*Pα”起始（packed） ----

            if niter > 1 and bshift != 0.0:
                if mol.rhf:
                    gap_a = evals[ihomo+1] - evals[ihomo] if ihomo+1 < norbs else 0.0
                else:
                    gap_a = evals [ihomo_a+1] - evals [ihomo_a] if ihomo_a+1 < norbs else 0.0
                    gap_b = evalsb[ihomo_b+1] - evalsb[ihomo_b] if ihomo_b+1 < norbs else 0.0

                shift  = np.clip(ten + gap_a + shift,  -20.0, shfmax)
                shiftb = np.clip(ten + gap_b + shiftb, -20.0, shfmax) if uhf else 0.0

                # 冻结/回滚
                if (not uhf and pl < plchek) or (uhf and max(pla,plb) < plchek):
                    shift, shiftb = shfto, shftbo
                else:
                    shfto, shftbo = shift, shiftb
            eps = 1e-16
            mol.f[:] = mol.h + shift*mol.pa
            mol.f[:] += eps * np.arange(1, mpack+1, dtype=mol.f.dtype)
            for i in range(norbs): mol.f[_pack_index(i,i)] -= shift
            if uhf:
                mol.fb[:] = mol.h + shiftb*mol.pb
                mol.fb[:] += eps * np.arange(1, mpack+1, dtype=mol.fb.dtype)
                for i in range(norbs): mol.fb[_pack_index(i,i)] -= shiftb
            # ---- 两电子部分 ----
            # 典型签名：fock2_np(f, ptot, p_spin, w, wj, wk, numat, nfirst, nlast, mode)
            # 对 RHF：ptot=p，p_spin=pa（或实际的 alpha 部分）；对 UHF：分别调用两次
            t1 = time.time()
            fock2(mol.f, mol.p, mol.pa, mol.w, mol.w, mol.wk if mol.wk is not None else mol.w,
                    mol.numat, mol.nfirst, mol.nlast, mode=2)     # 耗时严重
            t2 = time.time()
            print(f"fock2 time: {t2 - t1:.3f} s")
            # ---- UHF 的 beta Fock ----
            if uhf:
                fock2(mol.fb, mol.p, mol.pb, mol.w, mol.w, mol.wk if mol.wk is not None else mol.w,
                        mol.numat, mol.nfirst, mol.nlast, 2)

            # ---- 能量（本轮 F 上的电子能） ----
            # t1 = time.time()
            ee = helect(norbs, mol.pa, mol.h, mol.f)
            # t2 = time.time()
            # print(f"helect time: {t2 - t1:.3f} s")
            if uhf:
                ee += helect(norbs, mol.pb, mol.h, mol.fb)
            else:
                ee *= 2.
            
            # ---- 自洽/收敛判据 ----
            scorr = 0.0
            escf = (ee + mol.enuc) * EV2KCALMOL + mol.atheat + scorr
            # 快速退出/极限检查
            if niter >= 2000:
                # 这里只把 Fortran 的提示逻辑省略
                raise RuntimeError("UNABLE TO ACHIEVE SELF-CONSISTENCE")

            # —— 对角化得到轨道与新密度
            t1 = time.time()
            C, evals = eigenvectors_LAPACK_from_packed(mol.f, norbs)  # 耗时严重
            mol.c[:] = C
            mol.eigs[:] = evals
            t2 = time.time()
            print(f"eigenvectors_LAPACK_from_packed time: {t2 - t1:.3f} s")

            if uhf:
                Cb, evalsb = eigenvectors_LAPACK_from_packed(mol.fb, norbs)  
                mol.cb[:] = Cb
                mol.eigb[:] = evalsb

            if uhf:
                density_for_GPU(mol.c, mol.fract, mol.nalpha, mol.nalpha, 1.0, mol.mpack, norbs, 1, mol.pa, 0)
                density_for_GPU(mol.cb, mol.fract, mol.nbeta,  mol.nbeta,  1.0, mol.mpack, norbs, 1, mol.pb, 0)
                # —— 混合（就地更新），要先备份“混合前的旧密度”用来算 dP ——
                if not diis:
                    pla = self._cnvg_mix(mol.pa, paold, p1_a, niter, uhf=True)
                    plb = self._cnvg_mix(mol.pb, pbold, p1_b, niter, uhf=True)
                mol.p[:] = mol.pa + mol.pb
            else:
                # RHF：scale=2.0，总占据 na2eld
                na2el = nclose
                na1el = nalpha + nopen
                density_for_GPU(mol.c, mol.fract, na2el, na1el, 2.0, mol.mpack, norbs, 1, mol.p, 0)
                # —— 混合（就地更新），要先备份“混合前的旧密度”用来算 dP ——
                if not diis:
                    pl  = self._cnvg_mix(mol.p, pold, p1_tot, niter, uhf=False) 
                mol.pa[:] = 0.5 * mol.p
                mol.pb[:] = mol.pa
            
            # ---- 简化的收敛判断：能量/密度变化 ----
            sellim = max(selcon, 1e-15 * max(abs(ee), 1.0))
            diff = escf - eold

            if diff > 0:
                ten = ten - 1.0
                if shift > 4.0:
                    shfmax = 4.5
                if shift > shfmax:
                    shfmax = max(shfmax - 0.5, 0.0)
            else:
                ten = ten * 0.975 + 0.05
            max_dp = pl 
            if uhf:
                max_dp = np.sqrt(pla**2 + plb**2)
            converged = (niter > 4) and (abs(diff) < sellim) and ((not uhf and pl < pltest) or (uhf and max(pla,plb) < pltest))

            tscf = time.time()
            if self.verbose >= 4:
                print(f"cycle {niter:3d}: E_total = {ee:.12f} eV  Heat of Formation = {escf:.12f} Kcal/mol  dE = {diff:.3e}  dP_rms = {max_dp:.3e}  time = {tscf - tstart:.3f} s")
                
            if converged:
                # ===== 关掉收敛辅助，并重建“干净”的 Fock（无 level-shift/扰动）=====
                shift  = 0.0
                shiftb = 0.0

                # α-Fock: F = H (+ 两电子项)
                copy_vec(mol.f, mol.h)
                fock2(mol.f, mol.p, mol.pa, mol.w, mol.w, mol.wk if mol.wk is not None else mol.w,
                        mol.numat, mol.nfirst, mol.nlast, mode=2)

                # β-Fock（UHF）
                if uhf:
                    copy_vec(mol.fb, mol.h)
                    fock2(mol.fb, mol.p, mol.pb, mol.w, mol.w, mol.wk if mol.wk is not None else mol.w,
                            mol.numat, mol.nfirst, mol.nlast, 2)

                # ===== 用“干净”的 Fock 计算最终能量 =====
                ee = helect(norbs, mol.pa, mol.h, mol.f)
                if uhf:
                    ee += helect(norbs, mol.pb, mol.h, mol.fb)
                else:
                    ee *= 2.0

                # 最终精确对角化（与 Fortran 的 600 段一致）
                C, evals = eigenvectors_LAPACK_from_packed(mol.f, norbs)
                mol.c[:] = C
                mol.eigs[:] = evals

                if uhf:
                    Cb, evalsb = eigenvectors_LAPACK_from_packed(mol.fb, norbs)
                    mol.cb[:] = Cb
                    mol.eigb[:] = evalsb
                if self.verbose >= 0:
                    print('Converged result:')
                    print(f"E_total = {ee:.12f} eV  Heat of Formation = {escf:.12f} Kcal/mol")
                    print(f"HOMO and LUMO : {evals[ihomo]:.3e}  {evals[ihomo+1]:.3e} eV")
                
                escf  = (ee + mol.enuc) * EV2KCALMOL + mol.atheat + scorr
                ee_total = ee 
                return ee_total, escf
            eold = escf
            pold = mol.p.copy()
            paold = mol.pa.copy()
            pbold = mol.pb.copy()
        raise RuntimeError("SCF not converged")
    
    def _cnvg_mix(self, pnew: np.ndarray,
                p: np.ndarray,
                p1: np.ndarray,
                niter: int,
                uhf: bool ):
        """
        Fortran cnvg 的等价移植（packed 下三角 0-based）。
        就地更新 pnew、p、p1，并返回 pl（最大对角变化）。
        """
        print('CALL cnvg_mix')
        norbs = self.mol.norbs
        mpack = self.mol.mpack # norbs * (norbs + 1) // 2
        assert pnew.shape == p.shape == (mpack,)
        assert p1.shape == (norbs,)

        occ_cap = 1.0 if uhf else 2.0
        pl = 0.0

        # 阻尼日程
        damp = 1.0e10
        if niter > 3:
            damp = 5.0e-2

        extrap = (niter % 3) != 0
        faca = 0.0
        facb = 0.0

        # ---- 第一趟：只扫对角，测差、算 fac、并把 p 对角覆盖为 pnew，对角历史写入 p1 ----
        sum1 = 0.0
        for i in range(norbs):
            posd = i*(i+1)//2 + i
            a = float(pnew[posd])
            sum1 += a
            sa = abs(a - p[posd])
            if sa > pl:
                pl = sa
            if not extrap:
                faca += sa*sa
                facb += (a - 2.0*p[posd] + p1[i])**2
            p1[i] = p[posd]   # 旧对角入历史
            p[posd] = a       # 接受新对角

        fac = 0.0
        if facb > 1.0e-10 and faca < 100.0*facb:
            fac = np.sqrt(faca/facb)

        # ---- 第二趟：逐行先非对角，再对角（阻尼、截断、剪裁），并同步写回 pnew ----
        sum2 = 0.0
        ie = 0  # 线性游标
        for i in range(norbs):
            base = i*(i+1)//2
            # off-diagonals
            for j in range(i):
                pos = base + j
                a = pnew[pos]
                val = a + fac*(a - p[pos])
                p[pos] = val
                pnew[pos] = val
                ie += 1
            # diagonal
            posd = base + i
            dval = p[posd]
            prev = p1[i]
            if abs(dval - prev) > damp:
                dval = prev + np.copysign(damp, dval - prev)
            else:
                dval = dval + fac*(dval - prev)
            dval = max(0.0, min(occ_cap, dval))
            p[posd] = dval
            pnew[posd] = dval
            sum2 += dval
            ie += 1

        # ---- 归一化：仅调对角，占据裁剪后维持总占据 ----
        sum0 = sum1
        while True:
            scale = (sum1 / sum2) if (sum2 > 1.0e-3) else 0.0
            sum1 = sum0
            if (sum2 < 1.0e-3) or (abs(scale - 1.0) < 1.0e-5):
                break
            sum2 = 0.0
            for i in range(norbs):
                posd = i*(i+1)//2 + i
                dval = p[posd]*scale + 1.0e-20
                dval = max(0.0, dval)
                if dval > occ_cap:
                    dval = occ_cap
                    sum1 -= occ_cap
                else:
                    sum2 += dval
                p[posd] = dval
                pnew[posd] = dval

        return float(pl)    
    
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
                    E_minus = self._dhc_energy(pdi, padi, pbdi, cdi,
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
        return grad

    def pm6_forces(self) -> np.ndarray:
        """受力 = -梯度"""
        return - self.pm6_dcart_gradient(prec=True)