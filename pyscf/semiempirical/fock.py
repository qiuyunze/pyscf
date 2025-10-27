import numpy as np
from typing import Callable, Optional
import numpy as np
import time
from numba import njit
#### get fock matrix
# -------- 下三角 packed 索引（0-based）--------
'''def idx_packed(i: int, j: int) -> int:
    if i < j:
        i, j = j, i
    return i * (i + 1) // 2 + j'''
@njit(inline='always')
def _idx_packed(i: int, j: int) -> int:
    if i < j:
        i, j = j, i
    return i * (i + 1) // 2 + j

# =========================
#  JAB: Coulomb 4x4×4x4 块
# =========================
# 预先把 Fortran 中用到的 w(1..100) 索引表改成 0-based。
# suma 有 10 行，每行是 16 个 w 位置；sumb 也一样。
_JAB_SUMA_IDXS = [
    # suma(1)
    [1,11,31,61,11,21,41,71,31,41,51,81,61,71,81,91],
    # suma(2)
    [2,12,32,62,12,22,42,72,32,42,52,82,62,72,82,92],
    # suma(3)
    [3,13,33,63,13,23,43,73,33,43,53,83,63,73,83,93],
    # suma(4)
    [4,14,34,64,14,24,44,74,34,44,54,84,64,74,84,94],
    # suma(5)
    [5,15,35,65,15,25,45,75,35,45,55,85,65,75,85,95],
    # suma(6)
    [6,16,36,66,16,26,46,76,36,46,56,86,66,76,86,96],
    # suma(7)
    [7,17,37,67,17,27,47,77,37,47,57,87,67,77,87,97],
    # suma(8)
    [8,18,38,68,18,28,48,78,38,48,58,88,68,78,88,98],
    # suma(9)
    [9,19,39,69,19,29,49,79,39,49,59,89,69,79,89,99],
    # suma(10)
    [10,20,40,70,20,30,50,80,40,50,60,90,70,80,90,100],
]
_JAB_SUMB_IDXS = [
    # sumb(1)
    [1,2,4,7,2,3,5,8,4,5,6,9,7,8,9,10],
    # sumb(2)
    [11,12,14,17,12,13,15,18,14,15,16,19,17,18,19,20],
    # sumb(3)
    [21,22,24,27,22,23,25,28,24,25,26,29,27,28,29,30],
    # sumb(4)
    [31,32,34,37,32,33,35,38,34,35,36,39,37,38,39,40],
    # sumb(5)
    [41,42,44,47,42,43,45,48,44,45,46,49,47,48,49,50],
    # sumb(6)
    [51,52,54,57,52,53,55,58,54,55,56,59,57,58,59,60],
    # sumb(7)
    [61,62,64,67,62,63,65,68,64,65,66,69,67,68,69,70],
    # sumb(8)
    [71,72,74,77,72,73,75,78,74,75,76,79,77,78,79,80],
    # sumb(9)
    [81,82,84,87,82,83,85,88,84,85,86,89,87,88,89,90],
    # sumb(10)
    [91,92,94,97,92,93,95,98,94,95,96,99,97,98,99,100],
]

_KAB_W_IDXS = [
    # sum(1)
    [1,2,4,7,11,12,14,17,31,32,34,37,61,62,64,67],
    # sum(2)
    [2,3,5,8,12,13,15,18,32,33,35,38,62,63,65,68],
    # sum(3)
    [4,5,6,9,14,15,16,19,34,35,36,39,64,65,66,69],
    # sum(4)
    [7,8,9,10,17,18,19,20,37,38,39,40,67,68,69,70],
    # sum(5)
    [11,12,14,17,21,22,24,27,41,42,44,47,71,72,74,77],
    # sum(6)
    [12,13,15,18,22,23,25,28,42,43,45,48,72,73,75,78],
    # sum(7)
    [14,15,16,19,24,25,26,29,44,45,46,49,74,75,76,79],
    # sum(8)
    [17,18,19,20,27,28,29,30,47,48,49,50,77,78,79,80],
    # sum(9)
    [31,32,34,37,41,42,44,47,51,52,54,57,81,82,84,87],
    # sum(10)
    [32,33,35,38,42,43,45,48,52,53,55,58,82,83,85,88],
    # sum(11)
    [34,35,36,39,44,45,46,49,54,55,56,59,84,85,86,89],
    # sum(12)
    [37,38,39,40,47,48,49,50,57,58,59,60,87,88,89,90],
    # sum(13)
    [61,62,64,67,71,72,74,77,81,82,84,87,91,92,94,97],
    # sum(14)
    [62,63,65,68,72,73,75,78,82,83,85,88,92,93,95,98],
    # sum(15)
    [64,65,66,69,74,75,76,79,84,85,86,89,94,95,96,99],
    # sum(16)
    [67,68,69,70,77,78,79,80,87,88,89,90,97,98,99,100],
]
# 改为 0-based
# 确保索引表是 C 连续的 int64
_JAB_SUMA_IDXS = np.ascontiguousarray(np.asarray(_JAB_SUMA_IDXS, dtype=np.int64) - 1)
_JAB_SUMB_IDXS = np.ascontiguousarray(np.asarray(_JAB_SUMB_IDXS, dtype=np.int64) - 1)
_KAB_W_IDXS    = np.ascontiguousarray(np.asarray(_KAB_W_IDXS,    dtype=np.int64) - 1)

@njit(inline='always', fastmath=True, cache=True)
def _row_start_in_packed(i, j0):
    # 下三角 packed 的行起点（0-based）
    return i * (i + 1) // 2 + j0

@njit(fastmath=True, cache=True)
def jab_fast(ia: int, ja: int, pja: np.ndarray, pjb: np.ndarray,
             w_block: np.ndarray, f: np.ndarray) -> None:
    """
    4x4 J 块：无中间大临时，逐标量累加（pja/pjb 长度 16；w_block 长度 100）
    """
    suma = np.empty(10, dtype=w_block.dtype)
    sumb = np.empty(10, dtype=w_block.dtype)

    # 10×16 点积（避免 (10,16) 广播临时）
    for r in range(10):
        s = 0.0
        sb = 0.0
        for c in range(16):
            s  += w_block[_JAB_SUMA_IDXS[r, c]] * pja[c]
            sb += w_block[_JAB_SUMB_IDXS[r, c]] * pjb[c]
        suma[r] = s
        sumb[r] = sb

    # 写回 f：两块 4×4 块在 packed 下每行是连续片段
    idx = 0
    for di in range(4):
        start_i = _row_start_in_packed(ia + di, ia)
        start_j = _row_start_in_packed(ja + di, ja)
        ln = di + 1
        for t in range(ln):
            f[start_i + t] += sumb[idx + t]
            f[start_j + t] += suma[idx + t]
        idx += ln
@njit(fastmath=True, cache=True)
def j_light_heavy_fast(ia: int, ja: int,
                       w10: np.ndarray,   # 等于 w[kk:kk+10]
                       ptot: np.ndarray,
                       f: np.ndarray) -> None:
    """
    (ib-ia)>=3 且 (ja==jb) 的 J 段（左重右轻/你代码里称 LIGHT–HEAVY 的这一支）。
    完全复刻 Fortran 的 k_off 累加顺序，确保与慢版一致。
    """
    ll = _idx_packed(ja, ja)   # 轻侧（右侧）对角
    pt_ll = ptot[ll]

    sumdia = 0.0
    sumoff = 0.0
    k_off  = 0                 # 与 Fortran 同步：本地权重游标

    for di in range(4):
        j1_base = _idx_packed(ia + di, ia) - 1

        if di > 0:
            for jcol in range(1, di + 1):
                pos = j1_base + jcol
                wv  = w10[jcol + k_off - 1]       # ← w(kk + jcol + k_off - 1)
                f[pos] += pt_ll * wv
                sumoff += ptot[pos] * wv
            k_off  += di
            j1_base += di

        j1 = j1_base + 1
        k_off += 1
        wv = w10[k_off - 1]                        # 当行对角权重
        f[j1] += pt_ll * wv
        sumdia += ptot[j1] * wv

    f[ll] += 2.0 * sumoff + sumdia

@njit(fastmath=True, cache=True)
def j_heavy_light_fast(ia: int, ja: int,
                       w10: np.ndarray,   # 等于 w[kk:kk+10]
                       ptot: np.ndarray,
                       f: np.ndarray) -> None:
    """
    对应 (jb-ja)>=3 且 (ia==ib) 的 J 段（左轻右重；你代码里叫 HEAVY–LIGHT 的这一支）。
    完全复刻 Fortran 的 k_off 累加顺序，避免偏移推导误差。
    """
    ll = _idx_packed(ia, ia)   # 1×1 那边的对角
    pt_ll = ptot[ll]

    sumdia = 0.0
    sumoff = 0.0
    k_off  = 0                 # 对应 Fortran 里 kk 的“本地”偏移

    for di in range(4):
        j1_base = _idx_packed(ja + di, ja) - 1

        if di > 0:
            for jcol in range(1, di + 1):
                pos = j1_base + jcol
                wv  = w10[jcol + k_off - 1]   # ← 与 Fortran: w(kk + jcol + k_off - 1) 一致
                f[pos] += pt_ll * wv
                sumoff += ptot[pos] * wv
            k_off  += di
            j1_base += di

        j1 = j1_base + 1
        k_off += 1
        wv = w10[k_off - 1]                  # 当行对角
        f[j1] += pt_ll * wv
        sumdia += ptot[j1] * wv

    f[ll] += 2.0 * sumoff + sumdia

@njit(fastmath=True, cache=True)
def pk_from_p_packed_4x4_values(ia: int, ja: int, p: np.ndarray, out_pk16: np.ndarray):
    """从 packed p 取 (ia..ia+3, ja..ja+3) 的对称 4×4 子块，按行堆叠到 out_pk16[16]"""
    m = 0
    for r in range(4):
        ir = ia + r
        for c in range(4):
            jc = ja + c
            i = ir; j = jc
            if i < j:
                i, j = j, i
            out_pk16[m] = p[i * (i + 1) // 2 + j]  # 先写索引？不，是直接写值更好
            m += 1

@njit(fastmath=True, cache=True)
def kab_fast(ia: int, ja: int, pk16: np.ndarray,
             w_block: np.ndarray, f: np.ndarray) -> None:
    """
    4x4 K 块：pk16 长度 16（按行堆叠）
    """
    sums = np.empty(16, dtype=w_block.dtype)
    for r in range(16):
        s = 0.0
        for c in range(16):
            s += w_block[_KAB_W_IDXS[r, c]] * pk16[c]
        sums[r] = s

    if ia > ja:
        # 4 行，每行 4 个连续元素
        m = 0
        for r in range(4):
            j1 = ia + r
            start = _row_start_in_packed(j1, ja)
            f[start + 0] -= sums[m + 0]
            f[start + 1] -= sums[m + 1]
            f[start + 2] -= sums[m + 2]
            f[start + 3] -= sums[m + 3]
            m += 4
    else:
        # 16 个散点
        m = 0
        for r in range(4):
            j1 = ia + r
            for c in range(4):
                j2 = ja + c
                j3 = j2 * (j2 + 1) // 2 + j1
                f[j3] -= sums[m]
                m += 1

@njit(fastmath=True, cache=True)
def k_light_heavy_fast(ia: int, ja: int, ib: int,
                       w10: np.ndarray,   # w[kk:kk+10]
                       p: np.ndarray,
                       f: np.ndarray) -> None:
    """
    对应：(ib-ia)>=3 且 (ja==jb) 的 K 段（左轻右重）。
    原式：对每个 i 行，用 jindex 取 4 个权重，与 p(ia..ib, ja) 点积。
    """
    ncols = ib - ia + 1  # 应为 4
    # p(ia..ib, ja) 这 4 个 packed 元素与 i 无关，先备好
    p4 = np.empty(ncols, dtype=p.dtype)
    for jcol in range(ncols):
        p4[jcol] = p[_idx_packed(ia + jcol, ja)]

    k_acc = 0
    for i in range(ia, ib + 1):
        i1 = _idx_packed(i, ja)
        ssum = 0.0
        for t in range(ncols):  # 4 个权重
            ssum += p4[t] * w10[_JINDEX_4x4[k_acc + t]]
        f[i1] -= ssum
        k_acc += ncols  # 下一行偏移 +4

@njit(fastmath=True, cache=True)
def k_heavy_light_fast(ia: int, ja: int,
                       w10: np.ndarray,   # w[kk:kk+10]
                       p: np.ndarray,
                       f: np.ndarray, jindex: np.ndarray) -> None:
    """
    对应：(jb-ja)>=3 且 (ia==ib) 的 K 段（左重右轻）。
    原式：对 pos=k0..k0+3，每个用 jindex 取 4 个权重，与 p[k0:k0+4] 点积。
    """
    k0 = _idx_packed(ia, ja)
    # 连续 4 个 packed 元素
    p4 = np.empty(4, dtype=p.dtype)
    for t in range(4):
        p4[t] = p[k0 + t]

    jacc = 0
    for off in range(4):
        # 这 4 个权重的 jindex 切片
        ssum = 0.0
        for t in range(4):
            ssum += p4[t] * w10[jindex[jacc + t]]
        f[k0 + off] -= ssum

        jacc += 4

# -------- 下三角 packed 索引（0-based）--------
def _loc_pack(i: int, j: int) -> int:
    """局部下三角 (i>=j) 的 packed 索引（0-based）。"""
    if i < j:
        i, j = j, i
    return i * (i + 1) // 2 + j

def _unpack_full_from_packed_block(ptot: np.ndarray, ia: int, ib: int) -> np.ndarray:
    """从全局 packed ptot 提取局部块 ia..ib 的 Full 矩阵 (nloc x nloc)。"""
    nloc = ib - ia + 1
    full = np.empty((nloc, nloc), dtype=ptot.dtype)
    for r in range(nloc):
        gr = ia + r
        for c in range(nloc):
            gc = ia + c
            full[r, c] = ptot[_idx_packed(gr, gc)]
    return full

def _pack_from_full_block(full: np.ndarray) -> np.ndarray:
    """把 (nloc x nloc) 的全矩阵（对称）打包为下三角 packed 向量 (ilim,)。"""
    nloc = full.shape[0]
    out  = np.empty(nloc*(nloc+1)//2, dtype=full.dtype)
    t = 0
    for i in range(nloc):
        out[t:t+i+1] = full[i, :i+1]
        t += i+1
    return out

def _local_pairs_and_global_indices(ia: int, ib: int):
    """
    返回：
      - ij_local_pairs: 形如 [(i0,j0), ...]，i0>=j0 的局部下三角顺序（ilim 个）
      - f_gidx: 每个局部对 (i0,j0) 对应的全局 packed 索引
    """
    nloc = ib - ia + 1
    ij_local_pairs = []
    f_gidx = np.empty(nloc*(nloc+1)//2, dtype=int)
    t = 0
    for i0 in range(nloc):
        gi = ia + i0
        for j0 in range(i0+1):
            gj = ia + j0
            ij_local_pairs.append((i0, j0))
            f_gidx[t] = _idx_packed(gi, gj)
            t += 1
    return ij_local_pairs, f_gidx

def fock1_np(
    f: np.ndarray,          # (mpack,)  下三角packed，原位累加
    ptot: np.ndarray,       # (mpack,)
    pa: np.ndarray,         # (mpack,)
    mpack: int,             # 未用
    w: np.ndarray,          # (ilim, ilim)  局部packed×局部packed
    kr: int,                # 返回 kr + ilim**2
    ia: int,                # 原子AO块起点（含）
    ib: int,                # 原子AO块终点（含）
    ilim: int,              # ((ib-ia+1)*(ib-ia+2))//2
) -> int:
    nloc = ib - ia + 1
    assert ilim == nloc*(nloc+1)//2

    # 映射索引
    ij_pairs, f_gidx = _local_pairs_and_global_indices(ia, ib)

    # 提取局部密度
    Ptot_full = _unpack_full_from_packed_block(ptot, ia, ib)  # (nloc,nloc)
    Pa_full   = _unpack_full_from_packed_block(pa,   ia, ib)  # (nloc,nloc)
    Pvec      = _pack_from_full_block(Ptot_full)              # (ilim,)

    # ---- J 项：需要 (2 - δ_kl) 权重 ----
    # 构造 packed 顺序的权重向量：对角=1，非对角=2
    wgt = np.empty_like(Pvec)
    t = 0
    for i0 in range(nloc):
        for j0 in range(i0+1):
            wgt[t] = 1.0 if (i0 == j0) else 2.0
            t += 1
    sJ_vec = w @ (wgt * Pvec)   # (ilim,)

    # ---- K 项：按全矩阵做 einsum，与慢版等价 ----
    # A[j0,k] = pack(k,j0)；B[i0,l] = pack(l,i0)
    A = np.empty((nloc, nloc), dtype=int)
    B = np.empty((nloc, nloc), dtype=int)
    for j0 in range(nloc):
        for k in range(nloc):
            A[j0, k] = _loc_pack(k, j0)
    for i0 in range(nloc):
        for l in range(nloc):
            B[i0, l] = _loc_pack(l, i0)

    # W4[i0,j0,k,l] = w[ pack(k,j0), pack(l,i0) ]
    W4 = w[A[None, :, :, None], B[:, None, None, :]]  # (i0, j0, k, l)
    sK_full = np.einsum('kl, ijkl -> ij', Pa_full, W4, optimize=True)  # (nloc, nloc)
    sK_vec  = _pack_from_full_block(sK_full)

    # 写回
    f[f_gidx] += (sJ_vec - sK_vec)

    return kr + ilim**2

def infer_norbs_from_mpack(mpack: int) -> int:
    # n(n+1)/2 = mpack
    n = int((np.sqrt(8*mpack + 1) - 1) / 2)
    if n*(n+1)//2 != mpack:
        raise ValueError("mpack 与 norbs 不匹配")
    return n

# ----------- jindex 构造（与 Fortran 完全一致，但输出 0-based 偏移）-----------
def build_jindex_4x4() -> np.ndarray:
    """
    Fortran 中：
      jindex(m) = (ifact(ji)+ij)*10 + ifact(lk) + kl - 10   （1-based）
    这里复刻 1..4 的小 ifact，再把结果减 1 改为 0-based。
    """
    # 1-based ifact: ifact(n) = n*(n-1)/2
    def ifact1(n: int) -> int:
        return n * (n - 1) // 2

    jindex = []
    m = 0
    for i in range(1, 5):
        for j in range(1, 5):
            ij = min(i, j)
            ji = i + j - ij
            for k in range(1, 5):
                for l in range(1, 5):
                    m += 1
                    kl = min(k, l)
                    lk = k + l - kl
                    # 1-based公式
                    val_1based = (ifact1(ji) + ij) * 10 + ifact1(lk) + kl - 10
                    # Python 0-based 偏移（相对于 10 长度块）
                    jindex.append(val_1based - 1)
    return np.array(jindex, dtype=int)  # shape=(256,)

# _JINDEX_4x4 = build_jindex_4x4()
_JINDEX_4x4 = np.ascontiguousarray(build_jindex_4x4(), dtype=np.int64) 

# ----------- 主实现：fock2（0-based NumPy）-----------
def fock2_np(
    f: np.ndarray,              # (mpack,) 下三角packed，原位累加
    ptot: np.ndarray,           # (mpack,) 总密度
    p: np.ndarray,              # (mpack,) 自旋密度（α或β）
    w: np.ndarray,              # (n2elec,) 两电子积分线性数组（非PBC分支）
    wj: Optional[np.ndarray],   # (n2elec,) J 积分（PBC分支）
    wk: Optional[np.ndarray],   # (n2elec,) K 积分（PBC分支）
    numat: int,
    nfirst: np.ndarray,         # (numat,) 每原子的起始 AO 索引（0-based, inclusive）
    nlast: np.ndarray,          # (numat,) 每原子的结束 AO 索引（0-based, inclusive）
    mode=2,                     # fortran's default
    id_val = 0                  # 0 for non-periodic, !=0 for periodic/solid
):
    """
    Python/NumPy 0-based 版的 fock2。
    - f, ptot, p：packed 下三角（0-based）
    - nfirst/nlast：0-based、闭区间
    - 若 id_val==0 走“非周期”路径；否则走 wj/wk 的 PBC 路径
    - 需要时调用 jab/kab/fock1/addfck（请从外面注入对应的实现）
    """
    deriv  = False
    if numat < 0:
        deriv  = True
        numat = abs(numat)
    if numat == 0:
        return

    mpack = f.shape[0]
    norbs = infer_norbs_from_mpack(mpack)

    # ifact / i1fact（0-based版）
    ifact = np.fromfunction(lambda i: (i*(i+1))//2, (max(18, norbs),), dtype=int)
    i1fact = ifact + np.arange(max(18, norbs), dtype=int)

    # 预抽取每个原子的 4x4 交叉子块到 ptot2（最多81个元素；与 Fortran 保持）
    # 注意：Fortran把每原子的 (ib-ia+1)^2 下三角块打平成 16 或 81；这里使用同样大小的容器。
    # 为保持一致，这里我们统一存 81（9x9上三角+下三角组合），实际仅用到 16/10/1 个时取切片。
    max_per_atom = 81
    ptot2 = np.zeros((max(2, numat), max_per_atom), dtype=ptot.dtype)
    for i_atom in range(numat):
        ia = nfirst[i_atom]
        ib = nlast[i_atom]
        m = 0
        for j in range(ia, ib+1):
            for k in range(ia, ib+1):
                jk = _idx_packed(max(j, k), min(j, k))
                ptot2[i_atom, m] = ptot[jk]
                m += 1

    kk = 0  # w 的游标

    lid = (id_val == 0)  # 非周期
    one_e = 0 if id_val == 0 else -1  # 对应 Fortran 的 ione，用于“deriv”分支时的 ii-1 / ii-ione 的差别

    # 主双原子循环
    pk16 = np.empty(16, dtype=p.dtype)

    for ii in range(numat):
        # t0 = time.time()
        ia = nfirst[ii]
        ib = nlast[ii]
        # 这里省略 Fortran 的 deriv/numat<0 分支；若你需要，传个 deriv 标志进来再细分
        if deriv:
            iminus = ii - 0
        else:
            iminus = ii - one_e
        for jj in range(max(0, iminus+0)):  # 0..iminus
            
            ja = nfirst[jj]
            jb = nlast[jj]
            if lid:
                # 非周期路径
                if (ib - ia) >= 6 or (jb - ja) >= 6:
                    # 大轨道子块 → 调 fockdorbs
                    kk = fockdorbs_np(ia, ib, ja, jb, f, p, ptot, w, kk, ifact[:norbs])

                elif (ib - ia) >= 3 and (jb - ja) >= 3:
                    # HEAVY–HEAVY
                    # —— COULOMB (J) via jab ——（取 ii 与 jj 的 16 个元素）
                    # —— COULOMB (J) via jab（直接视图即可，无需 copy）
                    pja = ptot2[ii, :16]
                    pjb = ptot2[jj, :16]
                    w_block = w[kk: kk+100]               # 视图，无拷贝
                    jab_fast(ia, ja, pja, pjb, w_block, f)

                    # —— EXCHANGE (K) via kab（一次性取 4×4 的 16 项）
                    # 建议在 fock2_np 外层准备好：pk16 = np.empty(16, dtype=p.dtype) 供复用（你已有）
                    pk_from_p_packed_4x4_values(ia, ja, p, pk16)
                    kab_fast(ia, ja, pk16, w_block, f)
                    kk += 100

                elif (ib - ia) >= 3 and (ja == jb):
                    # LIGHT–HEAVY（左轻右重）→ J + K
                    # —— COULOMB J 部分（与 Fortran 同步的逐项累加）——
                    '''
                    sumdia = 0.0
                    sumoff = 0.0
                    ll = _idx_packed(ja, ja)  # i1fact(ja) 的 0-based 等价：i*(i+1)//2 + i
                    k_off = 0
                    for di in range(0, 4):
                        j1_base = _idx_packed(ia + di, ia) - 1  # 注意下面先 +1 再用
                        if di > 0:
                            for jcol in range(1, di+1):
                                pos = j1_base + jcol
                                f[pos] += ptot[ll] * w[kk + jcol + k_off - 1]
                                sumoff += ptot[pos] * w[kk + jcol + k_off - 1]
                            k_off += di
                            j1_base += di
                        j1 = j1_base + 1
                        k_off += 1
                        f[j1] += ptot[ll] * w[kk + k_off - 1]
                        sumdia += ptot[j1] * w[kk + k_off - 1]
                    f[ll] += 2.0 * sumoff + sumdia

                    # —— EXCHANGE K 部分 ——（用 jindex 查 10 元块）
                    k_acc = 0
                    for i in range(ia, ib+1):
                        i1 = _idx_packed(i, ja)
                        ssum = 0.0
                        ncols = ib - ia + 1
                        for jcol in range(1, ncols+1):
                            # p(ifact(j-1+ia)+ja) → p(idx_packed( (j-1+ia), ja ))
                            ppos = _idx_packed((jcol - 1 + ia), ja)
                            ssum += p[ppos] * w[kk + _JINDEX_4x4[jcol - 1 + k_acc]]
                        k_acc += ncols
                        f[i1] -= ssum
                    '''
                    w_block10 = w[kk: kk+10]
                    j_light_heavy_fast(ia, ja, w_block10, ptot, f)
                    # —— EXCHANGE K ——（同一个 10 块）
                    k_light_heavy_fast(ia, ja, ib, w_block10, p, f)
                    
                    kk += 10
                elif (jb - ja) >= 3 and (ia == ib):
                    # HEAVY–LIGHT（左重右轻）→ J + K
                    '''sumdia = 0.0
                    sumoff = 0.0
                    ll = _idx_packed(ia, ia)
                    k_off = 0
                    for di in range(0, 4):
                        j1_base = _idx_packed(ja + di, ja) - 1
                        if di > 0:
                            for jcol in range(1, di+1):
                                pos = j1_base + jcol
                                f[pos] += ptot[ll] * w[kk + jcol + k_off-1]
                                sumoff += ptot[pos] * w[kk + jcol + k_off-1]
                            k_off += di
                            j1_base += di
                        j1 = j1_base + 1
                        k_off += 1
                        f[j1] += ptot[ll] * w[kk + k_off-1]
                        sumdia += ptot[j1] * w[kk + k_off-1]
                    f[ll] += 2.0 * sumoff + sumdia
                    # —— EXCHANGE（4 列向量 * 10 元素块）——
                    k0 = _idx_packed(ia, ja)
                    jacc = 0
                    for pos in range(k0, k0 + 4):
                        ssum = 0.0
                        for lcol in range(1, 5):
                            ssum += p[k0 + (lcol - 1)] * w[kk + _JINDEX_4x4[lcol - 1 + jacc]]
                        jacc += 4
                        print('pos', pos)
                        f[pos] -= ssum'''
                    w_block10 = w[kk: kk+10]
                    j_heavy_light_fast(ia, ja, w_block10, ptot, f)
                    k_heavy_light_fast(ia, ja, w_block10, p, f, _JINDEX_4x4)
                    kk += 10    
                    '''
                    print('heavy-light')
                    w_block10 = w[kk: kk+10]
                    j_heavy_light_fast(ia, ja, w_block10, ptot, f)
                    k_heavy_light_fast(ia, ja, w_block10, p, f, _JINDEX_4x4)
                    kk += 10
                    '''
                
                elif (jb == ja) and (ia == ib):
                    # LIGHT–LIGHT（1 积分）
                    i1 = _idx_packed(ia, ia)
                    j1 = _idx_packed(ja, ja)
                    ij = _idx_packed(max(i1, j1), min(i1, j1))  # 注意：这里原式是 i1 + (ja-ia)；packed 下更安全
                    # 但 Fortran 的 ij = i1 + ja - ia（因两者都是对角位置），等价于 idx_packed(ia, ja)
                    ij = _idx_packed(ia, ja)

                    a = w[kk]  # Fortran: w(kk+1)
                    f[i1] += ptot[j1] * a
                    f[j1] += ptot[i1] * a
                    f[ij] -= p[ij] * a
                    kk += 1
                else:
                    # 不常见分支（理论上已覆盖四种组合）
                    pass
            else:
                # PBC / 周期路径：使用 wj / wk
                pass
        # t1 = time.time()
        # print('Fock2 pair time:%.4f s'%(t1-t0))
        if mode == 2:
            i_blk = (ib - ia + 1) * (ib - ia + 2) // 2
            w_block = w[kk : kk + i_blk*i_blk].reshape((i_blk, i_blk), order='F')
            kk = fock1_np(f, ptot, p, mpack, w_block, kk, ia, ib, i_blk)
        # t2 = time.time()
        # print('Fock2 self time:%.4f s'%(t2-t1))
# ----------- 子程序：fockdorbs（大子块，用 w 的块矩阵取值）-----------
def fockdorbs_np(
    ia: int, ib: int, ja: int, jb: int,
    f: np.ndarray, p: np.ndarray, ptot: np.ndarray,
    w: np.ndarray, kk: int, ifact: np.ndarray  # ifact 未用
) -> int:
    """
    完全等价的 NumPy 快速版本，支持 ia > ja 和 ia <= ja 两个分支。
    返回新的 kk（w 的游标）。
    """
    # --- 预生成 k,l 半三角对索引，并计算 bb 值（1.0 或 2.0） ---
    k_list, l_list, bb_list, kl_idx = [], [], [], []
    for k in range(ja, jb + 1):
        kc = k * (k + 1) // 2
        for l in range(ja, k + 1):
            k_list.append(k)
            l_list.append(l)
            bb_list.append(1.0 if k == l else 2.0)
            kl_idx.append(kc + l)

    k_vec = np.asarray(k_list, dtype=int)  # (M,)
    l_vec = np.asarray(l_list, dtype=int)  # (M,)
    bb_vec = np.asarray(bb_list, dtype=f.dtype)  # (M,)
    kl_vec = np.asarray(kl_idx, dtype=int)  # (M,)
    M = kl_vec.size

    # --- 主双原子循环 ---
    for i in range(ia, ib + 1):
        ka = i * (i + 1) // 2
        aa = 2.0
        for j in range(ia, i + 1):
            if i == j:
                aa = 1.0
            kb = j * (j + 1) // 2
            ij = ka + j

            # 对每个 (i,j) 读取对应的 w 子段，并推进 kk
            w_vec = w[kk: kk + M]  # (M,)
            kk += M

            # --- J 部分：f[ij] += sum_m (bb[m] * w[m]) * ptot[kl[m]]
            j_weight = bb_vec * w_vec
            f[ij] += np.dot(j_weight, ptot[kl_vec])

            # --- J 部分：f[kl] += aa * ptot[ij] * w[m] （散加）
            if aa != 0.0:
                np.add.at(f, kl_vec, aa * ptot[ij] * w_vec)

            # --- K 部分：交换部分四路散加 ---
            # aex = (aa * 0.25) * (bb[m] * w[m])
            aex_vec = (aa * 0.25) * j_weight

            # 构造交换所需的索引：ik = ka + k, il = ka + l, jk = kb + k, jl = kb + l
            ik_idx = ka + k_vec
            il_idx = ka + l_vec
            jk_idx = kb + k_vec
            jl_idx = kb + l_vec

            # 四路散加：f[ik] -= aex * p[jl]; f[il] -= aex * p[jk]; f[jk] -= aex * p[il]; f[jl] -= aex * p[ik]
            np.add.at(f, ik_idx, -aex_vec * p[jl_idx])
            np.add.at(f, il_idx, -aex_vec * p[jk_idx])
            np.add.at(f, jk_idx, -aex_vec * p[il_idx])
            np.add.at(f, jl_idx, -aex_vec * p[ik_idx])

    # --- 对于 ia <= ja 的情况 ---
    if ia <= ja:
        k_list, l_list, bb_list, kl_idx = [], [], [], []
        for k in range(ia, ib + 1):
            kc = k * (k + 1) // 2
            for l in range(ja, k + 1):
                k_list.append(k)
                l_list.append(l)
                bb_list.append(1.0 if k == l else 2.0)
                kl_idx.append(kc + l)

        k_vec = np.asarray(k_list, dtype=int)  # (M,)
        l_vec = np.asarray(l_list, dtype=int)  # (M,)
        bb_vec = np.asarray(bb_list, dtype=f.dtype)  # (M,)
        kl_vec = np.asarray(kl_idx, dtype=int)  # (M,)
        M = kl_vec.size

        for i in range(ia, ib + 1):
            ka = i * (i + 1) // 2
            aa = 2.0
            for j in range(ja, ib + 1):
                if i == j:
                    aa = 1.0
                kb = j * (j + 1) // 2
                ij = ka + j

                w_vec = w[kk: kk + M]  # (M,)
                kk += M

                # --- J 部分：f[ij] += sum_m (bb[m] * w[m]) * ptot[kl[m]]
                j_weight = bb_vec * w_vec
                f[ij] += np.dot(j_weight, ptot[kl_vec])

                # --- J 部分：f[kl] += aa * ptot[ij] * w[m] （散加）
                if aa != 0.0:
                    np.add.at(f, kl_vec, aa * ptot[ij] * w_vec)

                # --- K 部分：交换部分四路散加 ---
                aex_vec = (aa * 0.25) * j_weight

                ik_idx = ka + k_vec
                il_idx = ka + l_vec
                jk_idx = kb + k_vec
                jl_idx = kb + l_vec

                np.add.at(f, ik_idx, -aex_vec * p[jl_idx])
                np.add.at(f, il_idx, -aex_vec * p[jk_idx])
                np.add.at(f, jk_idx, -aex_vec * p[il_idx])
                np.add.at(f, jl_idx, -aex_vec * p[ik_idx])

    return kk