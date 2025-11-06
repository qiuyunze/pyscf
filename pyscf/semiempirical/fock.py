import numpy as np
from typing import Callable, Optional
import numpy as np
import time
from numba import njit

#### get fock matrix
# -------- lower triangle packed index（0-based）--------
@njit(inline='always')
def _idx_packed(i: int, j: int) -> int:
    if i < j:
        i, j = j, i
    return i * (i + 1) // 2 + j

# =========================
#  JAB: Coulomb 4x4×4x4 block
# =========================
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

_JAB_SUMA_IDXS = np.ascontiguousarray(np.asarray(_JAB_SUMA_IDXS, dtype=np.int64) - 1)
_JAB_SUMB_IDXS = np.ascontiguousarray(np.asarray(_JAB_SUMB_IDXS, dtype=np.int64) - 1)
_KAB_W_IDXS    = np.ascontiguousarray(np.asarray(_KAB_W_IDXS,    dtype=np.int64) - 1)

@njit(inline='always', fastmath=True, cache=True)
def _row_start_in_packed(i, j0):
    return i * (i + 1) // 2 + j0

@njit(fastmath=True, cache=True)
def jab_fast(ia: int, ja: int, pja: np.ndarray, pjb: np.ndarray,
             w_block: np.ndarray, f: np.ndarray) -> None:
    """
    4x4 J block
    """
    suma = np.empty(10, dtype=w_block.dtype)
    sumb = np.empty(10, dtype=w_block.dtype)

    # 10×16 dot
    for r in range(10):
        s = 0.0
        sb = 0.0
        for c in range(16):
            s  += w_block[_JAB_SUMA_IDXS[r, c]] * pja[c]
            sb += w_block[_JAB_SUMB_IDXS[r, c]] * pjb[c]
        suma[r] = s
        sumb[r] = sb

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
                       w10: np.ndarray,   
                       ptot: np.ndarray,
                       f: np.ndarray) -> None:
    ll = _idx_packed(ja, ja)   
    pt_ll = ptot[ll]

    sumdia = 0.0
    sumoff = 0.0
    k_off  = 0            

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
        wv = w10[k_off - 1]                     
        f[j1] += pt_ll * wv
        sumdia += ptot[j1] * wv

    f[ll] += 2.0 * sumoff + sumdia

@njit(fastmath=True, cache=True)
def j_heavy_light_fast(ia: int, ja: int,
                       w10: np.ndarray,   # w[kk:kk+10]
                       ptot: np.ndarray,
                       f: np.ndarray) -> None:
    ll = _idx_packed(ia, ia)   
    pt_ll = ptot[ll]

    sumdia = 0.0
    sumoff = 0.0
    k_off  = 0          

    for di in range(4):
        j1_base = _idx_packed(ja + di, ja) - 1

        if di > 0:
            for jcol in range(1, di + 1):
                pos = j1_base + jcol
                wv  = w10[jcol + k_off - 1] 
                f[pos] += pt_ll * wv
                sumoff += ptot[pos] * wv
            k_off  += di
            j1_base += di

        j1 = j1_base + 1
        k_off += 1
        wv = w10[k_off - 1]            
        f[j1] += pt_ll * wv
        sumdia += ptot[j1] * wv

    f[ll] += 2.0 * sumoff + sumdia

@njit(fastmath=True, cache=True)
def pk_from_p_packed_4x4_values(ia: int, ja: int, p: np.ndarray, out_pk16: np.ndarray):
    m = 0
    for r in range(4):
        ir = ia + r
        for c in range(4):
            jc = ja + c
            i = ir; j = jc
            if i < j:
                i, j = j, i
            out_pk16[m] = p[i * (i + 1) // 2 + j]  
            m += 1

@njit(fastmath=True, cache=True)
def kab_fast(ia: int, ja: int, pk16: np.ndarray,
             w_block: np.ndarray, f: np.ndarray) -> None:
    """
    4x4 K block
    """
    sums = np.empty(16, dtype=w_block.dtype)
    for r in range(16):
        s = 0.0
        for c in range(16):
            s += w_block[_KAB_W_IDXS[r, c]] * pk16[c]
        sums[r] = s

    if ia > ja:
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
    ncols = ib - ia + 1 
    p4 = np.empty(ncols, dtype=p.dtype)
    for jcol in range(ncols):
        p4[jcol] = p[_idx_packed(ia + jcol, ja)]

    k_acc = 0
    for i in range(ia, ib + 1):
        i1 = _idx_packed(i, ja)
        ssum = 0.0
        for t in range(ncols):  
            ssum += p4[t] * w10[_JINDEX_4x4[k_acc + t]]
        f[i1] -= ssum
        k_acc += ncols  

@njit(fastmath=True, cache=True)
def k_heavy_light_fast(ia: int, ja: int,
                       w10: np.ndarray,   # w[kk:kk+10]
                       p: np.ndarray,
                       f: np.ndarray, jindex: np.ndarray) -> None:
    k0 = _idx_packed(ia, ja)
    p4 = np.empty(4, dtype=p.dtype)
    for t in range(4):
        p4[t] = p[k0 + t]

    jacc = 0
    for off in range(4):
        ssum = 0.0
        for t in range(4):
            ssum += p4[t] * w10[jindex[jacc + t]]
        f[k0 + off] -= ssum

        jacc += 4

# -------- lower triangle packed index（0-based）--------
@njit(fastmath=True, cache=True)
def _unpack_full_from_packed_block(ptot: np.ndarray, ia: int, ib: int) -> np.ndarray:
    nloc = ib - ia + 1
    full = np.empty((nloc, nloc), dtype=ptot.dtype)
    for r in range(nloc):
        gr = ia + r
        for c in range(nloc):
            gc = ia + c
            full[r, c] = ptot[_idx_packed(gr, gc)]
    return full

@njit(fastmath=True, cache=True)
def _pack_from_full_block(full: np.ndarray) -> np.ndarray:
    nloc = full.shape[0]
    out  = np.empty(nloc*(nloc+1)//2, dtype=full.dtype)
    t = 0
    for i in range(nloc):
        out[t:t+i+1] = full[i, :i+1]
        t += i+1
    return out

@njit(fastmath=True, cache=True)
def _local_pairs_and_global_indices(ia: int, ib: int):
    nloc = ib - ia + 1
    ij_local_pairs = []
    f_gidx = np.empty(nloc*(nloc+1)//2, dtype=np.int64)
    t = 0
    for i0 in range(nloc):
        gi = ia + i0
        for j0 in range(i0+1):
            gj = ia + j0
            ij_local_pairs.append((i0, j0))
            f_gidx[t] = _idx_packed(gi, gj)
            t += 1
    return ij_local_pairs, f_gidx

@njit(fastmath=True, cache=True)
def fock1_np(
    f: np.ndarray,          # (mpack,)  
    ptot: np.ndarray,       # (mpack,)
    pa: np.ndarray,         # (mpack,)
    mpack: int,            
    w: np.ndarray,          # (ilim, ilim)  loc_packed×loc_packed
    kr: int,                # input kr
    ia: int,            
    ib: int,              
    ilim: int,              # ((ib-ia+1)*(ib-ia+2))//2
) -> int:
    """
    NumPy 0-based  fock1
    return updated kr
    """

    # nloc = ib - ia + 1  

    for i in range(ia, ib + 1):
        iw0 = i - ia                                # 0-based 
        for j in range(ia, i + 1):
            jw0 = j - ia                            # 0-based 
            # F global packed index
            ij = _idx_packed(i, j)
            # w local packed index
            ijw = _idx_packed(iw0, jw0)

            s = 0.0
            for k in range(ia, ib + 1):
                kw0 = k - ia
                for l in range(ia, ib + 1):
                    lw0 = l - ia

                    ijp = _idx_packed(k, l)
                    klw = _idx_packed(kw0, lw0)
                    ikw = _idx_packed(kw0, jw0)      # (k,j) in local packed
                    jlw = _idx_packed(lw0, iw0)      # (l,i) in local packed

                    s += ptot[ijp] * w[ijw, klw] - pa[ijp] * w[ikw, jlw]
            f[ij] += s

    return kr + ilim**2

def fock1_np_vec(
    f: np.ndarray,          # (mpack,)   packed triangle
    ptot: np.ndarray,       # (mpack,)
    pa: np.ndarray,         # (mpack,)
    mpack: int,            
    w: np.ndarray,          
    kr: int,                
    ia: int,                # AO start  index
    ib: int,                # AO end index
    ilim: int,              # ((ib-ia+1)*(ib-ia+2))//2
) -> int:
    '''
    Note: This function is not compatiable with numba.
    '''
    nloc = ib - ia + 1
    assert ilim == nloc*(nloc+1)//2

    ij_pairs, f_gidx = _local_pairs_and_global_indices(ia, ib)

    Ptot_full = _unpack_full_from_packed_block(ptot, ia, ib)  # (nloc,nloc)
    Pa_full   = _unpack_full_from_packed_block(pa,   ia, ib)  # (nloc,nloc)
    Pvec      = _pack_from_full_block(Ptot_full)              # (ilim,)

    # ---- weights for J (2 - δ_kl)  ----
    # diagnol=1，off-diagnol=2
    wgt = np.empty_like(Pvec)
    t = 0
    for i0 in range(nloc):
        for j0 in range(i0+1):
            wgt[t] = 1.0 if (i0 == j0) else 2.0
            t += 1
    sJ_vec = w @ (wgt * Pvec)   # (ilim,)

    # ---- einsum is used to calculate K ----
    A = np.empty((nloc, nloc), dtype=np.int64)
    B = np.empty((nloc, nloc), dtype=np.int64)
    for j0 in range(nloc):
        for k in range(nloc):
            A[j0, k] = _idx_packed(k, j0)
    for i0 in range(nloc):
        for l in range(nloc):
            B[i0, l] = _idx_packed(l, i0)

    W4 = w[A[None, :, :, None], B[:, None, None, :]]  # (i0, j0, k, l)
    sK_full = np.einsum('kl, ijkl -> ij', Pa_full, W4, optimize=True)  # (nloc, nloc)
    sK_vec  = _pack_from_full_block(sK_full)

    f[f_gidx] += (sJ_vec - sK_vec)

    return kr + ilim**2

@njit(fastmath=True, cache=True)
def infer_norbs_from_mpack(mpack: int) -> int:
    # n(n+1)/2 = mpack
    n = int((np.sqrt(8*mpack + 1) - 1) / 2)
    if n*(n+1)//2 != mpack:
        raise ValueError("inconsistent mpack and norbs")
    return n

# ----------- build jindex （ 0-based ）-----------
def build_jindex_4x4() -> np.ndarray:
    """
    Fortran jindex(m) = (ifact(ji)+ij)*10 + ifact(lk) + kl - 10   （1-based）
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
                    # 1-based
                    val_1based = (ifact1(ji) + ij) * 10 + ifact1(lk) + kl - 10
                    # Python 0-based 
                    jindex.append(val_1based - 1)
    return np.array(jindex, dtype=np.int64)  # shape=(256,)

# _JINDEX_4x4 = build_jindex_4x4()
_JINDEX_4x4 = np.ascontiguousarray(build_jindex_4x4(), dtype=np.int64)  # for numba acceleration


@njit(cache=True, fastmath=True)
def build_ifact(norbs):
    n = 18 if norbs < 18 else norbs
    i = np.arange(n, dtype=np.int64)
    return (i * (i + 1)) // 2

# ----------- Main implementation fock2（0-based NumPy）-----------
@njit(fastmath=True, cache=True)
def fock2_np(
    f: np.ndarray,              # (mpack,) lower triangle packed
    ptot: np.ndarray,           # (mpack,) total density
    p: np.ndarray,              # (mpack,) spin density
    w: np.ndarray,              # (n2elec,) 2-electron integral matrix（without PBC）
    wj: Optional[np.ndarray],   # (n2elec,) J integral (remain for PBE branch)
    wk: Optional[np.ndarray],   # (n2elec,) K integral (remain for PBE branch)
    numat: int,
    nfirst: np.ndarray,         # (numat,) （0-based, inclusive）
    nlast: np.ndarray,          # (numat,) （0-based, inclusive）
    mode=2,                     # fortran's default
    id_val = 0                  # 0 for non-periodic, !=0 for periodic/solid
):
    """
    Python/NumPy 0-based fock2。
    - f, ptot, p: packed lower triangle(0-based)
    - nfirst/nlast: 0-based、closed interval
    - id_val==0 for non-periodic, !=0 for periodic/solid
    """
    deriv  = False
    if numat < 0:
        deriv  = True
        numat = abs(numat)
    if numat == 0:
        return

    mpack = f.shape[0]
    norbs = infer_norbs_from_mpack(mpack)

    # ifact / i1fact（0-based）
    ifact = build_ifact(norbs) # np.fromfunction(lambda i: (i*(i+1))//2, (max(18, norbs),), dtype=np.int64)  

    max_per_atom = 81
    ptot2 = np.zeros((max(2, numat), max_per_atom), dtype=ptot.dtype)
    for i_atom in range(numat):
        ia = nfirst[i_atom]
        ib = nlast[i_atom]
        m = 0
        for j in range(ia, ib+1):
            for k in range(ia, ib+1):
                jk = _idx_packed(j,k)
                ptot2[i_atom, m] = ptot[jk]
                m += 1

    kk = 0  # current position of w

    lid = (id_val == 0)  # non-periodic
    one_e = 0 if id_val == 0 else -1  

    # di-atomic loop
    pk16 = np.empty(16, dtype=p.dtype)

    for ii in range(numat):
        ia = nfirst[ii]
        ib = nlast[ii]
        if deriv:
            iminus = ii - 0
        else:
            iminus = ii - one_e
        for jj in range(max(0, iminus+0)):  # 0..iminus
            
            ja = nfirst[jj]
            jb = nlast[jj]
            if lid:
                if (ib - ia) >= 6 or (jb - ja) >= 6:
                    kk = fockdorbs_np(ia, ib, ja, jb, f, p, ptot, w, kk, ifact[:norbs])

                elif (ib - ia) >= 3 and (jb - ja) >= 3:
                    # HEAVY–HEAVY
                    pja = ptot2[ii, :16]
                    pjb = ptot2[jj, :16]
                    w_block = w[kk: kk+100]               
                    jab_fast(ia, ja, pja, pjb, w_block, f)

                    pk_from_p_packed_4x4_values(ia, ja, p, pk16)
                    kab_fast(ia, ja, pk16, w_block, f)
                    kk += 100

                elif (ib - ia) >= 3 and (ja == jb):
                    # LIGHT–HEAVY → J + K
                    w_block10 = w[kk: kk+10]
                    j_light_heavy_fast(ia, ja, w_block10, ptot, f)
                    k_light_heavy_fast(ia, ja, ib, w_block10, p, f)
                    kk += 10
                elif (jb - ja) >= 3 and (ia == ib):
                    # HEAVY–LIGHT → J + K
                    w_block10 = w[kk: kk+10]
                    j_heavy_light_fast(ia, ja, w_block10, ptot, f)
                    k_heavy_light_fast(ia, ja, w_block10, p, f, _JINDEX_4x4)
                    kk += 10    
                
                elif (jb == ja) and (ia == ib):
                    # LIGHT–LIGHT (H-H)
                    i1 = _idx_packed(ia, ia)
                    j1 = _idx_packed(ja, ja)
                    ij = _idx_packed(ia, ja)

                    a = w[kk]  
                    f[i1] += ptot[j1] * a  
                    f[j1] += ptot[i1] * a
                    f[ij] -= p[ij] * a
                    kk += 1
        if mode == 2:
            i_blk = (ib - ia + 1) * (ib - ia + 2) // 2
            w_block = w[kk : kk + i_blk*i_blk].reshape((i_blk, i_blk))  
            kk = fock1_np(f, ptot, p, mpack, w_block, kk, ia, ib, i_blk)

@njit(fastmath=True, cache=True)
def fockdorbs_np(
    ia: int, ib: int, ja: int, jb: int,
    f: np.ndarray, p: np.ndarray, ptot: np.ndarray,
    w: np.ndarray, kk: int, ifact: np.ndarray 
) -> int:
    """
    NumPy 0-based fockdorbs.   
    return new kk.
    """
    if ia > ja:
        for i in range(ia, ib+1):
            ka = i * (i + 1) // 2
            aa = 2.0
            for j in range(ia, i+1):
                if i == j:
                    aa = 1.0
                kb = j * (j + 1) // 2
                ij = ka + j
                for k in range(ja, jb+1):
                    kc = k * (k + 1) // 2
                    ik = ka + k
                    jk = kb + k
                    bb = 2.0
                    for l in range(ja, k+1):
                        if k == l:
                            bb = 1.0
                        il = ka + l
                        jl = kb + l
                        kl = kc + l
                        a = w[kk]; kk += 1
                        f[ij] += bb * a * ptot[kl]
                        f[kl] += aa * a * ptot[ij]
                        aex = a * aa * bb * 0.25
                        f[ik] -= aex * p[jl]
                        f[il] -= aex * p[jk]
                        f[jk] -= aex * p[il]
                        f[jl] -= aex * p[ik]
    else:
        # stride: kref + (n2-1)*nn + (n1-1)
        kref = kk
        nn = (jb - ja + 1)
        nn = nn * (nn + 1) // 2
        n1 = 0
        for i in range(ja, jb+1):
            ka = i * (i + 1) // 2
            aa = 2.0
            for j in range(ja, i+1):
                n1 += 1
                if i == j:
                    aa = 1.0
                kb = j * (j + 1) // 2
                ij = ka + j
                n2 = 0
                for k in range(ia, ib+1):
                    kc = k * (k + 1) // 2
                    ik = ka + k
                    jk = kb + k
                    bb = 2.0
                    for l in range(ia, k+1):
                        n2 += 1
                        if k == l:
                            bb = 1.0
                        il = ka + l
                        jl = kb + l
                        kl = kc + l
                        a = w[kref + (n2-1)*nn + (n1-1)]
                        kk += 1
                        f[ij] += bb * a * ptot[kl]
                        f[kl] += aa * a * ptot[ij]
                        aex = a * aa * bb * 0.25
                        f[ik] -= aex * p[jl]
                        f[il] -= aex * p[jk]
                        f[jk] -= aex * p[il]
                        f[jl] -= aex * p[ik]
    return kk

def fockdorbs_np_vec(
    ia: int, ib: int, ja: int, jb: int,
    f: np.ndarray, p: np.ndarray, ptot: np.ndarray,
    w: np.ndarray, kk: int, ifact: np.ndarray  
) -> int:
    """
    Numpy fast by reducing the 4-fold loop, but failed with numba.
    """
    k_list, l_list, bb_list, kl_idx = [], [], [], []
    for k in range(ja, jb + 1):
        kc = k * (k + 1) // 2
        for l in range(ja, k + 1):
            k_list.append(k)
            l_list.append(l)
            bb_list.append(1.0 if k == l else 2.0)
            kl_idx.append(kc + l)

    k_vec = np.asarray(k_list, dtype=np.int64)  # (M,)
    l_vec = np.asarray(l_list, dtype=np.int64)  # (M,)
    bb_vec = np.asarray(bb_list, dtype=f.dtype)  # (M,)
    kl_vec = np.asarray(kl_idx, dtype=np.int64)  # (M,)
    M = kl_vec.size

    for i in range(ia, ib + 1):
        ka = i * (i + 1) // 2
        aa = 2.0
        for j in range(ia, i + 1):
            if i == j:
                aa = 1.0
            kb = j * (j + 1) // 2
            ij = ka + j

            w_vec = w[kk: kk + M]  # (M,)
            kk += M

            j_weight = bb_vec * w_vec
            f[ij] += np.dot(j_weight, ptot[kl_vec])

            if aa != 0.0:
                np.add.at(f, kl_vec, aa * ptot[ij] * w_vec)

            # aex = (aa * 0.25) * (bb[m] * w[m])
            aex_vec = (aa * 0.25) * j_weight

            ik_idx = ka + k_vec
            il_idx = ka + l_vec
            jk_idx = kb + k_vec
            jl_idx = kb + l_vec

            # f[ik] -= aex * p[jl]; f[il] -= aex * p[jk]; f[jk] -= aex * p[il]; f[jl] -= aex * p[ik]
            np.add.at(f, ik_idx, -aex_vec * p[jl_idx])
            np.add.at(f, il_idx, -aex_vec * p[jk_idx])
            np.add.at(f, jk_idx, -aex_vec * p[il_idx])
            np.add.at(f, jl_idx, -aex_vec * p[ik_idx])

    # --- ia <= ja ---
    if ia <= ja:
        k_list, l_list, bb_list, kl_idx = [], [], [], []
        for k in range(ia, ib + 1):
            kc = k * (k + 1) // 2
            for l in range(ja, k + 1):
                k_list.append(k)
                l_list.append(l)
                bb_list.append(1.0 if k == l else 2.0)
                kl_idx.append(kc + l)

        k_vec = np.asarray(k_list, dtype=np.int64)  # (M,)
        l_vec = np.asarray(l_list, dtype=np.int64)  # (M,)
        bb_vec = np.asarray(bb_list, dtype=f.dtype)  # (M,)
        kl_vec = np.asarray(kl_idx, dtype=np.int64)  # (M,)
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

                # --- f[ij] += sum_m (bb[m] * w[m]) * ptot[kl[m]]
                j_weight = bb_vec * w_vec
                f[ij] += np.dot(j_weight, ptot[kl_vec])

                # --- f[kl] += aa * ptot[ij] * w[m] 
                if aa != 0.0:
                    np.add.at(f, kl_vec, aa * ptot[ij] * w_vec)

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