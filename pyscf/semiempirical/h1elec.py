import numpy as np
from math import sqrt, exp
from bfn import bfn_np
from set_mopac import set_np

fact = np.array([
    1.0,               # 0!
    1.0,               # 1!
    2.0,               # 2!
    6.0,               # 3!
    24.0,              # 4!
    120.0,             # 5!
    720.0,             # 6!
    5040.0,            # 7!
    40320.0,           # 8!
    362880.0,          # 9!
    3628800.0,         # 10!
    39916800.0,        # 11!
    479001600.0,       # 12!
    6227020800.0,      # 13!
    8.71782912e10,     # 14!
    1.307674368e12,    # 15!
    2.092278989e13,    # 16!
    3.556874281e14     # 17!
], dtype=np.float64)
# ------------------------------
# ss: 通用三重积分（完全翻译）
# 依赖 bfn 
# ------------------------------
def ss_np(
    na:int, nb:int, la1:int, lb1:int, m1:int,
    ua:float, ub:float, r1:float, a0:float,
) -> float:
    """
    Fortran 'ss' 的等价实现。
    """
    m  = m1  - 1
    lb = lb1 - 1
    la = la1 - 1
    r = r1 / a0

    # binomial 系数表 bi(i,j): i,j=0..12
    bi = np.zeros((13,13), dtype=float)
    for i in range(13):
        bi[i,0] = 1.0
        bi[i,i] = 1.0
    for i in range(12):
        bi[i+1,1:i+1] = bi[i,1:i+1] + bi[i,0:i]

    # aff(la,m,i) 表（只用到 la<=2,m<=2,i<=2 的条目）
    aff = np.zeros((3,3,3), dtype=float)
    aff[0,0,0] = 1.0
    aff[1,0,0] = 1.0
    aff[1,1,0] = sqrt(0.5)
    aff[2,0,0] = 1.5
    aff[2,1,0] = sqrt(1.5)
    aff[2,2,0] = sqrt(0.375)
    aff[2,0,2] = -0.5  # 对应手册里的 C(2,0,1)

    # af(n)
    p = (ua + ub)*r*0.5
    b = (ua - ub)*r*0.5
    quo = 1.0/p
    af = np.empty(20, dtype=float)
    af[0] = quo*exp(-p)
    for n in range(1,20):
        af[n] = n*quo*af[n-1] + af[0]

    bf = np.asarray(bfn_np(b), dtype=float)

    lam1 = la - m
    lbm1 = lb - m

    total = 0.0
    for i in range(0, lam1+1, 2):
        ia = na + i - la
        ic = la - i - m
        for j in range(0, lbm1+1, 2):
            ib = nb + j - lb
            id_ = lb - j - m

            sum1 = 0.0
            iab = ia + ib

            for k1 in range(ia+1):
                for k2 in range(ib+1):
                    for k3 in range(ic+1):
                        for k4 in range(id_+1):
                            for k5 in range(m+1):
                                iaf = iab - k1 - k2 + k3 + k4 + 2*k5
                                for k6 in range(m+1):
                                    ibf = k1 + k2 + k3 + k4 + 2*k6
                                    parity = (m + k2 + k4 + k5 + k6) & 1
                                    sgn = 1.0 if parity == 0 else -1.0
                                    sum1 += ( bi[id_,k4]*bi[ic,k3]*bi[ib,k2]*bi[ia,k1]
                                              * bi[m,k5]*bi[m,k6]
                                              * sgn * af[iaf] * bf[ibf] )
            # total += sum1 * aff[la,m,i//2] * aff[lb,m,j//2]
            total += sum1 * aff[la, m, i] * aff[lb, m, j]

    val = ( total * (r**(na+nb+1)) * (ua**na)*(ub**nb) * 0.5
            * sqrt( ua*ub / (fact[2*na]*fact[2*nb]) * ((2*la+1)*(2*lb+1)) ) )
    return float(val)

# ------------------------------
# diat2: 主族小原子的快速路径
# 依赖 set 提供 {sa,sb,a[],b[], isp,ips}
# ------------------------------
def diat2_np(
    na: int, esa: float, epa: float,
    r12: float,
    nb: int, esb: float, epb: float,
    a0: float,
) -> np.ndarray:
    """
    返回 s(3,3,3)；其三个第三维分别对应 (sigma,pi,delta) 的组合（Fortran s(:,:,1/2/3)）。
    """
    # 形状与默认值
    s = np.zeros((3, 3, 3), dtype=float)
    rab = r12 / a0

    # Fortran 的 inmb / iii 表
    inmb = np.array([1, 0, 2, 2, 3, 4, 5, 6, 7, 0, 8, 8, 8, 9, 10, 11, 12], dtype=int)
    iii  = np.array([
        1,2,4, 2,4,4, 2,4,4,4, 2,4,4,4,4, 2,4,4,4,4,4, 2,4,4,4,4,4,4,
        3,5,5,5,5,5,5,6, 3,5,5,5,5,5,5,6,6,
        3,5,5,5,5,5,5,6,6,6,
        3,5,5,5,5,5,5,6,6,6,6,
        3,5,5,5,5,5,5,6,6,6,6,6
    ], dtype=int)

    jmax = max(inmb[na], inmb[nb])
    jmin = min(inmb[na], inmb[nb])
    nbond = (jmax*(jmax - 1))//2 + jmin
    ii = int(iii[nbond-1]) if nbond-1 < len(iii) and nbond-1 >= 0 else 1
    # 工具：调用 set_np，并把返回内容拆出来
    def call_set(ua, ub):
        sa, sb, avec, bvec, isp, ips = set_np(ua, ub, na, nb, rab, ii)
        # 转成 NumPy
        a = np.asarray(avec, dtype=float)  # Fortran 是 1-based，下文已照写同样的下标（注意使用时）
        b = np.asarray(bvec, dtype=float)
        return sa, sb, a, b, int(isp), int(ips)

    # 各 case 的表达式 —— 完全逐句翻译自 Fortran
    if ii not in (2,3,4,5,6):   # default == 1
        sa, sb, a, b, isp, ips = call_set(esa, esb)
        w = 0.25*sqrt((sa*sb*rab*rab)**3)
        # a(3)*b(1)-b(3)*a(1)  —— 注意 Fortran 下标从 1 起，这里用 0-based 取 a[2] 等
        s[0,0,0] = w*(a[2]*b[0] - b[2]*a[0])
        return s

    elif ii == 2:
        sa, sb, a, b, isp, ips = call_set(esa, esb)
        rab4 = (rab**4)*0.125
        w = sqrt(sa**3 * sb**5) * rab4
        s[0,0,0] = (1.0/sqrt(3.0)) * w * (a[3]*b[0] - b[3]*a[0] + a[2]*b[1] - b[2]*a[1])

        if na > 1:
            sa, sb, a, b, isp, ips = call_set(epa, esb)
        if nb > 1:
            sa, sb, a, b, isp, ips = call_set(esa, epb)
        w = sqrt(sa**3 * sb**5) * rab4
        # s(isp,ips,1)
        s[isp-1, ips-1, 0] = w*(a[2]*b[0]-b[2]*a[0] + a[3]*b[1]-b[3]*a[1])
        return s

    elif ii == 3:
        sa, sb, a, b, isp, ips = call_set(esa, esb)
        rab4 = (rab**5)*0.0625
        w = sqrt(sa**3 * sb**7 / 22.5) * rab4
        s[0,0,0] = w*(a[4]*b[0]-b[4]*a[0] + 2.0*(a[3]*b[1]-b[3]*a[1]))

        if na > 1:
            sa, sb, a, b, isp, ips = call_set(epa, esb)
        if nb > 1:
            sa, sb, a, b, isp, ips = call_set(esa, epb)
        w = sqrt(sa**3 * sb**7 / 7.5) * rab4
        s[isp-1, ips-1, 0] = w*( a[3]*(b[0]+b[2]) - b[3]*(a[0]+a[2]) + b[1]*(a[2]+a[4]) - a[1]*(b[2]+b[4]) )
        return s

    elif ii == 4:
        sa, sb, a, b, isp, ips = call_set(esa, esb)
        rab4 = (rab**5)*0.0625
        w = sqrt((sa*sb)**5) * rab4
        s[0,0,0] = w*(a[4]*b[0] + b[4]*a[0] - 2.0*a[2]*b[2]) / 3.0

        # s(isp,ips,1)
        sa, sb, a, b, isp, ips = call_set(esa, epb)
        # Fortran: if (na > nb) call set(epa, esb, ...)
        if na > nb:
            sa, sb, a, b, isp, ips = call_set(epa, esb)
        w = sqrt((sa*sb)**5) * rab4
        rt3 = 1.0/sqrt(3.0)
        d = a[3]*(b[0]-b[2]) - a[1]*(b[2]-b[4])
        e = b[3]*(a[0]-a[2]) - b[1]*(a[2]-a[4])
        s[isp-1, ips-1, 0] = w*rt3*(d + e)

        # s(ips,isp,1)
        sa, sb, a, b, isp, ips = call_set(epa, esb)
        if na > nb:
            sa, sb, a, b, isp, ips = call_set(esa, epb)
        w = sqrt((sa*sb)**5) * rab4
        d = a[3]*(b[0]-b[2]) - a[1]*(b[2]-b[4])
        e = b[3]*(a[0]-a[2]) - b[1]*(a[2]-a[4])
        s[ips-1, isp-1, 0] = w*rt3*(d - e)

        # s(2,2,1) & s(2,2,2)  (2==p通道)
        sa, sb, a, b, isp, ips = call_set(epa, epb)
        w = sqrt((sa*sb)**5) * rab4
        s[1,1,0] = -w*( b[2]*(a[4]+a[0]) - a[2]*(b[4]+b[0]) )
        s[1,1,1] = 0.5*w*( a[4]*(b[0]-b[2]) - b[4]*(a[0]-a[2]) - a[2]*b[0] + b[2]*a[0] )
        return s

    elif ii == 5:
        sa, sb, a, b, isp, ips = call_set(esa, esb)
        rab6 = (rab**6)*0.03125/sqrt(7.5)
        w = sqrt(sa**5 * sb**7) * rab6
        rt3 = 1.0/sqrt(3.0)
        s[0,0,0] = w*( a[5]*b[0] + a[4]*b[1] - 2.0*(a[3]*b[2] + a[2]*b[3]) + a[1]*b[4] + a[0]*b[5] )/3.0

        sa, sb, a, b, isp, ips = call_set(esa, epb)
        if na > nb:
            sa, sb, a, b, isp, ips = call_set(epa, esb)
        w = sqrt(sa**5 * sb**7) * rab6
        s[isp-1, ips-1, 0] = w*rt3*( a[5]*b[1] + a[4]*b[0] - 2.0*(a[3]*b[3] + a[2]*b[2]) + a[1]*b[5] + a[0]*b[4] )

        sa, sb, a, b, isp, ips = call_set(epa, esb)
        if na > nb:
            sa, sb, a, b, isp, ips = call_set(esa, epb)
        w = sqrt(sa**5 * sb**7) * rab6
        s[ips-1, isp-1, 0] = -w*rt3*( a[4]*(2.0*b[2]-b[0]) - b[4]*(2.0*a[2]-a[0]) - a[1]*(b[5]-2.0*b[3]) + b[1]*(a[5]-2.0*a[3]) )

        sa, sb, a, b, isp, ips = call_set(epa, epb)
        w = sqrt(sa**5 * sb**7) * rab6
        s[1,1,0] = -w*( b[3]*(a[0]+a[4]) - a[3]*(b[0]+b[4]) + b[2]*(a[1]+a[5]) - a[2]*(b[1]+b[5]) )
        s[1,1,1] = 0.5*w*( a[5]*(b[0]-b[2]) - b[5]*(a[0]-a[2]) + a[4]*(b[1]-b[3]) - b[4]*(a[1]-a[3]) - a[3]*b[0] + b[3]*a[0] - a[2]*b[1] + b[2]*a[1] )
        return s

    elif ii == 6:
        sa, sb, a, b, isp, ips = call_set(esa, esb)
        rab4 = (rab**7)/480.0
        w = sqrt((sa*sb)**7) * rab4
        rt3 = 1.0/sqrt(3.0)
        s[0,0,0] = w*( a[6]*b[0] - 3.0*(a[4]*b[2]-a[2]*b[4]) - a[0]*b[6] )/3.0

        sa, sb, a, b, isp, ips = call_set(esa, epb)
        if na > nb:
            sa, sb, a, b, isp, ips = call_set(epa, esb)
        w = sqrt((sa*sb)**7) * rab4
        d = a[5]*(b[0]-b[2]) - 2.0*a[3]*(b[2]-b[4]) + a[1]*(b[4]-b[6])
        e = b[5]*(a[0]-a[2]) - 2.0*b[3]*(a[2]-a[4]) + b[1]*(a[4]-a[6])
        s[isp-1, ips-1, 0] = w*rt3*(d - e)

        sa, sb, a, b, isp, ips = call_set(epa, esb)
        if na > nb:
            sa, sb, a, b, isp, ips = call_set(esa, epb)
        w = sqrt((sa*sb)**7) * rab4
        d = a[5]*(b[0]-b[2]) - 2.0*a[3]*(b[2]-b[4]) + a[1]*(b[4]-b[6])
        e = b[5]*(a[0]-a[2]) - 2.0*b[3]*(a[2]-a[4]) + b[1]*(a[4]-a[6])
        s[ips-1, isp-1, 0] = -w*rt3*((-d) - e)

        sa, sb, a, b, isp, ips = call_set(epa, epb)
        w = sqrt((sa*sb)**7) * rab4
        d = a[2]*(b[6]+b[2]+b[2]) - a[4]*(b[0]+b[4]+b[4]) - b[4]*a[0] + a[6]*b[2]
        s[1,1,0] = -w*d
        d = a[6]*(b[0]-b[2]) + b[6]*(a[0]-a[2])
        e = a[4]*(b[4]-b[2]-b[0]) + b[4]*(a[4]-a[2]-a[0]) + 2.0*a[2]*b[2]
        s[1,1,1] = 0.5*w*(d + e)
        return s

    return s

def coe_np(x2: float, y2: float, z2: float, norbi: int, norbj: int):
    """
    NumPy 版本的 coe。
    参数
    ----
    x2, y2, z2 : float
        原子 j 在 i 的局部坐标中的位移分量（与 Fortran 相同）。
    norbi, norbj : int
        原子 i / j 的 AO 个数（用于决定是否填充 p / d 项）。
    返回
    ----
    r : float
        两原子距离
    c : ndarray, shape (3, 5, 5)
        系数张量。等价于 Fortran 的 c(3,5,5)；Fortran 代码里通过线性索引 c(1..75) 赋值。
    """

    rt34 = 0.86602540378444  # sqrt(3)/2
    rt13 = 0.57735026918963  # 1/sqrt(3)

    # 距离与方向余弦
    xy2 = x2*x2 + y2*y2
    r = np.sqrt(xy2 + z2*z2)
    xy = np.sqrt(xy2)
    if xy >= 1.0e-10:
        ca = x2 / xy
        sa = y2 / xy
        cb = z2 / r if r != 0.0 else 0.0
        sb = xy / r if r != 0.0 else 0.0
    else:
        if z2 <= 0.0:
            if z2 != 0.0:
                ca = -1.0; cb = -1.0; sa = 0.0; sb = 0.0
            else:
                ca = 0.0; cb = 0.0; sa = 0.0; sb = 0.0
        else:  # z2 > 0
            ca = 1.0; cb = 1.0; sa = 0.0; sb = 0.0

    # 目标张量（等价 Fortran c(3,5,5)）
    c = np.zeros((3, 5, 5), dtype=float)

    # —— 辅助：把 Fortran 线性索引 c(n) (1-based) 映射到 c[i,j,k] (0-based, C-order 显式实现 Fortran 的列主序) ——
    # Fortran 线性索引公式（列主序）: n = 1 + (i-1) + 3*(j-1) + 3*5*(k-1)
    def setc(n1based: int, val: float):
        n0 = n1based - 1
        k = n0 // (3*5)
        rem = n0 % (3*5)
        j = rem // 3
        i = rem % 3
        c[i, j, k] = val

    # 与 Fortran 完全相同的赋值序列
    nij = max(int(norbi), int(norbj))

    # c(37) = 1.d0
    setc(37, 1.0)

    if nij >= 2:
        setc(56, ca*cb)
        setc(41, ca*sb)
        setc(26, -sa)
        setc(53, -sb)
        setc(38, cb)
        setc(23, 0.0)
        setc(50, sa*cb)
        setc(35, sa*sb)
        setc(20, ca)

        if nij >= 5:
            c2a = 2.0*ca*ca - 1.0
            c2b = 2.0*cb*cb - 1.0
            s2a = 2.0*sa*ca
            s2b = 2.0*sb*cb

            setc(75, c2a*cb*cb + 0.5*c2a*sb*sb)
            setc(60, 0.5*c2a*s2b)
            setc(45, rt34*c2a*sb*sb)
            setc(30, -s2a*sb)
            setc(15, -s2a*cb)

            setc(72, -0.5*ca*s2b)
            setc(57,  ca*c2b)
            setc(42,  rt34*ca*s2b)
            setc(27, -sa*cb)
            setc(12,  sa*sb)

            setc(69, rt13*1.5*sb*sb)
            setc(54, -rt34*s2b)
            setc(39, cb*cb - 0.5*sb*sb)

            setc(66, -0.5*sa*s2b)
            setc(51,  sa*c2b)
            setc(36,  rt34*sa*s2b)
            setc(21,  ca*cb)
            setc(6,  -ca*sb)

            setc(63, s2a*cb*cb + 0.5*s2a*sb*sb)
            setc(48, 0.5*s2a*s2b)
            setc(33, rt34*s2a*sb*sb)
            setc(18, c2a*sb)
            setc(3,  c2a*cb)

    return c
# ------------------------------
# diat: 主入口
# ------------------------------
def diat_np(
    ni: int,
    nj: int,
    xj: np.ndarray,                 # R_j - R_i 的向量 (3,)
    a0=0.529177210903,
    cutof1=10**2,                  # 距离平方截断
    natorb=None,             # 元素 -> 该原子AO数 (1/4/9)
    zs=None, zp=None, zd=None,
    npq=None,                # 形如 (107,3) 或更大：各元素的 n_pq（对应 s,p,d 列）
) -> np.ndarray:
    """
    返回 9x9 的二原子重叠 'di'（未裁剪，后续调用者可按 natorb 裁剪到 (n_i_orb,n_j_orb)）。
    """
    # ==== 常量、表 ====
    # Fortran: ival(3,5) column-major:
    # data ival/ 1,0,9, 1,3,8, 1,4,7, 1,2,6, 0,0,5/
    # i=1..3 (s,p,d), k=1..5 (m = -δ,-π,σ,π,δ) -> AO 索引(1..9)
    ival = np.array([
        [1, 1, 1, 1, 0],  # i=1 (s)
        [0, 3, 4, 2, 0],  # i=2 (p)
        [9, 8, 7, 6, 5],  # i=3 (d)
    ], dtype=int)

    # ---- 初始化 ----
    di = np.zeros((9, 9), dtype=float)

    x2, y2, z2 = float(xj[0]), float(xj[1]), float(xj[2])
    r2 = x2*x2 + y2*y2 + z2*z2

    # 取 s-主量子数; Fortran: pq1=npq(ni,1), pq2=npq(nj,1)（1-based列 -> s列）
    pq1 = int(npq[ni, 0])
    pq2 = int(npq[nj, 0])

    # 快速返回：零AO或超截断或太近
    if pq1 == 0 or pq2 == 0 or r2 >= cutof1: 
        return di
    if natorb[ni] == 0 or natorb[nj] == 0:
        return di

    # 系数 c(3,5,5) 和 “本征重叠分量” s(3,3,3) 的容器
    # s[:,:,0/1/2] 对应 Fortran 中的 s1/s2/s3
    c = np.zeros((3, 5, 5), dtype=float)
    s = np.zeros((3, 3, 3), dtype=float)

    # 系数矩阵 c <- coe(...)
    c[:] = np.asarray(coe_np(x2, y2, z2, int(natorb[ni]), int(natorb[nj])), dtype=float)

    # 原代码：r<0.001D0 直接返回（保持 0 矩阵）
    if r2 < 1.0e-6:
        return di

    # 计算是否使用 diat2 快速路径（与 Fortran 一致）
    # 原 Fortran：use_diat2(i) 对 1..17 的元素，natorb(i)<5 为 True；但 2 和 10 为 False。
    # 若你使用 0-based 元素编号且数组长度==107，下面逻辑能工作；否则请自行调节。
    def use_diat2_flag(Z):
        if Z < 17:
            flag = (natorb[Z] < 5) and (Z not in (1, 9))  # 注意：这里的 2 和 10 是“元素号”维度；如你做了 0 基，请对应调整
            return flag
        return False
    
    # ---- s 的生成：diat2 或 通用 ss 积分 ----
    if use_diat2_flag(ni) and use_diat2_flag(nj):   # 只对 1..17 元素有效
        s[:] = diat2_np(na=ni, esa=zs[ni], epa=zp[ni],
                        r12=sqrt(r2), nb=nj, esb=zs[nj], epb=zp[nj],
                        a0=a0)
    else:
        ul1 = np.array([zs[ni], zp[ni], max(zd[ni], 0.3)], dtype=float)
        ul2 = np.array([zs[nj], zp[nj], max(zd[nj], 0.3)], dtype=float)

        # i=1..ia, j=1..ib, k<=min(i,j)
        ia = min(int(npq[ni, 0]) + 1, 3)   # s-列 +1, capped by 3 (s/p/d)
        ib = min(int(npq[nj, 0]) + 1, 3)
        newk = min(ia-1, ib-1)             # 0-based内部计数
        for i in range(ia):                # 0..ia-1  -> Fortran i=1..ia
            pq1_i = int(npq[ni, i])        # npq(ni,i)（列：s/p/d）
            for j in range(ib):            # 0..ib-1
                pq2_j = int(npq[nj, j])
                nk1 = min(i, j) + 1        # Fortran 的 1..min(i,j)
                for k in range(nk1):       # k=0..nk1-1 -> Fortran kss=k+1
                    pi = max(pq1_i, i+1)   # Fortran: pi=max(pq1,iss)；iss=i
                    pj = max(pq2_j, j+1)
                    s[i, j, k] = ss_np(
                        pi, pj, i+1, j+1, k+1,
                        ul1[i], ul2[j],
                        r1=sqrt(r2), a0=a0
                    )

    # ---- 把 s & c 合成 di ----
    # s1/s2/s3
    s1, s2, s3 = s[:, :, 0], s[:, :, 1], s[:, :, 2]
    # c1..c5
    c1, c2, c3, c4, c5 = c[:, :, 0], c[:, :, 1], c[:, :, 2], c[:, :, 3], c[:, :, 4]

    ia = min(int(npq[ni, 0]) + 1, 3)
    ib = min(int(npq[nj, 0]) + 1, 3)

    for i in range(ia):               # i: 0(s),1(p),2(d)
        kmin, kmax = 3 - (i+1), 1 + (i+1)   # Fortran: k = 4-i .. 2+i  (1-based)，对应 0-based: 3-i .. 1+i
        for j in range(ib):           # j: 0(s),1(p),2(d)
            aa = -1.0 if (j == 1) else 1.0
            bb = -1.0 if (j == 2) else (1.0 if (j != 1) else 1.0)
            lmin, lmax = 3 - (j+1), 1 + (j+1)

            for k in range(kmin, kmax+1):
                for l in range(lmin, lmax+1):
                    ii = ival[i, k]   # AO 索引 (1..9 或 0)
                    jj = ival[j, l]
                    if ii == 0 or jj == 0:
                        continue
                    # 转 0-based
                    ii0 = ii - 1
                    jj0 = jj - 1
                    di[ii0, jj0] = (
                        di[ii0, jj0]
                        + s1[i, j]*(c3[i, k]*c3[j, l]) * aa
                        + s2[i, j]*(c4[i, k]*c4[j, l] + c2[i, k]*c2[j, l]) * bb
                        + s3[i, j]*(c5[i, k]*c5[j, l] + c1[i, k]*c1[j, l])
                    )
    return di


def h1elec_np(
    ni: int,
    nj: int,
    xi: np.ndarray,
    xj: np.ndarray,
    env = None,         # 初始化环境参数, 需要包含 natorb, betas, betap, betad, cutofs zs6 zp6 zd6 npq等
    # natorb=None,
    # betas=None,
    # betap=None,        # 元素 -> β_p
    # betad=None,        # 元素 -> β_d
    # cutofs=7.0**2,            # 距离平方截断（与 Fortran 相同语义）
    dummy_code = 101    # Fortran 的“dummy”元素号。如果你整体改 0-basis，这里改为 101
) -> np.ndarray:
    """
    NumPy 版本的一电子两中心块 H(ni,nj)。

    参数
    ----
    ni, nj : int
        两个原子的“元素/类型索引”。若你用 0-basis 原子号，请确保与
        natorb/betas/betap/betad 的索引一致。
    xi, xj : (3,) array
        两个原子的笛卡尔坐标（单位与上游一致，常见为 Å）。
    natorb, betas, betap, betad : ndarray
        与 Fortran 模块 parameters_C 中同义的参数数组（按元素索引取值）。
    cutofs : float
        距离平方截断阈值；Fortran 中与 rab(=|xi-xj|^2) 比较的是同一量纲。
    dummy_code : int
        “假原子”代码。Fortran 用 102；若你把原子号统一减 1，请改为 101。

    返回
    ----
    smat : (n_i_orb, n_j_orb) ndarray
        一电子矩阵块。
    """
    xi = np.asarray(xi, dtype=float).ravel()
    xj = np.asarray(xj, dtype=float).ravel()

    # 元素索引转化为0-basis
    ni = ni - 1
    nj = nj - 1
    # 距离平方
    rab = float(np.dot(xi - xj, xi - xj))
    # 截断：与 Fortran 保持一致（rab 是平方距离）
    if (rab > env.cutofs) or (rab > 3.24 and (ni == dummy_code or nj == dummy_code)):
        ni_orb = int(env.natorb[ni])
        nj_orb = int(env.natorb[nj])
        return np.zeros((ni_orb, nj_orb), dtype=float)

    # 方向向量 j 相对 i
    # xjuc = xi - xj
    xjuc = xj - xi

    # diat：方向相关的无量纲基矩阵（Fortran: call diat(ni,nj,xjuc,smat)）
    base = np.asarray(diat_np(ni, nj, xjuc, cutof1=env.cutofs, natorb=env.natorb, zs=env.zs6, zp=env.zp6, zd=env.zd6, npq=env.npq), dtype=float)
    # base 可能是 9x9；只取到实际 AO 维度
    ni_orb = int(env.natorb[ni])
    nj_orb = int(env.natorb[nj])
    smat = base[:ni_orb, :nj_orb].copy()

    # 构造 β/2（按 AO 排列：s, px, py, pz, d1..d5）
    # bi/bj 都长度 9，随后再切片到实际 AO 数
    bi = np.empty(9, dtype=float)
    bj = np.empty(9, dtype=float)

    bi[0] = 0.5 * env.betas6[ni]
    bi[1:4] = 0.5 * env.betap6[ni]
    bi[4:9] = 0.5 * env.betad6[ni]

    bj[0] = 0.5 * env.betas6[nj]
    bj[1:4] = 0.5 * env.betap6[nj]
    bj[4:9] = 0.5 * env.betad6[nj]

    # Fortran: do j=1,norbj; smat(:norbi,j) *= (bi(:norbi) + bj(j))
    # 向量化：对列广播
    smat *= (bi[:ni_orb, None] + bj[:nj_orb][None, :])

    return smat
