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

def ss_np(
    na:int, nb:int, la1:int, lb1:int, m1:int,
    ua:float, ub:float, r1:float, a0:float,
) -> float:
    """
    Fortran 'ss' 
    """
    m  = m1  - 1
    lb = lb1 - 1
    la = la1 - 1
    r = r1 / a0

    # binomial coefficient table bi(i,j): i,j=0..12
    bi = np.zeros((13,13), dtype=float)
    for i in range(13):
        bi[i,0] = 1.0
        bi[i,i] = 1.0
    for i in range(12):
        bi[i+1,1:i+1] = bi[i,1:i+1] + bi[i,0:i]

    # aff(la,m,i) table
    aff = np.zeros((3,3,3), dtype=float)
    aff[0,0,0] = 1.0
    aff[1,0,0] = 1.0
    aff[1,1,0] = sqrt(0.5)
    aff[2,0,0] = 1.5
    aff[2,1,0] = sqrt(1.5)
    aff[2,2,0] = sqrt(0.375)
    aff[2,0,2] = -0.5  

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
# diat2: main-group atom
# set {sa,sb,a[],b[], isp,ips}
# ------------------------------
def diat2_np(
    na: int, esa: float, epa: float,
    r12: float,
    nb: int, esb: float, epb: float,
    a0: float,
) -> np.ndarray:
    """
    return s(3,3,3): (sigma,pi,delta)
    """
    s = np.zeros((3, 3, 3), dtype=float)
    rab = r12 / a0

    # Fortran  inmb / iii 
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

    def call_set(ua, ub):
        sa, sb, avec, bvec, isp, ips = set_np(ua, ub, na, nb, rab, ii)
        a = np.asarray(avec, dtype=float) 
        b = np.asarray(bvec, dtype=float)
        return sa, sb, a, b, int(isp), int(ips)

    if ii not in (2,3,4,5,6):   # default == 1
        sa, sb, a, b, isp, ips = call_set(esa, esb)
        w = 0.25*sqrt((sa*sb*rab*rab)**3)
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

        sa, sb, a, b, isp, ips = call_set(esa, epb)
        if na > nb:
            sa, sb, a, b, isp, ips = call_set(epa, esb)
        w = sqrt((sa*sb)**5) * rab4
        rt3 = 1.0/sqrt(3.0)
        d = a[3]*(b[0]-b[2]) - a[1]*(b[2]-b[4])
        e = b[3]*(a[0]-a[2]) - b[1]*(a[2]-a[4])
        s[isp-1, ips-1, 0] = w*rt3*(d + e)

        sa, sb, a, b, isp, ips = call_set(epa, esb)
        if na > nb:
            sa, sb, a, b, isp, ips = call_set(esa, epb)
        w = sqrt((sa*sb)**5) * rab4
        d = a[3]*(b[0]-b[2]) - a[1]*(b[2]-b[4])
        e = b[3]*(a[0]-a[2]) - b[1]*(a[2]-a[4])
        s[ips-1, isp-1, 0] = w*rt3*(d - e)

        # s(2,2,1) & s(2,2,2)  (2==p)
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
    NumPy  coe
    ----
    x2, y2, z2 : float
    norbi, norbj : int
         AO number of atom i / j
    return
    ----
    r : float
    c : ndarray, shape (3, 5, 5)
    """

    rt34 = 0.86602540378444  # sqrt(3)/2
    rt13 = 0.57735026918963  # 1/sqrt(3)

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

    c = np.zeros((3, 5, 5), dtype=float)

    def setc(n1based: int, val: float):
        n0 = n1based - 1
        k = n0 // (3*5)
        rem = n0 % (3*5)
        j = rem // 3
        i = rem % 3
        c[i, j, k] = val

    nij = max(int(norbi), int(norbj))

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
# diat: entrance
# ------------------------------
def diat_np(
    ni: int,
    nj: int,
    xj: np.ndarray,                 # R_j - R_i  (3,)
    a0=0.529177210903,
    cutof1=10**2,                  # squared distance cutoff
    natorb=None,            
    zs=None, zp=None, zd=None,
    npq=None,                
) -> np.ndarray:
    """
    return 9x9 'di'
    """
    # Fortran: ival(3,5) column-major:
    # data ival/ 1,0,9, 1,3,8, 1,4,7, 1,2,6, 0,0,5/
    # i=1..3 (s,p,d), k=1..5 (m = -δ,-π,σ,π,δ) -> AO 索引(1..9)
    ival = np.array([
        [1, 1, 1, 1, 0],  # i=1 (s)
        [0, 3, 4, 2, 0],  # i=2 (p)
        [9, 8, 7, 6, 5],  # i=3 (d)
    ], dtype=int)

    di = np.zeros((9, 9), dtype=float)

    x2, y2, z2 = float(xj[0]), float(xj[1]), float(xj[2])
    r2 = x2*x2 + y2*y2 + z2*z2

    pq1 = int(npq[ni, 0])
    pq2 = int(npq[nj, 0])

    if pq1 == 0 or pq2 == 0 or r2 >= cutof1: 
        return di
    if natorb[ni] == 0 or natorb[nj] == 0:
        return di

    c = np.zeros((3, 5, 5), dtype=float)
    s = np.zeros((3, 3, 3), dtype=float)

    c[:] = np.asarray(coe_np(x2, y2, z2, int(natorb[ni]), int(natorb[nj])), dtype=float)

    if r2 < 1.0e-6:
        return di

    def use_diat2_flag(Z):
        if Z < 17:
            flag = (natorb[Z] < 5) and (Z not in (1, 9))  
            return flag
        return False
    
    if use_diat2_flag(ni) and use_diat2_flag(nj):   # 1..17 element
        s[:] = diat2_np(na=ni, esa=zs[ni], epa=zp[ni],
                        r12=sqrt(r2), nb=nj, esb=zs[nj], epb=zp[nj],
                        a0=a0)
    else:
        ul1 = np.array([zs[ni], zp[ni], max(zd[ni], 0.3)], dtype=float)
        ul2 = np.array([zs[nj], zp[nj], max(zd[nj], 0.3)], dtype=float)

        # i=1..ia, j=1..ib, k<=min(i,j)
        ia = min(int(npq[ni, 0]) + 1, 3)   # s-column +1, capped by 3 (s/p/d)
        ib = min(int(npq[nj, 0]) + 1, 3)
        for i in range(ia):                # 0..ia-1  -> Fortran i=1..ia
            pq1_i = int(npq[ni, i])        # npq(ni,i)（column：s/p/d）
            for j in range(ib):            # 0..ib-1
                pq2_j = int(npq[nj, j])
                nk1 = min(i, j) + 1        # Fortran  1..min(i,j)
                for k in range(nk1):       # k=0..nk1-1 -> Fortran kss=k+1
                    pi = max(pq1_i, i+1)   # Fortran: pi=max(pq1,iss)；iss=i
                    pj = max(pq2_j, j+1)
                    s[i, j, k] = ss_np(
                        pi, pj, i+1, j+1, k+1,
                        ul1[i], ul2[j],
                        r1=sqrt(r2), a0=a0
                    )

    # s1/s2/s3
    s1, s2, s3 = s[:, :, 0], s[:, :, 1], s[:, :, 2]
    # c1..c5
    c1, c2, c3, c4, c5 = c[:, :, 0], c[:, :, 1], c[:, :, 2], c[:, :, 3], c[:, :, 4]

    ia = min(int(npq[ni, 0]) + 1, 3)
    ib = min(int(npq[nj, 0]) + 1, 3)

    for i in range(ia):               # i: 0(s),1(p),2(d)
        kmin, kmax = 3 - (i+1), 1 + (i+1)   # Fortran: k = 4-i .. 2+i  (1-based) 0-based: 3-i .. 1+i
        for j in range(ib):           # j: 0(s),1(p),2(d)
            aa = -1.0 if (j == 1) else 1.0
            bb = -1.0 if (j == 2) else (1.0 if (j != 1) else 1.0)
            lmin, lmax = 3 - (j+1), 1 + (j+1)

            for k in range(kmin, kmax+1):
                for l in range(lmin, lmax+1):
                    ii = ival[i, k]   # AO index (1..9 或 0)
                    jj = ival[j, l]
                    if ii == 0 or jj == 0:
                        continue
                    # 0-based
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
    env = None,         # initialized environment, natorb, betas, betap, betad, cutofs zs6 zp6 zd6 npq
    dummy_code = 101    
) -> np.ndarray:
    """
    NumPy 2-center 1-electron integral H(ni,nj)。
    parameter
    ----
    ni, nj : int
        atomic number
    xi, xj : (3,) array
        cartesian coordinates of atom i and j
    natorb, betas, betap, betad : ndarray
    cutofs : float   
        MOPAC default: 15 angs.  
    dummy_code : int

    return
    ----
    smat : (n_i_orb, n_j_orb) ndarray
    """
    xi = np.asarray(xi, dtype=float).ravel()
    xj = np.asarray(xj, dtype=float).ravel()

    ni = ni - 1
    nj = nj - 1

    rab = float(np.dot(xi - xj, xi - xj))

    if (rab > env.cutofs) or (rab > 3.24 and (ni == dummy_code or nj == dummy_code)):
        ni_orb = int(env.natorb[ni])
        nj_orb = int(env.natorb[nj])
        return np.zeros((ni_orb, nj_orb), dtype=float)

    # xjuc = xi - xj
    xjuc = xj - xi

    # （Fortran: call diat(ni,nj,xjuc,smat)）
    base = np.asarray(diat_np(ni, nj, xjuc, cutof1=env.cutofs, natorb=env.natorb, zs=env.zs6, zp=env.zp6, zd=env.zd6, npq=env.npq), dtype=float)

    ni_orb = int(env.natorb[ni])
    nj_orb = int(env.natorb[nj])
    smat = base[:ni_orb, :nj_orb].copy()  # cut; Note: maybe it's more efficient to produce the final result directly

    # β/2（s, px, py, pz, d1..d5）
    bi = np.empty(9, dtype=float)
    bj = np.empty(9, dtype=float)

    bi[0] = 0.5 * env.betas6[ni]
    bi[1:4] = 0.5 * env.betap6[ni]
    bi[4:9] = 0.5 * env.betad6[ni]

    bj[0] = 0.5 * env.betas6[nj]
    bj[1:4] = 0.5 * env.betap6[nj]
    bj[4:9] = 0.5 * env.betad6[nj]

    # Fortran: do j=1,norbj; smat(:norbi,j) *= (bi(:norbi) + bj(j))
    smat *= (bi[:ni_orb, None] + bj[:nj_orb][None, :])

    return smat
