import numpy as np
from h1elec import h1elec_np
from utils import prtpar, print_title, vecprt_w
import sys

#### inid
####  └─ inighd
####      ├─ ddpo
####      ├─ scprm
####      └─ eiscor
# The relationship with original MOPAC: b_F(i,j) = b[i-1, j-1]; fx[i] = i!=fx[i+1]
fx = np.empty(30, dtype=float)
fx[0] = 1.0
for i in range(1, 30):
    fx[i] = fx[i-1] * i 

b = np.zeros((30, 30), dtype=float)
b[:, 0] = 1.0
for i in range(1, 30):
    b[i, 1:i+1] = b[i-1, :i] + b[i-1, 1:i+1]                                             # b[i,j] = C(i, j)（0-basis）
Ha2eV = 27.211386245988  # Ha2eV
def rsc(k, na, ea, nb, eb, nc, ec, nd, ed):
    '''
    Calculate the one-center two-electron integrals based on STO (Slater-Condon parameter)
    k: type of the integral can be 0 1 2 3 4 for spd-basis
    na, nb: principle number quantum number of AO, corresponding to electron 1 
    ea, eb: enponent of AO, corresponding to electron 1
    nc, nd: principle number quantum number of AO, corresponding to electron 2
    ec, ed: enponent of AO, corresponding to electron 2
    '''
    aea = np.log(ea)
    aeb = np.log(eb)
    aec = np.log(ec)
    aed = np.log(ed)
    eab = ea + eb
    ecd = ec + ed
    e = eab + ecd 
    nab = na + nb
    ncd = nc + nd
    n = nab + ncd
    ae = np.log(e)                                                                        # Not equal to 1
    a2 = np.log(2)
    aeab = np.log(eab)
    aecd = np.log(ecd)
    ff = fx[n-1]/np.sqrt(fx[2*na]*fx[2*nb]*fx[2*nc]*fx[2*nd])
    c = Ha2eV * ff * np.exp(na*aea + nb*aeb + nc*aec + nd* aed + 0.5*(aea + aeb + aec + aed) + a2*(n+2) - ae*n)   # Unit in eV
    s0 = 1.0/e
    s1 = 0.0
    s2 = 0.0
    m = ncd - k
    m2 = ncd  + k + 1
    for i in range(m):
        s0 *= e/ecd
        s1 += s0*(b[ncd-k-1, i] - b[m2-1, i])/b[n-1, i]
    for i in range(m, m2):
        s0 *= e/ecd
        s2 += s0*(b[m2-1, i])/b[n-1, i]
    s3 = np.exp(ae*n - aecd*m2 - aeab*(nab-k)) / b[n-1, m2-1]
    rsc = c * (s1 - s2 + s3)
    return rsc

def scprm(ni, env=None):
    '''
    Given element ni (one-center), calculate all the radial integrals would be used in MNDO/d model.
    '''
    ns = env.iii[ni]
    nd = env.iiid[ni]
    es = env.zsn6[ni]   # ToDo: wrap it to enable models beyong PM6
    ep = env.zpn6[ni]
    ed = env.zdn6[ni]
    r016 = rsc(0, ns, es, ns, es, nd, ed, nd, ed)   # R0ssdd
    r036 = rsc(0, ns, ep, ns, ep, nd, ed, nd, ed)   # R0ppdd
    r066 = rsc(0, nd, ed, nd, ed, nd, ed, nd, ed)   # R0dddd=F0dd
    r155 = rsc(1, ns, ep, nd, ed, ns, ep, nd, ed)   # R1pdpd=G1pd
    r125 = rsc(1, ns, es, ns, ep, ns, ep, nd, ed)   # R1sppd
    r244 = rsc(2, ns, es, nd, ed, ns, es, nd, ed)   # R2sdsd
    r236 = rsc(2, ns, ep, ns, ep, nd, ed, nd, ed)   # R2ppdd
    r266 = rsc(2, nd, ed, nd, ed, nd, ed, nd, ed)   # R2dd = F2dd
    r234 = rsc(2, ns, ep, ns, ep, ns, es, nd, ed)   # R2ppsd
    r246 = rsc(2, ns, es, nd, ed, nd, ed, nd, ed)   # R2sddd
    r355 = rsc(3, ns, ep, nd, ed, ns, ep, nd, ed)   # R3pdpd=G3pd
    r466 = rsc(4, nd, ed, nd, ed, nd, ed, nd, ed)   # R4dddd=F4dd
    return r016, r036, r066, r155, r125, r244, r236, r266, r234, r246, r355, r466

def eiscor(eisol, r016, r066, r244, r266, r466, ni):
    '''
    Adds in the ONE-CENTER terms for the atomic energy of
          those atoms that have partly filled "d" shells
    '''
    ir016 = np.zeros(101, dtype=int)
    ir066 = np.zeros(101, dtype=int)
    ir244 = np.zeros(101, dtype=int)
    ir266 = np.zeros(101, dtype=int)
    ir466 = np.zeros(101, dtype=int)

    # 20..28: Sc..Cu
    ir016[20:29] = [ 2,  4,  6,  5, 10, 12, 14, 16, 10]
    ir066[20:29] = [ 0,  1,  3, 10, 10, 15, 21, 28, 45]
    ir244[20:29] = [ 1,  2,  3,  5,  5,  6,  7,  8,  5]
    ir266[20:29] = [ 0,  8, 15, 35, 35, 35, 43, 50, 70]
    ir466[20:29] = [ 0,  1,  8, 35, 35, 35, 36, 43, 70]

    # 38..46: Y..Ag
    ir016[38:47] = [ 2,  4,  4,  5, 10,  7,  8,  0, 10]
    ir066[38:47] = [ 0,  1,  6, 10, 10, 21, 28, 45, 45]
    ir244[38:47] = [ 1,  2,  4,  5,  5,  5,  5,  0,  5]
    ir266[38:47] = [ 0,  8, 21, 35, 35, 43, 50, 70, 70]
    ir466[38:47] = [ 0,  1, 21, 35, 35, 36, 43, 70, 70]

    # 57: La
    ir016[56] = 2
    ir066[56] = 0
    ir244[56] = 1
    ir266[56] = 0
    ir466[56] = 0

    # 70: Lu
    ir016[70] = 2
    ir066[70] = 0
    ir244[70] = 1
    ir266[70] = 0
    ir466[70] = 0

    # 72..80: Hf..Hg
    ir016[71:80] = [ 4,  6,  5, 10, 12, 14,  9, 10, 0]
    ir066[71:80] = [ 1,  3, 10, 10, 15, 21, 36, 45, 0]
    ir244[71:80] = [ 2,  3,  5,  5,  6,  7,  5,  5, 0]
    ir266[71:80] = [ 8, 15, 35, 35, 35, 43, 56, 70, 0]
    ir466[71:80] = [ 1,  8, 35, 35, 35, 36, 56, 70, 0]
    
    corr = ir016[ni]*r016 + ir066[ni]*r066 - ir244[ni]*r244/5.0 - ir266[ni]*r266/49.0 - ir466[ni]*r466/49.0
    eisol[ni] += corr

def inighd(ni, env=None):
    '''
    ONE-CENTER TWO-ELECTRON INTEGRALS FOR SPD-BASIS.
    '''
    # ToDo: whether dorbs is included in ni element
    # if dorbs: 
    #   return

    s3 = 1.7320508
    s5 = 2.23606797
    s15 = 3.87298334
    r016, r036, r066, r155, r125, r244, r236, r266, r234, r246, r355, r466 = scprm(ni,env=env)
    if  env.f0sd6[ni] > 0.001:
        r016 = env.f0sd6[ni]    # TO use emperical parameter.
    '''
    Reason: analytical value based on the integral of STO is less confidential for d species
    1. NDDO approximation
    2. Screening/penetration/multireference effect
    3. realistic effect
    4. Hard to fit the thermodynamics
    F0sd: the repulsion between s-d -> charge distribution; ionicity/covalence; enthalpy
    G2sd: the exchange between s-d -> spin-state energetics; coordination field spliting ; geometry; ...
    '''
    if env.g2sd6[ni] > 0.001:
        r244 = env.g2sd6[ni]
    eiscor(env.eisol, r016, r066, r244, r266, r466, ni)
    repd = env.repd
    repd[0, ni] = r016
    repd[1, ni] = 2.0/(3.0*s5)*r125
    repd[2, ni] = 1.0/s15 *r125 
    repd[3, ni] = 2.0/(5*s5)*r234
    repd[4, ni] = r036+4.0/35.0*r236 
    repd[5, ni] = r036+2.0/35.0*r236 
    repd[6, ni] = r036-4.0/35.0*r236 
    repd[7, ni] = -1.0/(3.0*s5)*r125
    repd[8, ni] = np.sqrt(3.0/125.0)*r234
    repd[9, ni] = s3/35.0*r236
    repd[10, ni] = 3.0/35.0*r236
    repd[11, ni] = -1.0/(5.0*s5)*r234
    repd[12, ni] = r036 -2.0/35.0*r236
    repd[13, ni] = -2.0*s3/35.0*r236
    repd[14, ni] = -repd[2,ni]
    repd[15, ni] = -repd[10,ni]
    repd[16, ni] = -repd[8, ni]
    repd[17, ni] = -repd[13,ni]
    repd[18, ni] = 1.0/5.0*r244
    repd[19, ni] = 2.0/(7.0*s5)*r246
    repd[20, ni] = repd[19, ni]/2.0
    repd[21, ni] = -repd[19, ni]
    repd[22, ni] = 4.0/15.0*r155 + 27.0/245.0*r355
    repd[23, ni] = 2.0*s3/15.0*r155 - 9.0*s3/245.0*r355
    repd[24, ni] = 1.0/15.0*r155 + 18.0/245.0*r355
    repd[25, ni] = (-s3/15.0*r155) + 12.0*s3/245.0*r355
    repd[26, ni] = (-s3/15.0*r155) - 3.0*s3/245.0*r355
    repd[27, ni] = -repd[26,ni]
    repd[28, ni] = r066 + 4.0/49.0*r266 + 4.0/49.0*r466
    repd[29, ni] = r066 + 2.0/49.0*r266 - 24.0/441.0*r466
    repd[30, ni] = r066 - 4.0/49.0*r266 + 6.0/441.0*r466
    repd[31, ni] = np.sqrt(3.0/245.0)*r246
    repd[32, ni] = 1.0/5.0*r155 + 24.0/245.0*r355
    repd[33, ni] = 1.0/5.0*r155 - 6.0/245.0*r355
    repd[34, ni] = 3.0/49.0*r355
    repd[35, ni] = 1.0/49.0*r266 + 30.0/441.0*r466
    repd[36, ni] = s3/49.0*r266 - 5.0*s3/441.0*r466
    repd[37, ni] = r066 - 2.0/49.0*r266 - 4.0/441.0*r466
    repd[38, ni] = (-2.0*s3/49.0*r266) + 10.0*s3/441.0*r466
    repd[39, ni] = -repd[31,ni]
    repd[40, ni] = -repd[33,ni]
    repd[41, ni] = -repd[34,ni]
    repd[42, ni] = -repd[36,ni]
    repd[43, ni] = 3.0/49.0*r266 + 20.0/441.0*r466
    repd[44, ni] = -repd[38,ni]
    repd[45, ni] = 1.0/5.0*r155 - 3.0/35.0*r355
    repd[46, ni] = -repd[45,ni]
    repd[47, ni] = 4.0/49.0*r266 + 15.0/441.0*r466
    repd[48, ni] = 3.0/49.0*r266 - 5.0/147.0*r466
    repd[49, ni] = -repd[48,ni]
    repd[50, ni] = r066 + 4.0/49.0*r266 - 34.0/441.0*r466
    repd[51, ni] = 35.0/441.0*r466
    env.f0dd[ni] = r066
    env.f2dd[ni] = r266
    env.f4dd[ni] = r466
    env.f0sd[ni] = r016
    env.g2sd6[ni] = r244
    env.f0pd[ni] = r036
    env.f2pd[ni] = r236
    env.g1pd[ni] = r155
    env.g3pd[ni] = r355
    # return may be not required
def aijl(z1, z2, n1, n2, l):
    zz = z1 + z2 + 1e-20
    aijl = fx[n1+n2+l] / np.sqrt(fx[2*n1]*fx[2*n2]) * (2*z1/zz)**(n1+0.5) * (2*z2/zz)**(n2+0.5) * 2**l / zz**l    # Note: additional 2^l
    return aijl

def aijm(ni, env=None):
    '''
    AIJ-values for evaluation of 2-center 2-electron integrals
    and of hybrid contribution to dipole moment in MNDO-D
    Deinition can be found in DOI: 10.1007/BF01134863.
    Result are stored in array AIJ(6,5,107)
    '''
    z1 = env.zs6[ni]
    z2 = env.zp6[ni]
    z3 = env.zd6[ni]
    nsp = env.iii[ni]
    if ni < 2: 
        return
    zz = z1*z2
    if zz < 1e-2:
        return
    
    env.aij[1, ni] = aijl(z1, z2, nsp, nsp, 1)  # sp
    env.aij[2, ni] = aijl(z2, z2, nsp, nsp, 2)  # pp
    if env.dorbs[ni]:
        nd = env.iiid[ni]
        env.aij[3, ni] = aijl(z1, z3, nsp, nd, 2) # sd
        env.aij[4, ni] = aijl(z2, z3, nsp, nd, 1) # pd
        env.aij[5, ni] = aijl(z3, z3, nd, nd, 2)  # dd

def poij(l, d, fg):
    '''
    Determine additive terms rho=poij for 2-center 2-electron integrals from the requirement that appropriate 1-center 2-electron integrals are reproduced
    l: 0, 1, 2, 3, 4 for spd-basis  多极矩阶数
    d: distance between 2 centers
    fg: targeted 1-center 2-electron integral
    代码把“两中心表达”与“给定的一中心数值 fg”做差的平方作为目标函数 f(a)，在区间 [0.1,5.0] 上做黄金分割搜索找最小点。
    '''
    niter = 100
    epsil = 1E-8
    # Golden-Section Search
    g1 = 0.382
    g2 = 0.618
    f1 = 0
    f2 = 0
    if l == 0:
        return 0.5*Ha2eV/fg     # charge in klopman's approximation 
    dsq  = d*d 
    ev4 = Ha2eV/4.0
    ev8 = Ha2eV/8.0
    a1 = 0.1
    a2 = 5.0
    if l == 1:
        for i in range(niter):
            delta = a2 - a1
            if delta < epsil:
                break 
            y1 = a1 + delta*g1
            y2 = a1 + delta*g2
            f1 = (ev4*(1.0/y1 - 1.0/np.sqrt(y1**2 + dsq)) - fg)**2
            f2 = (ev4*(1.0/y2 - 1.0/np.sqrt(y2**2 + dsq)) - fg)**2
            if f1 < f2:
                a2 = y2
            else:
                a1 = y1
    if l == 2:
        for i in range(niter):
            delta = a2 - a1
            if delta < epsil:
                break
            y1 = a1 + delta*g1
            y2 = a1 + delta*g2
            f1 = (ev8*(1.0/y1 - 2.0/np.sqrt(y1**2+dsq*0.5) + 1.0/np.sqrt(y1**2+dsq)) - fg)**2
            f2 = (ev8*(1.0/y2 - 2.0/np.sqrt(y2**2+dsq*0.5) + 1.0/np.sqrt(y2**2+dsq)) - fg)**2
            if f1 < f2:
                a2 = y2    
            else:
                a1 = y1
    if f1 >= f2:
        return a2
    else:
        return a1

def ddpo(ni, env=None):
    '''
    Calculation of charge separations and additive terms used to compute the 2-center 2-electron integrals in MNDO/d
    DD(6,107) from array aij computed in AIJM
    PO(9, 107) from POIJ
    SECOND INDEX OF DD AND PO     SS 1, SP 2,PP 8, PP 3, SD 4, PD 5, DD
    SEE EQUATIONS (12)-(16) OF TCA PAPER FOR DD.
    SEE EQUATIONS (19)-(26) OF TCA PAPER FOR PO.
    SPECIAL CONVENTION FOR ATOMIC CORE: ADDITIVE TERM PO(9,NI)
    USED IN THE EVALUATION OF THE CORE-ELECTRON ATTRACTIONS AND
    CORE-CORE REPULSIONS.
    需已有全局数组：gss6,hsp6,gpp6,gp2,dorbs,aij,repd,po,ddp
    以及 poij(l,d,fg)（或内部用全局 Ha2eV）
    '''
    # additive term for ss
    fg = env.gss6[ni]
    po = env.po
    aij = env.aij
    ddp = env.ddp
    dorbs = env.dorbs
    repd = env.repd

    if fg > 0.1:
        po[0, ni] = poij(0, 1.0, fg)   # monopole d == 1
    if ni >= 2:   # python start from 0;
        # other terms for sp basis
        # sp 
        d = aij[1, ni] / np.sqrt(12.0)
        fg = env.hsp6[ni]
        ddp[1, ni] = d
        po[1, ni] = poij(1, d, fg)
        # pp 
        po[6, ni] = po[0, ni]
        d = np.sqrt(aij[2, ni]*0.1) 
        fg = 0.5*(env.gpp6[ni] - env.gp26[ni])  # rotational invariance
        ddp[2, ni] = d
        po[2, ni] = poij(2, d, fg)
        if dorbs[ni]:
            # terms involving D orbitals
            # sd l=2
            da = np.sqrt(1.0/60.0)
            d = np.sqrt(aij[3, ni]*da)
            fg = repd[18, ni]    # hsd
            ddp[3, ni] = d
            po[3, ni] = poij(2, d, fg)
            # pd j=1
            d = aij[4, ni]/np.sqrt(20.0)
            fg = repd[22, ni] -1.8*repd[34, ni]   # hpd
            ddp[4, ni] = d 
            po[4, ni] = poij(1, d, fg)
            # dd l=0
            fg = 0.2*(repd[28, ni] + 2.0*repd[29, ni] + 2.0*repd[30, ni])
            if fg > 1E-5:
                po[7, ni] = poij(0, 1.0, fg)
            else:
                po[7, ni] = 1E5
            # dd l=2
            d = np.sqrt(aij[5, ni]/14.0)
            fg = repd[43, ni] - (20.0/35.0)*repd[51, ni]
            ddp[5, ni] = d
            po[5, ni] = poij(2, d, fg)
    
def inid(env=None):   
    '''
    DEFINE SEVERAL PARAMETERS FOR D-ORBITAL CALCULATIONS.
    '''
    po = env.po
    am = env.am
    ad = env.ad
    aq = env.aq
    ddp = env.ddp
    dd = env.dd
    qq = env.qq
    
    for ni in range(107):
        if not env.dorbs[ni]:
            continue
        aijm(ni, env=env)                               # update the aij 6*107 matrix
        if env.zdn6[ni] > 1E-4:
            inighd(ni, env=env)                         # update the repd 53*107 matrix
        ddpo(ni, env=env)                               # update the ddp 6*107 multipole distance matrix and po 9*107 R->0 limit electron density, with corresponding single-center two-electron integral
    for i in range(106):
        if env.natorb[i] < 6 or env.main_group[i]:     
            if am[i] < 1E-4:                   
                am[i] = 1.0
            po[0, i] = 0.5/am[i]              
            if ad[i]>1E-5:
                po[1, i] = 0.5/ad[i]
            if aq[i]>1E-5:
                po[2, i] = 0.5/aq[i]
            po[6, i] = po[0, i]
            ddp[1, i] = dd[i]                 
            ddp[2, i] = qq[i]*np.sqrt(2.0)
        po[8, i] = po[0, i]
        if env.pocord6[i] > 1E-5: 
            po[8, i] = env.pocord6[i]
    po[1, 0] = 0
    po[2, 0] = 0

def sp_two_electron(env=None):
    '''
    One-center two-electron integrals via slater-Condon/Racah radial integration formula
    GSS, GSP, GPP, GP2, HSP
    '''
    for ni in range(80):
        ns = env.iii[ni]
        es = env.zsn6[ni]
        ep = env.zpn6[ni]
        if es < 1E-4 or ep < 1E-4 or env.main_group[ni]:
            continue
        env.gss6[ni] = rsc(0, ns, es, ns, es, ns, es, ns, es)
        env.gsp6[ni] = rsc(0, ns, es, ns, es, ns, ep, ns, ep)
        env.hsp6[ni] = rsc(1, ns, es, ns, ep, ns, es, ns, ep)/3.0
        r033 = rsc(0, ns, ep, ns, ep, ns, ep, ns, ep)
        r233 = rsc(2, ns, ep, ns, ep, ns, ep, ns, ep)
        env.gpp6[ni] = r033 + 0.16*r233
        env.gp26[ni] = r033 - 0.08*r233


def calpar(env=None):
    '''
    Based on the basic parameters and semi-empirical model methods of the semi-empirical model, derive and standardize a set of secondary parameters and constants; remove the MINDO model branch from the original code.
    '''
    # local variables 
    gssc = np.zeros(107)
    gspc = np.zeros(107)
    hspc = np.zeros(107)
    gp2c = np.zeros(107)
    gppc = np.zeros(107)

    nspqn = np.array([1]*2+[2]*8+[3]*8+[4]*18+[5]*18+[6]*32+[0]*16)
    # set scaling parameters
    p = 2.0
    p4 = p**4
    sp_two_electron(env)      # update the parameters
    # am, ad, aq, dd, qq have been updated 
    iop = env.iop
    ios = env.ios
    iod = env.iod
    zs6 = env.zs6
    zp6 = env.zp6
    gpp6 = env.gpp6
    gp26 = env.gp26
    gsp6 = env.gsp6
    hsp6 = env.hsp6
    eisol = env.eisol
    uss6 = env.uss6
    upp6 = env.upp6
    udd6 = env.udd6
    gss6 =  env.gss6
    dd = env.dd
    qq = env.qq
    am = env.am
    ad = env.ad
    aq = env.aq
    for i in range(1,97):
        gssc[i] = max(ios[i] - 1, 0)   # GSSC is the number of two-electron terms of type <SS|SS>
        k = iop[i]
        gspc[i] = ios[i]*k              # GSSC is the number of two-electron terms of type <SS|PP>
        l = min(k, 6-k)
        gp2c[i] = (k*(k-1))/2 + 0.5*(l*(l - 1))/2          # GP2C is the number of two-electron terms of type <PP|PP> plus 0.5 of the number of HPP integrals
        gppc[i] = -0.5*(l*(l - 1))/2    # GPPC is minus 0.5 times the number of HPP integrals.
        hspc[i] = -k*ios[i]*0.5         # HSPC is the number of two-electron terms of type <SP|SP>
        if zp6[i]<1E-4 and zs6[i]<1E-4:
            continue
        zp6[i] = max(0.3, zp6[i])   # lower limit
        hpp = 0.5*(gpp6[i] - gp26[i])
        hpp = max(0.1, hpp)
        # isolated atomic electron energy
        eisol[i] = uss6[i]*ios[i] + upp6[i]*iop[i] + udd6[i]*iod[i] + gss6[i]*gssc[i] + gpp6[i]*gppc[i] + \
                    gsp6[i]*gspc[i] + gp26[i]*gp2c[i] + hsp6[i]*hspc[i]
        qn = nspqn[i]
        # Based on the radial moment constants of atomic Slater orbitals, they correspond to the "length scales" of the dipole and quadrupole channels respectively. 
        # They are not geometric distances, but radial integral scales derived from the Slater exponents and principal quantum numbers of the s/p valence orbitals on the same atom, which are used to self-consistently align one-center two-electron integrals (such as HSP and HPP) with the Slater exponents.
        # Theoret. Chim. Acta (Bert.) 46, 89-104 (1977) 式 （15） （16）已推广
        dd[i] = (2.0*qn + 1)*(4.0*zs6[i]*zp6[i])**(qn + 0.5)/(zs6[i]+zp6[i])**(2.0*qn + 2)/np.sqrt(3)
        qq[i] = np.sqrt((4.0*qn*qn + 6.0*qn + 2.0)/20.0)/zp6[i]
        # CALCULATE ADDITIVE TERMS, IN ATOMIC UNITS.
        jmax = 5
        gdd1 = (hsp6[i]/(Ha2eV*dd[i]**2))**(1.0/3.0)   
        d1 = gdd1
        d2 = gdd1 + 0.04
        for j in range(jmax):
            df = d2 - d1
            hsp1 = 0.50*d1 - 0.50/np.sqrt(4.0*dd[i]**2+1.0/d1**2)
            hsp2 = 0.50*d2 - 0.50/np.sqrt(4.0*dd[i]**2+1.0/d2**2)
            if abs(hsp2 - hsp1) < 1E-25:
                break
            d3 = d1 + df*(hsp6[i]/Ha2eV-hsp1)/(hsp2 - hsp1)
            d1 = d2
            d2 = d3
        gqq = (p4*hpp/(Ha2eV*48.0*qq[i]**4))**0.2
        q1 = gqq
        q2 = gqq + 0.04
        for j in range(jmax):
            qf = q2 - q1
            hpp1 = 0.25*q1 - 0.5/np.sqrt(4.0*qq[i]**2+1.0/q1**2) + 0.25/np.sqrt(8.0*qq[i]**2+1.0/q1**2)
            hpp2 = 0.25*q2 - 0.5/np.sqrt(4.0*qq[i]**2+1.0/q2**2) + 0.25/np.sqrt(8.0*qq[i]**2+1.0/q2**2)
            if abs(hpp2 - hpp1) < 1E-25:
                break
            q3 = q1 + qf*(hpp/Ha2eV - hpp1)/(hpp2 - hpp1)
            q1 = q2
            q2 = q3
        am[i] = gss6[i] / Ha2eV
        ad[i] = d2
        aq[i] = q2
    for i in range(107):
        if am[i] < 1E-20:
            if gss6[i] > 1E-20:
                am[i] = gss6[i]/Ha2eV
            else:
                am[i] = 1.0
    # H atom
    # print(eisol[24])
    eisol[0] = uss6[0]
    am[0] = gss6[0] / Ha2eV
    ad[0] = am[0]
    aq[0] = am[0]
    '''for i in range(100):
        if f0sd_store[i] < 1E-20:
            fosd[i] = 0.0
        if g2sd_store[i] < 1E-20:
            g2sd[i] = 0.0'''
    
    inid(env=env)   # Calculate derived parameters for "d" orbital work

    am[101] = 1E-10

#### rotatd
####  ├─ reppd
####  │   └─ to_point          # Note: long distance limit is not implemented in this basic pyPM6 module.
####  ├─ reppd2                # augmented term
####  ├─ tx                    # combination of radial vectors generated by reppd function
####  ├─ spcore                # electron-core tensor cored
####  ├─ rotmat                # rotate the local s/p/d tensor into the laboratory system
####  ├─ w2mat                 # write the two-electron matrix block into the one-dimensional buffer w and advance kr
####  ├─ elenuc                # accumulate the electron-core tensor cored with the rotated s/p/d tensor into the lower triangle of H
####  └─ aijm 或 ccrep(*)      # core-core repulsion term
a0 =  0.529177210903
def charg_np(r, l1, l2, m, da, db, add):
    # Q-Q
    if l1 == 0 and l2 == 0:
        return 1.0/np.sqrt(r*r + add)

    # Z-Q
    elif l1 == 1 and l2 == 0:
        val = (-1.0/np.sqrt((r + da)*(r + da) + add)
               +  1.0/np.sqrt((r - da)*(r - da) + add))
        return val/2.0

    # Q-Z
    elif l1 == 0 and l2 == 1:
        val = ( 1.0/np.sqrt((r + db)*(r + db) + add)
              - 1.0/np.sqrt((r - db)*(r - db) + add))
        return val/2.0

    # Z-Z (m=0)
    elif l1 == 1 and l2 == 1 and m == 0:
        dzdz = ( 1.0/np.sqrt((r + da - db)**2 + add)
               +1.0/np.sqrt((r - da + db)**2 + add)
               -1.0/np.sqrt((r - da - db)**2 + add)
               -1.0/np.sqrt((r + da + db)**2 + add))
        return dzdz/4.0

    # X-X (m=1)
    elif l1 == 1 and l2 == 1 and m == 1:
        dxdx = ( 2.0/np.sqrt(r*r + (da - db)*(da - db) + add)
               -2.0/np.sqrt(r*r + (da + db)*(da + db) + add))
        return dxdx/4.0

    # Q-ZZ
    elif l1 == 0 and l2 == 2:
        qqzz = ( 1.0/np.sqrt((r - db)*(r - db) + add)
               -2.0/np.sqrt(r*r + db*db + add)
               +1.0/np.sqrt((r + db)*(r + db) + add))
        return qqzz/4.0

    # ZZ-Q
    elif l1 == 2 and l2 == 0:
        qzzq = ( 1.0/np.sqrt((r - da)*(r - da) + add)
               -2.0/np.sqrt(r*r + da*da + add)
               +1.0/np.sqrt((r + da)*(r + da) + add))
        return qzzq/4.0

    # Z-ZZ (m=0)
    elif l1 == 1 and l2 == 2 and m == 0:
        dzqzz = ( 1.0/np.sqrt((r - da - db)**2 + add)
                -2.0/np.sqrt((r - da)**2 + db*db + add)
                +1.0/np.sqrt((r + db - da)**2 + add)
                -1.0/np.sqrt((r - db + da)**2 + add)
                +2.0/np.sqrt((r + da)**2 + db*db + add)
                -1.0/np.sqrt((r + da + db)**2 + add))
        return dzqzz/8.0

    # ZZ-Z (m=0)
    elif l1 == 2 and l2 == 1 and m == 0:
        qzzdz = (-1.0/np.sqrt((r - da - db)**2 + add)
                 +2.0/np.sqrt((r - db)**2 + da*da + add)
                 -1.0/np.sqrt((r + da - db)**2 + add)
                 +1.0/np.sqrt((r - da + db)**2 + add)
                 -2.0/np.sqrt((r + db)**2 + da*da + add)
                 +1.0/np.sqrt((r + da + db)**2 + add))
        return qzzdz/8.0

    # ZZ-ZZ (m=0)
    elif l1 == 2 and l2 == 2 and m == 0:
        zzzz = ( 1.0/np.sqrt((r - da - db)**2 + add)
               +1.0/np.sqrt((r + da + db)**2 + add)
               +1.0/np.sqrt((r - da + db)**2 + add)
               +1.0/np.sqrt((r + da - db)**2 + add)
               -2.0/np.sqrt((r - da)**2 + db*db + add)
               -2.0/np.sqrt((r - db)**2 + da*da + add)
               -2.0/np.sqrt((r + da)**2 + db*db + add)
               -2.0/np.sqrt((r + db)**2 + da*da + add)
               +2.0/np.sqrt(r*r + (da - db)*(da - db) + add)
               +2.0/np.sqrt(r*r + (da + db)*(da + db) + add))
        xyxy = ( 4.0/np.sqrt(r*r + (da - db)*(da - db) + add)
               +4.0/np.sqrt(r*r + (da + db)*(da + db) + add)
               -8.0/np.sqrt(r*r + da*da + db*db + add))
        return zzzz/16.0 - xyxy/64.0

    # X-ZX (m=1)
    elif l1 == 1 and l2 == 2 and m == 1:
        ab = db/np.sqrt(2.0)
        dxqxz = (-2.0/np.sqrt((r - ab)**2 + (da - ab)**2 + add)
                 +2.0/np.sqrt((r + ab)**2 + (da - ab)**2 + add)
                 +2.0/np.sqrt((r - ab)**2 + (da + ab)**2 + add)
                 -2.0/np.sqrt((r + ab)**2 + (da + ab)**2 + add))
        return dxqxz/8.0

    # ZX-X (m=1)
    elif l1 == 2 and l2 == 1 and m == 1:
        aa = da/np.sqrt(2.0)
        qxzdx = (-2.0/np.sqrt((r + aa)**2 + (aa - db)**2 + add)
                 +2.0/np.sqrt((r - aa)**2 + (aa - db)**2 + add)
                 +2.0/np.sqrt((r + aa)**2 + (aa + db)**2 + add)
                 -2.0/np.sqrt((r - aa)**2 + (aa + db)**2 + add))
        return qxzdx/8.0

    # ZX-ZX (m=1)
    elif l1 == 2 and l2 == 2 and m == 1:
        aa = da/np.sqrt(2.0); ab = db/np.sqrt(2.0)
        qxzqxz = ( 2.0/np.sqrt((r + aa - ab)**2 + (aa - ab)**2 + add)
                  -2.0/np.sqrt((r + aa + ab)**2 + (aa - ab)**2 + add)
                  -2.0/np.sqrt((r - aa - ab)**2 + (aa - ab)**2 + add)
                  +2.0/np.sqrt((r - aa + ab)**2 + (aa - ab)**2 + add)
                  -2.0/np.sqrt((r + aa - ab)**2 + (aa + ab)**2 + add)
                  +2.0/np.sqrt((r + aa + ab)**2 + (aa + ab)**2 + add)
                  +2.0/np.sqrt((r - aa - ab)**2 + (aa + ab)**2 + add)
                  -2.0/np.sqrt((r - aa + ab)**2 + (aa + ab)**2 + add))
        return qxzqxz/16.0

    # XX-XX (m=2)
    elif l1 == 2 and l2 == 2 and m == 2:
        xyxy = ( 4.0/np.sqrt(r*r + (da - db)**2 + add)
               +4.0/np.sqrt(r*r + (da + db)**2 + add)
               -8.0/np.sqrt(r*r + da*da + db*db + add))
        return xyxy/16.0

    else:
        raise ValueError(f"unsupported combination: l1={l1}, l2={l2}, m={m}")

def rijkl(ni, nj, ij, kl, li, lj, lk, ll, ic, r, env=None):
    '''
    (ij,kl) multipole interaction
    '''
    # 0-based indices
    indx = env.indx
    po = env.po
    ddp = env.ddp
    ch = env.ch

    ni_idx = ni 
    nj_idx = nj 

    # L组合限制
    l1min = min(abs(li - lj), 2)
    l1max = min(li+lj, 2)
    l2min = min(abs(lk-ll), 2)
    l2max = min(lk+ll, 2)
    # 索引
    lij_val = int(indx[li, lj])
    lkl_val = int(indx[lk, ll])
    total = 0.0

    for l1 in range(l1min, l1max + 1):
        # pick pij, dij for the left pair
        if l1 == 0:
            # Fortran cases: lij = 0,2,5  # 1,3,6
            if   lij_val == 0:
                # po row: 1->0 (or 9->8 when ic==1)
                pij = po[8, ni_idx] if ic == 1 else po[0, ni_idx]
            elif lij_val == 2:
                # po row: 7->6
                pij = po[6, ni_idx]
            elif lij_val == 5:
                # po row: 8->7
                pij = po[7, ni_idx]
            else:
                raise ValueError(f"unsupported lij for l1==0: lij={lij_val}")
            dij = 0.0
        else:
            pij = po[lij_val , ni_idx]   # lij (1-based) -> row lij-1
            dij = ddp[lij_val, ni_idx]

        for l2 in range(l2min, l2max + 1):
            # pick pkl, dkl for the right pair
            if l2 == 0:
                if   lkl_val == 0:
                    pkl = po[8, nj_idx] if ic == 2 else po[0, nj_idx]
                elif lkl_val == 2:
                    pkl = po[6, nj_idx]
                elif lkl_val == 5:
                    pkl = po[7, nj_idx]
                else:
                    raise ValueError(f"unsupported lkl for l2==0: lkl={lkl_val}")
                dkl = 0.0
            else:
                pkl = po[lkl_val, nj_idx]
                dkl = ddp[lkl_val, nj_idx]

            add = (pij + pkl) ** 2
            lmin = min(l1, l2)

            s1 = 0.0 
            for m in range(-lmin, lmin + 1):
                ccc = ch[ij, l1, m + 2] * ch[kl, l2, m + 2]  
                if ccc == 0.0:
                    continue
                mm = abs(m)
                s1 = s1 + charg_np(r, l1, l2, mm, dij, dkl, add) * ccc

            total = total + s1
    return total

def reppd(ni, nj, rij, env=None):  
    """
    NumPy 0-based reppd
    Parameters
    ----------
    ni, nj : int
        0-based atomic index
    rij : float
        atomic distance (Å)
    natorb, dd, qq, am, ad, aq : (N,) arrays
        ( MOPAC/NDDO)。
    po : (9, N) array
        parameter table

    Returns
    -------
    ri : (22,) float64
        2 center 2 electron integrals
    gab : float
    """
    td = 2.0
    half = 0.5
    
    am = env.am
    dd = env.dd
    qq = env.qq
    ad = env.ad
    aq = env.aq
    po = env.po
    # output
    ri = np.zeros(22, dtype=np.float64)

    # scale the distance
    r = rij / a0
    rsq = r * r

    #  nri（FORTRAN 1-based -> PYTHON 0-based）
    nri = np.array([ 1,-1, 1, 1,-1, 1, 1,-1,-1,-1, 1, 1,-1,-1,-1, 1, 1, 1, 1, 1, 1, 1],
                   dtype=np.int8)

    # Heavy atom ? (natorb >= 3)
    si = (env.natorb[ni] >= 3)
    sj = (env.natorb[nj] >= 3)

    # ===== G_AB (core-core) aee_cc =====
    aee_cc = po[8, ni] + po[8, nj]   # Fortran po(9,ni)+po(9,nj)
    aee_cc = aee_cc * aee_cc
    gab = Ha2eV / np.sqrt(rsq + aee_cc)

    # ===== aee_te =====
    aee_te = (half / am[ni] + half / am[nj]) ** 2

    # 3 classes：H-H；Heavy-H；Heavy-Heavy
    if (not si) and (not sj):
        # ------ H - H  (SS/SS) ------
        ri[0] = Ha2eV / np.sqrt(rsq + aee_te)

    elif si and (not sj):
        # ------ Heavy - H ------
        da = dd[ni]
        qa = qq[ni] * td
        ade = (half / ad[ni] + half / am[nj]) ** 2
        aqe = (half / aq[ni] + half / am[nj]) ** 2

        arg = np.empty(7, dtype=np.float64)
        arg[0] = rsq + aee_te
        arg[1] = (r + da) ** 2 + ade
        arg[2] = (r - da) ** 2 + ade
        arg[3] = (r + qa) ** 2 + aqe
        arg[4] = (r - qa) ** 2 + aqe
        arg[5] = rsq + aqe
        arg[6] = arg[5] + qa * qa
        sqr = np.sqrt(arg)

        ev1, ev2 = Ha2eV / 2.0, Ha2eV / 4.0
        ee = Ha2eV / sqr[0]
        ri[0] = ee
        ri[1] = ev1 / sqr[1] - ev1 / sqr[2]                  # (SO/SS)
        ri[2] = ee + ev2 / sqr[3] + ev2 / sqr[4] - ev1 / sqr[5]  # (OO/SS)
        ri[3] = ee + ev1 / sqr[6] - ev1 / sqr[5]             # (PP/SS)

    elif (not si) and sj:
        # ------ H - Heavy ------
        db = dd[nj]
        qb = qq[nj] * td
        aed = (half / am[ni] + half / ad[nj]) ** 2
        aeq = (half / am[ni] + half / aq[nj]) ** 2

        arg = np.empty(7, dtype=np.float64)
        arg[0] = rsq + aee_te
        arg[1] = (r - db) ** 2 + aed
        arg[2] = (r + db) ** 2 + aed
        arg[3] = (r - qb) ** 2 + aeq
        arg[4] = (r + qb) ** 2 + aeq
        arg[5] = rsq + aeq
        arg[6] = arg[5] + qb * qb
        sqr = np.sqrt(arg)

        ev1, ev2 = Ha2eV / 2.0, Ha2eV / 4.0
        ee = Ha2eV / sqr[0]
        ri[0]  = ee
        ri[4]  = ev1 / sqr[1] - ev1 / sqr[2]                 # (SS/OS)
        ri[10] = ee + ev2 / sqr[3] + ev2 / sqr[4] - ev1 / sqr[5]  # (SS/OO)
        ri[11] = ee + ev1 / sqr[6] - ev1 / sqr[5]            # (SS/PP)

    else:
        # ------ Heavy - Heavy ------
        da = dd[ni]; db = dd[nj]
        qa = qq[ni] * td; qb = qq[nj] * td

        ade = (half / ad[ni] + half / am[nj]) ** 2
        aqe = (half / aq[ni] + half / am[nj]) ** 2
        aed = (half / am[ni] + half / ad[nj]) ** 2
        aeq = (half / am[ni] + half / aq[nj]) ** 2
        axx = (half / ad[ni] + half / ad[nj]) ** 2
        adq = (half / ad[ni] + half / aq[nj]) ** 2
        aqd = (half / aq[ni] + half / ad[nj]) ** 2
        aqq = (half / aq[ni] + half / aq[nj]) ** 2

        arg = np.empty(72, dtype=np.float64)

        arg[0]  = rsq + aee_te
        arg[1]  = (r + da) ** 2 + ade
        arg[2]  = (r - da) ** 2 + ade
        arg[3]  = (r - qa) ** 2 + aqe
        arg[4]  = (r + qa) ** 2 + aqe
        arg[5]  = rsq + aqe
        arg[6]  = arg[5] + qa * qa

        arg[7]  = (r - db) ** 2 + aed
        arg[8]  = (r + db) ** 2 + aed
        arg[9]  = (r - qb) ** 2 + aeq
        arg[10] = (r + qb) ** 2 + aeq
        arg[11] = rsq + aeq
        arg[12] = arg[11] + qb * qb

        arg[13] = rsq + axx + (da - db) ** 2
        arg[14] = rsq + axx + (da + db) ** 2
        arg[15] = (r + da - db) ** 2 + axx
        arg[16] = (r - da + db) ** 2 + axx
        arg[17] = (r - da - db) ** 2 + axx
        arg[18] = (r + da + db) ** 2 + axx

        arg[19] = (r + da) ** 2 + adq
        arg[20] = arg[19] + qb * qb
        arg[21] = (r - da) ** 2 + adq
        arg[22] = arg[21] + qb * qb

        arg[23] = (r - db) ** 2 + aqd
        arg[24] = arg[23] + qa * qa
        arg[25] = (r + db) ** 2 + aqd
        arg[26] = arg[25] + qa * qa

        arg[27] = (r + da - qb) ** 2 + adq
        arg[28] = (r - da - qb) ** 2 + adq
        arg[29] = (r + da + qb) ** 2 + adq
        arg[30] = (r - da + qb) ** 2 + adq

        arg[31] = (r + qa - db) ** 2 + aqd
        arg[32] = (r + qa + db) ** 2 + aqd
        arg[33] = (r - qa - db) ** 2 + aqd
        arg[34] = (r - qa + db) ** 2 + aqd

        arg[35] = rsq + aqq
        arg[36] = arg[35] + (qa - qb) ** 2
        arg[37] = arg[35] + (qa + qb) ** 2
        arg[38] = arg[35] + qa * qa
        arg[39] = arg[35] + qb * qb
        arg[40] = arg[38] + qb * qb

        arg[41] = (r - qb) ** 2 + aqq
        arg[42] = arg[41] + qa * qa
        arg[43] = (r + qb) ** 2 + aqq
        arg[44] = arg[43] + qa * qa
        arg[45] = (r + qa) ** 2 + aqq
        arg[46] = arg[45] + qb * qb
        arg[47] = (r - qa) ** 2 + aqq
        arg[48] = arg[47] + qb * qb

        arg[49] = (r + qa - qb) ** 2 + aqq
        arg[50] = (r + qa + qb) ** 2 + aqq
        arg[51] = (r - qa - qb) ** 2 + aqq
        arg[52] = (r - qa + qb) ** 2 + aqq

        qa0 = qq[ni]
        qb0 = qq[nj]
        arg[53] = (da - qb0) ** 2 + (r - qb0) ** 2 + adq
        arg[54] = (da - qb0) ** 2 + (r + qb0) ** 2 + adq
        arg[55] = (da + qb0) ** 2 + (r - qb0) ** 2 + adq
        arg[56] = (da + qb0) ** 2 + (r + qb0) ** 2 + adq

        arg[57] = (r + qa0) ** 2 + (qa0 - db) ** 2 + aqd
        arg[58] = (r - qa0) ** 2 + (qa0 - db) ** 2 + aqd
        arg[59] = (r + qa0) ** 2 + (qa0 + db) ** 2 + aqd
        arg[60] = (r - qa0) ** 2 + (qa0 + db) ** 2 + aqd

        arg[61] = rsq + aqq + td * (qa0 - qb0) ** 2
        arg[62] = rsq + aqq + td * (qa0 + qb0) ** 2
        arg[63] = rsq + aqq + td * (qa0 * qa0 + qb0 * qb0)

        arg[64] = (r + qa0 - qb0) ** 2 + (qa0 - qb0) ** 2 + aqq
        arg[65] = (r + qa0 - qb0) ** 2 + (qa0 + qb0) ** 2 + aqq
        arg[66] = (r + qa0 + qb0) ** 2 + (qa0 - qb0) ** 2 + aqq
        arg[67] = (r + qa0 + qb0) ** 2 + (qa0 + qb0) ** 2 + aqq
        arg[68] = (r - qa0 - qb0) ** 2 + (qa0 - qb0) ** 2 + aqq
        arg[69] = (r - qa0 - qb0) ** 2 + (qa0 + qb0) ** 2 + aqq
        arg[70] = (r - qa0 + qb0) ** 2 + (qa0 - qb0) ** 2 + aqq
        arg[71] = (r - qa0 + qb0) ** 2 + (qa0 + qb0) ** 2 + aqq

        sqr = np.sqrt(arg)
        ev1, ev2, ev3, ev4 = Ha2eV/2.0, Ha2eV/4.0, Ha2eV/8.0, Ha2eV/16.0

        ee     = Ha2eV / sqr[0]
        dze    = (-ev1/sqr[1])  + ev1/sqr[2]
        qzze   =  ev2/sqr[3]    + ev2/sqr[4]   - ev1/sqr[5]
        qxxe   =  ev1/sqr[6]    - ev1/sqr[5]
        edz    = (-ev1/sqr[7])  + ev1/sqr[8]
        eqzz   =  ev2/sqr[9]    + ev2/sqr[10]  - ev1/sqr[11]
        eqxx   =  ev1/sqr[12]   - ev1/sqr[11]
        dxdx   =  ev1/sqr[13]   - ev1/sqr[14]
        dzdz   =  ev2/sqr[15]   + ev2/sqr[16]  - ev2/sqr[17] - ev2/sqr[18]
        dzqxx  =  ev2/sqr[19]   - ev2/sqr[20]  - ev2/sqr[21] + ev2/sqr[22]
        qxxdz  =  ev2/sqr[23]   - ev2/sqr[24]  - ev2/sqr[25] + ev2/sqr[26]
        dzqzz  = (-ev3/sqr[27]) + ev3/sqr[28]  - ev3/sqr[29] + ev3/sqr[30] \
                 - ev2/sqr[21]  + ev2/sqr[19]
        qzzdz  = (-ev3/sqr[31]) + ev3/sqr[32]  - ev3/sqr[33] + ev3/sqr[34] \
                 + ev2/sqr[23]  - ev2/sqr[25]
        qxxqxx =  ev3/sqr[36]   + ev3/sqr[37]  - ev2/sqr[38] - ev2/sqr[39] + ev2/sqr[35]
        qxxqyy =  ev2/sqr[40]   - ev2/sqr[38]  - ev2/sqr[39] + ev2/sqr[35]
        qxxqzz =  ev3/sqr[42]   + ev3/sqr[44]  - ev3/sqr[41] - ev3/sqr[43] - ev2/sqr[38] + ev2/sqr[35]
        qzzqxx =  ev3/sqr[46]   + ev3/sqr[48]  - ev3/sqr[45] - ev3/sqr[47] - ev2/sqr[39] + ev2/sqr[35]
        qzzqzz =  ev4/sqr[49]   + ev4/sqr[50]  + ev4/sqr[51] + ev4/sqr[52] \
                 - ev3/sqr[47]  - ev3/sqr[45]  - ev3/sqr[41] - ev3/sqr[43] + ev2/sqr[35]
        dxqxz  = (-ev2/sqr[53]) + ev2/sqr[54]  + ev2/sqr[55] - ev2/sqr[56]
        qxzdx  = (-ev2/sqr[57]) + ev2/sqr[58]  + ev2/sqr[59] - ev2/sqr[60]
        qxzqxz =  ev3/sqr[64]   - ev3/sqr[66]  - ev3/sqr[68] + ev3/sqr[70] \
                 - ev3/sqr[65]  + ev3/sqr[67]  + ev3/sqr[69] - ev3/sqr[71]

        ri[0]  = ee
        ri[1]  = -dze
        ri[2]  = ee + qzze
        ri[3]  = ee + qxxe
        ri[4]  = -edz
        ri[5]  = dzdz
        ri[6]  = dxdx
        ri[7]  = (-edz) - qzzdz
        ri[8]  = (-edz) - qxxdz
        ri[9]  = -qxzdx
        ri[10] = ee + eqzz
        ri[11] = ee + eqxx
        ri[12] = (-dze) - dzqzz
        ri[13] = (-dze) - dzqxx
        ri[14] = -dxqxz
        ri[15] = ee + eqzz + qzze + qzzqzz
        ri[16] = ee + eqzz + qxxe + qxxqzz
        ri[17] = ee + eqxx + qzze + qzzqxx
        ri[18] = ee + eqxx + qxxe + qxxqxx
        ri[19] = qxzqxz
        ri[20] = ee + eqxx + qxxe + qxxqyy
        ri[21] = half * (qxxqxx - qxxqyy)

    ri *= nri
    return ri, float(gab)

def reppd2(ni, nj, r, ri, env=None):
    """
    NumPy 版 reppd2（全 0-basis）。

    Parameters
    ----------
    ni, nj : int
        0-based atomic index
    r : float
        atomic distance(Bohr)。
    ri : (22,) array_like
        2-center local integrals
    tore : (N,) array_like
    dorbs : (N,) bool array
    indexd : (9,9) int array (0-based)
        AO 对 (i,j) → SPD index (0..44)。
    ind2 : (45,45) int array (0-based)
        (ij,kl) → (0..490)。
    isym : (491,) int array (0-based, 可带符号)
        index table; Correct the sign of integrals
    rijkl : callable
        rijkl(ni,nj,ij,kl,li,lj,lk,ll,ic,r) -> float
    to_point_fn : callable
        point,const = to_point_fn(r_angstrom)

    Returns
    -------
    rep : (491,) float64
    core : (10,2) float64
    """
    tore = env.tore
    dorbs = env.dorbs
    indexd = env.indexd
    ind2 = env.ind2
    isym = env.isym
    

    ri = np.asarray(ri, dtype=np.float64)
    rep = np.zeros(491, dtype=np.float64)
    core = np.zeros((10, 2), dtype=np.float64)

    # --- the first 34 terns are derived from local integral ri ---
    # Fortran ipos (1-based) -> 0-based：
    ipos = np.array([
        1, 5,11,12,12, 2, 6,13,14,14, 3, 8,16,18,18, 7,15,10,20, 4, 9,17,19,21,
        7,15,10,20,22, 4, 9,17,21,19
    ], dtype=int) - 1
    rep[:34] = ri[ipos]

    if not (dorbs[ni] or dorbs[nj]):
        return rep, core

    # --- AO type（i=0..8）：S(0), 3*P(1), 5*D(2) ---
    lorb = np.array([0, 1,1,1, 2,2,2,2,2], dtype=int)

    # lasti/lastk： AO (d→9，p→4，H/He→1)
    lasti = 9 if dorbs[ni] else (1 if ni < 2 else 4)   
    lastk = 9 if dorbs[nj] else (1 if nj < 2 else 4)
    
    # --- Main loop ---
    for i in range(lasti):
        li = lorb[i]
        for j in range(i + 1):
            lj = lorb[j]
            ij = int(indexd[i, j])         # 0..44
            coul_first = (i == j)

            for k in range(lastk):
                lk = lorb[k]
                for l in range(k + 1):
                    ll = lorb[l]
                    kl = int(indexd[k, l])  # 0..44

                    coulomb = coul_first and (k == l)

                    idx = int(ind2[ij, kl])  # 0..490
                    if idx <= 33:
                        continue  

                    nold = int(isym[idx])  
                    if nold >= 35:
                        rep[idx] = rep[nold-1]
                    elif nold <= -35:
                        rep[idx] = -rep[-nold-1]
                    elif nold == 0:
                        val = rijkl(ni, nj, ij, kl, li, lj, lk, ll, 0, r, env=env) * Ha2eV
                        rep[idx] = val
                    #else:
                    #    rep[idx] = rep[nold]
    # d at right hand
    if dorbs[nj]: 
        # <S S | D S>
        kl = int(indexd[4, 0])  # (5,1) -> 0-based (4,0)
        core[4, 1] = -rijkl(ni, nj, ij, kl, 0, 0, 2, 0, 1, r, env=env) * Ha2eV * tore[ni]
        # <S S | D P>
        kl = int(indexd[4, 1])  # (5,2)
        core[5, 1] = -rijkl(ni, nj, ij, kl, 0, 0, 2, 1, 1, r, env=env) * Ha2eV * tore[ni]
        # <S S | D D>
        kl = int(indexd[4, 4])  # (5,5)
        core[6, 1] = -rijkl(ni, nj, ij, kl, 0, 0, 2, 2, 1, r, env=env) * Ha2eV * tore[ni]
        # <S S | D+ P+>
        kl = int(indexd[5, 2])  # (6,3)
        core[7, 1] = -rijkl(ni, nj, ij, kl, 0, 0, 2, 1, 1, r, env=env) * Ha2eV * tore[ni]
        # <S S | D+ D+>
        kl = int(indexd[5, 5])  # (6,6)
        core[8, 1] = -rijkl(ni, nj, ij, kl, 0, 0, 2, 2, 1, r, env=env) * Ha2eV * tore[ni]
        # <S S | D# D#>
        kl = int(indexd[7, 7])  # (8,8)
        core[9, 1] = -rijkl(ni, nj, ij, kl, 0, 0, 2, 2, 1, r, env=env) * Ha2eV * tore[ni]

    # dorbs(ni):  d at left hand
    if dorbs[ni]:
        # <D S | S S>
        kl = int(indexd[4, 0])  # (5,1)
        core[4, 0] = -rijkl(ni, nj, kl, ij, 2, 0, 0, 0, 2, r, env=env) * Ha2eV * tore[nj]
        # <D P | S S>
        kl = int(indexd[4, 1])  # (5,2)
        core[5, 0] = -rijkl(ni, nj, kl, ij, 2, 1, 0, 0, 2, r, env=env) * Ha2eV * tore[nj]
        # <D D | S S>
        kl = int(indexd[4, 4])  # (5,5)
        core[6, 0] = -rijkl(ni, nj, kl, ij, 2, 2, 0, 0, 2, r, env=env) * Ha2eV * tore[nj]
        # <D+ P+ | S S>
        kl = int(indexd[5, 2])  # (6,3)
        core[7, 0] = -rijkl(ni, nj, kl, ij, 2, 1, 0, 0, 2, r, env=env) * Ha2eV * tore[nj]
        # <D+ D+ | S S>
        kl = int(indexd[5, 5])  # (6,6)
        core[8, 0] = -rijkl(ni, nj, kl, ij, 2, 2, 0, 0, 2, r, env=env) * Ha2eV * tore[nj]
        # <D# D# | S S>
        kl = int(indexd[7, 7])  # (8,8)
        core[9, 0] = -rijkl(ni, nj, kl, ij, 2, 2, 0, 0, 2, r, env=env) * Ha2eV * tore[nj]

    return rep, core

def rotmat(nj, ni, coordi, coordj, env=None):
    """
    NumPy 0-based rotmat 

    Parameters
    ----------
    nj, ni : int
        0-based atomic index for determining dorbs
    coordi, coordj : array-like, shape (3,)
    dorbs : array-like of bool

    Returns
    -------
    r   : float
    sp  : (3,3) ndarray
    pp  : (6,3,3) ndarray
    sd  : (5,5) ndarray
    dp  : (15,5,3) ndarray
    d_d : (15,5,5) ndarray
    """
    small = 1.0e-07
    pt5sq3 = 0.8660254037841  # = sqrt(3)/2

    coordi = np.asarray(coordi, dtype=float).reshape(3)
    coordj = np.asarray(coordj, dtype=float).reshape(3)
    x11, x22, x33 = (coordj - coordi)
    b = x11 * x11 + x22 * x22
    r = float(np.sqrt(b + x33 * x33))
    sqb = np.sqrt(b) if b > 0.0 else 0.0
    sb = sqb / r if r > 0.0 else 0.0

    dorbs = env.dorbs
    # angular function：sa=sin(phi), ca=cos(phi); sb=sin(theta), cb=cos(theta)
    if sb > small:
        ca = x11 / sqb
        sa = x22 / sqb
        cb = x33 / r
    else:
        sa = 0.0
        sb = 0.0
        if x33 < 0.0:
            ca = -1.0
            cb = -1.0
        elif x33 > 0.0:
            ca = 1.0
            cb = 1.0
        else:
            ca = 0.0
            cb = 0.0

    # 3×3 rotation matrix p（Fortran: p(row,col)）
    p = np.empty((3, 3), dtype=float)
    p[0, 0] = ca * sb
    p[1, 0] = ca * cb
    p[2, 0] = -sa

    p[0, 1] = sa * sb
    p[1, 1] = sa * cb
    p[2, 1] = ca

    p[0, 2] = cb
    p[1, 2] = -sb
    p[2, 2] = 0.0

    # === output matrix ===
    # p *= -1
    sp = p.copy()                 # S-P
    pp  = np.zeros((6, 3, 3))     # P-P
    sd  = np.zeros((5, 5))        # S-D
    dp  = np.zeros((15, 5, 3))    # D-P
    d_d = np.zeros((15, 5, 5))    # D-D

    # --------- P-P block ---------
    for k in range(3):  # k = 0..2
        pp[0, k, k] = p[k, 0] * p[k, 0]
        pp[1, k, k] = p[k, 1] * p[k, 1]
        pp[2, k, k] = p[k, 2] * p[k, 2]
        pp[3, k, k] = p[k, 0] * p[k, 1]
        pp[4, k, k] = p[k, 0] * p[k, 2]
        pp[5, k, k] = p[k, 1] * p[k, 2]
        if k == 0:
            continue
        # lower triangle j=0..k-1
        j_slice = slice(0, k)
        pp[0, k, j_slice] = 2.0 * p[k, 0] * p[:k, 0]
        pp[1, k, j_slice] = 2.0 * p[k, 1] * p[:k, 1]
        pp[2, k, j_slice] = 2.0 * p[k, 2] * p[:k, 2]
        pp[3, k, j_slice] = p[k, 0] * p[:k, 1] + p[k, 1] * p[:k, 0]
        pp[4, k, j_slice] = p[k, 0] * p[:k, 2] + p[k, 2] * p[:k, 0]
        pp[5, k, j_slice] = p[k, 1] * p[:k, 2] + p[k, 2] * p[:k, 1]

    if bool(dorbs[ni]) or bool(dorbs[nj]):
        c2a = 2.0 * ca * ca - 1.0
        c2b = 2.0 * cb * cb - 1.0
        s2a = 2.0 * sa * ca
        s2b = 2.0 * sb * cb

        d = np.empty((5, 5), dtype=float)
        # first column
        d[0, 0] = pt5sq3 * c2a * sb * sb
        d[1, 0] = 0.5 * c2a * s2b
        d[2, 0] = -s2a * sb
        d[3, 0] = c2a * (cb * cb + 0.5 * sb * sb)
        d[4, 0] = -s2a * cb
        # second column
        d[0, 1] = pt5sq3 * ca * s2b
        d[1, 1] = ca * c2b
        d[2, 1] = -sa * cb
        d[3, 1] = -0.5 * ca * s2b
        d[4, 1] = sa * sb
        # third column
        d[0, 2] = cb * cb - 0.5 * sb * sb
        d[1, 2] = -pt5sq3 * s2b
        d[2, 2] = 0.0
        d[3, 2] = pt5sq3 * sb * sb
        d[4, 2] = 0.0
        # fourth column
        d[0, 3] = pt5sq3 * sa * s2b
        d[1, 3] = sa * c2b
        d[2, 3] = ca * cb
        d[3, 3] = -0.5 * sa * s2b
        d[4, 3] = -ca * sb
        # fifth column
        d[0, 4] = pt5sq3 * s2a * sb * sb
        d[1, 4] = 0.5 * s2a * s2b
        d[2, 4] = c2a * sb
        d[3, 4] = s2a * (cb * cb + 0.5 * sb * sb)
        d[4, 4] = c2a * cb

        sd = d.copy()  # S-D

        # D-P：15×5×3
        for k in range(5):  # k = 0..4
            dp[0,  k, :] = d[k, 0] * p[:, 0]
            dp[1,  k, :] = d[k, 0] * p[:, 1]
            dp[2,  k, :] = d[k, 0] * p[:, 2]
            dp[3,  k, :] = d[k, 1] * p[:, 0]
            dp[4,  k, :] = d[k, 1] * p[:, 1]
            dp[5,  k, :] = d[k, 1] * p[:, 2]
            dp[6,  k, :] = d[k, 2] * p[:, 0]
            dp[7,  k, :] = d[k, 2] * p[:, 1]
            dp[8,  k, :] = d[k, 2] * p[:, 2]
            dp[9,  k, :] = d[k, 3] * p[:, 0]
            dp[10, k, :] = d[k, 3] * p[:, 1]
            dp[11, k, :] = d[k, 3] * p[:, 2]
            dp[12, k, :] = d[k, 4] * p[:, 0]
            dp[13, k, :] = d[k, 4] * p[:, 1]
            dp[14, k, :] = d[k, 4] * p[:, 2]

        # D-D
        for k in range(5):  # 0..4
            d_d[0,  k, k] = d[k, 0] * d[k, 0]
            d_d[1,  k, k] = d[k, 1] * d[k, 1]
            d_d[2,  k, k] = d[k, 2] * d[k, 2]
            d_d[3,  k, k] = d[k, 3] * d[k, 3]
            d_d[4,  k, k] = d[k, 4] * d[k, 4]
            d_d[5,  k, k] = d[k, 0] * d[k, 1]
            d_d[6,  k, k] = d[k, 0] * d[k, 2]
            d_d[7,  k, k] = d[k, 1] * d[k, 2]
            d_d[8,  k, k] = d[k, 0] * d[k, 3]
            d_d[9,  k, k] = d[k, 1] * d[k, 3]
            d_d[10, k, k] = d[k, 2] * d[k, 3]
            d_d[11, k, k] = d[k, 0] * d[k, 4]
            d_d[12, k, k] = d[k, 1] * d[k, 4]
            d_d[13, k, k] = d[k, 2] * d[k, 4]
            d_d[14, k, k] = d[k, 3] * d[k, 4]
            if k == 0:
                continue
            j_slice = slice(0, k)
            d_d[0,  k, j_slice] = 2.0 * d[k, 0] * d[:k, 0]
            d_d[1,  k, j_slice] = 2.0 * d[k, 1] * d[:k, 1]
            d_d[2,  k, j_slice] = 2.0 * d[k, 2] * d[:k, 2]
            d_d[3,  k, j_slice] = 2.0 * d[k, 3] * d[:k, 3]
            d_d[4,  k, j_slice] = 2.0 * d[k, 4] * d[:k, 4]
            d_d[5,  k, j_slice] = d[k, 0] * d[:k, 1] + d[k, 1] * d[:k, 0]
            d_d[6,  k, j_slice] = d[k, 0] * d[:k, 2] + d[k, 2] * d[:k, 0]
            d_d[7,  k, j_slice] = d[k, 1] * d[:k, 2] + d[k, 2] * d[:k, 1]
            d_d[8,  k, j_slice] = d[k, 0] * d[:k, 3] + d[k, 3] * d[:k, 0]
            d_d[9,  k, j_slice] = d[k, 1] * d[:k, 3] + d[k, 3] * d[:k, 1]
            d_d[10, k, j_slice] = d[k, 2] * d[:k, 3] + d[k, 3] * d[:k, 2]
            d_d[11, k, j_slice] = d[k, 0] * d[:k, 4] + d[k, 4] * d[:k, 0]
            d_d[12, k, j_slice] = d[k, 1] * d[:k, 4] + d[k, 4] * d[:k, 1]
            d_d[13, k, j_slice] = d[k, 2] * d[:k, 4] + d[k, 4] * d[:k, 2]
            d_d[14, k, j_slice] = d[k, 3] * d[:k, 4] + d[k, 4] * d[:k, 3]
    
    return r, sp, pp, sd, dp, d_d

def ccrep_pm6(ni, nj, r_bohr, gab, eps_div=1e-12, env=None):
    """
    Calculate PM6 core-core repulsion (eV).

    Parameters
    ----------
    ni, nj : int
    r_bohr : float。
    gab : float
        mono-pole term 
    a0 : float
    alp, tore : ndarray
    guess1, guess2, guess3 : (nZ, 4) ndarray
    alpb, xfac : ndarray
    par1..par4 : float
        empirical correction; PM6 would use par1, par2, par3, par4.
    eps_div : float

    Returns
    -------
    enuclr : float
        
    """
    # 1) Bohr -> Å
    r = float(r_bohr) * float(a0)
    tore = env.tore
    xfac = env.xfac
    alpb = env.alpb
    guess1 = env.gues61
    guess2 = env.gues62
    guess3 = env.gues63
    par1 = env.v_par6[0]   # Used in ccrep for scalar correction of C-C triple bonds.
    par2 = env.v_par6[1]   # Used in ccrep for exponent correction of C-C triple bonds.
    par3 = env.v_par6[2]
    par4 = env.v_par6[3]
    # 2) monopole term 
    enuc = float(tore[ni]) * float(tore[nj]) * float(gab)
    # 3) bonding 
    fff = float(xfac[ni, nj])
    has_bond_params = abs(fff) > 1e-5

    # 4) scale term in PM6 
    if has_bond_params:
        abond = float(alpb[ni, nj])
        if abond < 1e-6:
            abond = 1.2  

        scale = 1.0 + 2.0 * fff * np.exp(-abond * (r + 0.0003 * r**6))

        # specific correction 
        i_big = max(ni, nj)
        j_small = min(ni, nj)

        # H-X（CH/NH/OH）
        if j_small == 0:  # H
            if i_big in (5, 6):   # C or N（0-basis）
                scale = 1.0 + 2.0 * fff * np.exp(-abond * r**2)
            elif i_big == 7:      # O（0-basis）
                scale = 1.0 + 2.0 * fff * np.exp(-abond * r**2) - par3 * np.exp(-2.0 * par4 * r)

        # C≡C triple bond
        if j_small == 5 and i_big == 5:  # C-C（0-basis）
            scale = scale + par1 * np.exp(-par2 * r)

        # Si-O long-range weak interaction correction
        if j_small == 7 and i_big == 13:  # O-Si（0-basis：O=7, Si=13）
            scale = scale - 0.7e-3 * np.exp(-(r - 2.9)**2)

        enuclr = enuc * scale
    else:
        # general core-core term without bonding parameters 
        in_f_block = (56 <= ni <= 70) or (56 <= nj <= 70)
        k = 3.0 if in_f_block else 2.18
        scale = 10.0 * np.exp(-k * r)
        enuclr = abs(scale * enuc) + enuc

    # 5) VdW / Gaussian correction（PM6）
    scale_vdw = 0.0
    invr = 1.0 / max(r, eps_div)
    for Z in (ni, nj):
        ax = float(guess2[Z, 0]) * (r - float(guess3[Z, 0]))**2
        if ax < 25.0:
            scale_vdw += float(tore[ni]) * float(tore[nj]) * invr * float(guess1[Z, 0]) * np.exp(-ax)

    i_max = 0 if has_bond_params and (abond > 1e-4) else 4

    for ig in range(i_max):  # ig=0..3 <=> Fortran 1..4
        g1 = float(guess1[ni, ig])
        if g1 != 0.0:
            ax = float(guess2[ni, ig]) * (r - float(guess3[ni, ig]))**2
            if ax <= 25.0:
                scale_vdw += float(tore[ni]) * float(tore[nj]) * invr * g1 * np.exp(-ax)
        g1 = float(guess1[nj, ig])
        if g1 != 0.0:
            ax = float(guess2[nj, ig]) * (r - float(guess3[nj, ig]))**2
            if ax <= 25.0:
                scale_vdw += float(tore[ni]) * float(tore[nj]) * invr * g1 * np.exp(-ax)

    enuclr = enuclr + scale_vdw
    # 6) additional “12” potentials (short distance)
    zi = (ni + 1.0)**(0.3333) # (1.0 / 3.0) 
    zj = (nj + 1.0)**(0.3333) # (1.0 / 3.0)   # follow mopac: 0.3333. numerical inconsistence would be generated if 1/3 is used
    ax = r / (zi + zj)
    if ax < 3.0:
        lj12 = 1.0e-8 / (ax**12)
        enuclr = enuclr + min(lj12, 1.0e5)
    return float(enuclr)

def spcore(ni, nj, r_bohr, env=None):
    """
    NumPy 0 based  spcore

    Parameters
    ----------
    ni, nj : int
    r_bohr : floa
    ev : float
    tore : (nZ,) ndarray
    po : (>=9, nZ) ndarray
    ddp : (>=4, nZ) ndarray

    Returns
    -------
    core : (10, 2) ndarray
    """
    core = np.zeros((10, 2), dtype=float)
    po = env.po
    tore = env.tore
    ddp = env.ddp


    r = float(r_bohr)
    r2 = r * r
    
    # Fortran: aci = po(9,ni); acj = po(9,nj)
    aci = float(po[8, ni])
    acj = float(po[8, nj])

    # SS-core（Fortran: ssi=(aci+po(1,nj))^2; ssj=(acj+po(1,ni))^2）
    ssi = (aci + float(po[0, nj])) ** 2
    ssj = (acj + float(po[0, ni])) ** 2
    core[0, 0] = -float(tore[nj]) * Ha2eV / np.sqrt(r2 + ssj)  # core(1,1)
    core[0, 1] = -float(tore[ni]) * Ha2eV / np.sqrt(r2 + ssi)  # core(1,2)

    # heavy atoms?
    heavy_i = (ni >= 2)
    heavy_j = (nj >= 2)

    pxy = np.array([1.0, -0.5, -0.5, 0.5, 0.25, 0.25, 0.5], dtype=float)

    if heavy_i:
        # Fortran:
        # ppj=(acj+po(7,ni))^2; da=ddp(2,ni); qa=ddp(3,ni)/sqrt(2)
        # adj=(po(2,ni)+acj)^2; aqj=(po(3,ni)+acj)^2
        ppj = (acj + float(po[6, ni])) ** 2
        da  = float(ddp[1, ni])
        qa  = float(ddp[2, ni]) / np.sqrt(2.0)
        twoqa = 2.0 * qa
        adj = (float(po[1, ni]) + acj) ** 2
        aqj = (float(po[2, ni]) + acj) ** 2

        xj = np.empty(7, dtype=float)
        xj[0] = r2 + ppj
        xj[1] = r2 + aqj
        xj[2] = (r + da) ** 2 + adj
        xj[3] = (r - da) ** 2 + adj
        xj[4] = (r - twoqa) ** 2 + aqj
        xj[5] = (r + twoqa) ** 2 + aqj
        xj[6] = r2 + twoqa * twoqa + aqj
        xj = pxy / np.sqrt(xj)

        aj2 = (xj[2] + xj[3]) * Ha2eV
        aj3 = (xj[0] + xj[1] + xj[4] + xj[5]) * Ha2eV
        aj4 = (xj[0] + xj[1] + xj[6]) * Ha2eV
        core[1, 0] = -float(tore[nj]) * aj2  # core(2,1)
        core[2, 0] = -float(tore[nj]) * aj3  # core(3,1)
        core[3, 0] = -float(tore[nj]) * aj4  # core(4,1)

    if heavy_j:
        # Fortran：
        # ppi=(aci+po(7,nj))^2; db=ddp(2,nj); qb=ddp(3,nj)/sqrt(2)
        # adi=(po(2,nj)+aci)^2; aqi=(po(3,nj)+aci)^2
        ppi = (aci + float(po[6, nj])) ** 2
        db  = float(ddp[1, nj])
        qb  = float(ddp[2, nj]) / np.sqrt(2.0)
        twoqb = 2.0 * qb
        adi = (float(po[1, nj]) + aci) ** 2
        aqi = (float(po[2, nj]) + aci) ** 2

        xi = np.empty(7, dtype=float)
        xi[0] = r2 + ppi
        xi[1] = r2 + aqi
        xi[2] = (r + db) ** 2 + adi
        xi[3] = (r - db) ** 2 + adi
        xi[4] = (r - twoqb) ** 2 + aqi
        xi[5] = (r + twoqb) ** 2 + aqi
        xi[6] = r2 + twoqb * twoqb + aqi
        xi = pxy / np.sqrt(xi)

        ai2 = -(xi[2] + xi[3]) * Ha2eV
        ai3 =  (xi[0] + xi[1] + xi[4] + xi[5]) * Ha2eV
        ai4 =  (xi[0] + xi[1] + xi[6]) * Ha2eV
        core[1, 1] = -float(tore[ni]) * ai2  # core(2,2)
        core[2, 1] = -float(tore[ni]) * ai3  # core(3,2)
        core[3, 1] = -float(tore[ni]) * ai4  # core(4,2)
    return core

def tx(ii, kk, rep, sp, pp, sd, dp, d_d, env=None):
    ind2   = env.ind2   # (45,45) 
    indx   = env.indx   # (9,9)   -> 0..44
    indexd = env.indexd # (9,9)   -> 0..44

    met = np.array(
        [1,2,3,2,3,3,2,3,3,3,4,5,5,5,6,4,5,5,5,6,6,4,5,5,5,6,6,6,4,5,5,5,6,6,6,6,4,5,5,5,6,6,6,6,6],
        dtype=int
    )

    v    = np.zeros((45,45), dtype=float)
    logv = np.zeros((45,45), dtype=bool)

    limkl0 = indx[kk-1, kk-1]  
    
    for i1 in range(ii):                 # 0..ii-1
        for j1 in range(i1+1):           # 0..i1
            ij = indexd[i1, j1]          # 0..44
            for k1 in range(kk):               # 0..kk-1
                for l1 in range(k1+1):         # 0..k1 
                    kl = indexd[k1, l1]        # 0..44
                    nd = int(ind2[ij, kl])     
                    if nd == -1:
                        continue
                    wrepp = float(rep[nd])   
                    ll0 = indx[k1, l1]         # 0..44
                    mm  = int(met[ll0])        # 1..6
                    if   mm == 1:
                        v[ij, 0] = wrepp

                    elif mm == 2:  # SO
                        k = k1  - 1          # 0..2
                        v[ij, 1] += sp[k, 0] * wrepp
                        v[ij, 3] += sp[k, 1] * wrepp
                        v[ij, 6] += sp[k, 2] * wrepp

                    elif mm == 3:  # PP
                        k = k1 - 1           # 0..2
                        l = l1 - 1           # 0..2
                        v[ij, 2] += pp[0, k, l] * wrepp
                        v[ij, 5] += pp[1, k, l] * wrepp
                        v[ij, 9] += pp[2, k, l] * wrepp
                        v[ij, 4] += pp[3, k, l] * wrepp
                        v[ij, 7] += pp[4, k, l] * wrepp
                        v[ij, 8] += pp[5, k, l] * wrepp

                    elif mm == 4:  # DS
                        k = k1 - 4        # 0..4
                        v[ij,10] += sd[k,0] * wrepp
                        v[ij,15] += sd[k,1] * wrepp
                        v[ij,21] += sd[k,2] * wrepp
                        v[ij,28] += sd[k,3] * wrepp
                        v[ij,36] += sd[k,4] * wrepp

                    elif mm == 5:  # DP
                        k = k1 - 4        # 0..4
                        l = l1 - 1          # 0..2
                        v[ij,11] += dp[ 0, k, l] * wrepp
                        v[ij,12] += dp[ 1, k, l] * wrepp
                        v[ij,13] += dp[ 2, k, l] * wrepp
                        v[ij,16] += dp[ 3, k, l] * wrepp
                        v[ij,17] += dp[ 4, k, l] * wrepp
                        v[ij,18] += dp[ 5, k, l] * wrepp
                        v[ij,22] += dp[ 6, k, l] * wrepp
                        v[ij,23] += dp[ 7, k, l] * wrepp
                        v[ij,24] += dp[ 8, k, l] * wrepp
                        v[ij,29] += dp[ 9, k, l] * wrepp
                        v[ij,30] += dp[10, k, l] * wrepp
                        v[ij,31] += dp[11, k, l] * wrepp
                        v[ij,37] += dp[12, k, l] * wrepp
                        v[ij,38] += dp[13, k, l] * wrepp
                        v[ij,39] += dp[14, k, l] * wrepp

                    elif mm == 6:  # DD
                        k = k1 - 4        # 0..4
                        l = l1 - 4        # 0..4
                        v[ij,14] += d_d[ 0, k, l] * wrepp
                        v[ij,20] += d_d[ 1, k, l] * wrepp
                        v[ij,27] += d_d[ 2, k, l] * wrepp
                        v[ij,35] += d_d[ 3, k, l] * wrepp
                        v[ij,44] += d_d[ 4, k, l] * wrepp
                        v[ij,19] += d_d[ 5, k, l] * wrepp
                        v[ij,25] += d_d[ 6, k, l] * wrepp
                        v[ij,26] += d_d[ 7, k, l] * wrepp
                        v[ij,32] += d_d[ 8, k, l] * wrepp
                        v[ij,33] += d_d[ 9, k, l] * wrepp
                        v[ij,34] += d_d[10, k, l] * wrepp
                        v[ij,40] += d_d[11, k, l] * wrepp
                        v[ij,41] += d_d[12, k, l] * wrepp
                        v[ij,42] += d_d[13, k, l] * wrepp
                        v[ij,43] += d_d[14, k, l] * wrepp

            if limkl0 >= 0:
                row = v[ij, :limkl0+1]
                logv[ij, :limkl0+1] = (row != 0.0)

    return v, logv

def w2mat(ww, w, kr, limij, limkl):
    """
    NumPy w2mat: store ww(kl, ij) into one-dimensional w and update current position kr。

    Parameters
    ----------
    ww : array_like
        - 2D: shape (limkl, limij)
        - 1D: length limkl*limij
    limij : int
        upper limit of ij 
    limkl : int
        upper limit of kl 
    kr : int, optional
    w : ndarray or None

    Returns
    -------
    w_out : (limij*limkl,) ndarray
    kr_new : int
        kr + limij*limkl
    """
    L = int(limij) * int(limkl)

    # normalize ww -> 1D 
    ww = np.asarray(ww)
    if ww.ndim == 2:
        if ww.shape != (limkl, limij):
            raise ValueError(f"ww.shape should be ({limkl}, {limij}), got {ww.shape}")
        flat = ww.flatten(order='F')  # Fortran 顺序：先列后行 => (kl 内层, ij 外层)
    elif ww.ndim == 1:
        if ww.size != L:
            raise ValueError(f"ww length should be {L}, got {ww.size}")
        flat = ww
    else:
        raise ValueError("ww must be 1D or 2D")

    if w is None:
        w_out = flat.copy()
    else:
        if w.ndim != 1 or w.size < L:
            raise ValueError("w must be 1D with size >= limij*limkl")
        w_out = w
        w_out[kr:kr+L] = flat

    return  kr + L


# ---------------------------
# wstore: one-center block in 2-electron integrals
# ---------------------------
def wstore_np(w_buf, kr, ni, ilim, env):
    ni = ni - 1  # called in hcore with the input of 1-based atomic number
    block = w_buf[kr:kr+ilim*ilim].reshape((ilim, ilim), order='F')

    block[:, :] = 0.0
    block[0, 0] = env.gss6[ni]

    if env.natorb[ni] > 2:
        ip0 = 0
        ipx, ipy, ipz = ip0 + 2, ip0 + 5, ip0 + 9
        gsp = float(env.gsp6[ni]); gpp = float(env.gpp6[ni])
        gp2 = float(env.gp26[ni]); hsp = float(env.hsp6[ni])

        # s-p
        block[ipx, ip0] = block[ip0, ipx] = gsp
        block[ipy, ip0] = block[ip0, ipy] = gsp
        block[ipz, ip0] = block[ip0, ipz] = gsp
        # p-p diag
        block[ipx, ipx] = block[ipy, ipy] = block[ipz, ipz] = gpp
        # p-p offdiag
        block[ipy, ipx] = block[ipx, ipy] = gp2
        block[ipz, ipx] = block[ipx, ipz] = gp2
        block[ipz, ipy] = block[ipy, ipz] = gp2
        # hsp
        block[ip0+1, ip0+1] = hsp
        block[ip0+3, ip0+3] = hsp
        block[ip0+6, ip0+6] = hsp
        # 0.5*(gpp-gp2)
        val = 0.5*(gpp - gp2)
        block[ip0+4, ip0+4] = val
        block[ip0+7, ip0+7] = val
        block[ip0+8, ip0+8] = val

        # d one-center（243 terms）
        if ilim > 10:
            ij = env.intij - 1
            kl = env.intkl - 1
            rp = env.intrep - 1
            block[ij, kl] = env.repd[rp, ni]

    return kr + ilim*ilim


def elenuc(
    ia, ib, ja, jb, h,
    sp, sd, pp, dp, d_d, cored,
    env=None
):
    """
    electron-nucli interaction energy

    Parameters
    ----------
    ia, ib, ja, jb : int
          first block : i ∈ [ia, ib]
          second block : i ∈ [ja, jb]
    h : (mpack,) ndarray
        upper triangle
      sp : (3,3) ndarray
      sd : (5,5) ndarray
      pp : (6,3,3) ndarray
      dp : (15,5,3) ndarray
      d_d : (15,5,5) ndarray
      cored : (10,2) ndarray
      indpp : (3,3) ndarray[int]
      inddp : (5,3) ndarray[int]
      inddd : (5,5) ndarray[int]

    """
    indpp = env.indpp
    inddp = env.inddp
    inddd = env.inddd

    indpp = np.asarray(indpp, dtype=int)
    inddp = np.asarray(inddp, dtype=int)
    inddd = np.asarray(inddd, dtype=int)

    def add_block(k1b, l1b, n_col):
        """ add block of electron-nucli interaction energy """
        # k1b..l1b 为 1-basis；ind1/ind2 
        for i in range(k1b, l1b + 1):
            ind1 = i - k1b  # 0,1,2,...  
            for j in range(k1b, i + 1):
                ind2 = j - k1b
                m_1b = (i * (i - 1)) // 2 + j      # Fortran-like (1-basis)
                m = m_1b - 1                       # Python 0-basis

                if ind1 == 0:
                    # j==k1b（ind2==0） case
                    if ind2 == 0:
                        # -- (SS/)
                        h[m] += cored[0, n_col]
                    else:
                        if ind2 < 4:
                            ipp = int(indpp[ind1 - 1, ind2 - 1])  
                            h[m] += ( cored[2, n_col] * pp[ipp, 0, 0]
                                     + cored[3, n_col] * (pp[ipp, 1, 1] + pp[ipp, 2, 2]) )
                        else:
                            idd = int(inddd[ind1 - 4, ind2 - 4])  
                            h[m] += ( cored[6, n_col] * d_d[idd, 0, 0]
                                     + cored[8, n_col] * (d_d[idd, 1, 1] + d_d[idd, 2, 2])
                                     + cored[9, n_col] * (d_d[idd, 3, 3] + d_d[idd, 4, 4]) )
                else:
                    if ind1 < 4:
                        # ---- P column
                        if ind2 == 0:
                            # -- (SP/)
                            h[m] += sp[0, ind1 - 1] * cored[1, n_col]
                        elif ind2 < 4:
                            # -- (PP/)
                            ipp = int(indpp[ind1 - 1, ind2 - 1])      # 0..5
                            h[m] += ( cored[2, n_col] * pp[ipp, 0, 0]
                                     + cored[3, n_col] * (pp[ipp, 1, 1] + pp[ipp, 2, 2]) )
                        else:
                            # -- (P and D)
                            idd = int(inddd[ind1 - 4, ind2 - 4])
                            h[m] += ( cored[6, n_col] * d_d[idd, 0, 0]
                                     + cored[8, n_col] * (d_d[idd, 1, 1] + d_d[idd, 2, 2])
                                     + cored[9, n_col] * (d_d[idd, 3, 3] + d_d[idd, 4, 4]) )
                    else:
                        # ---- D column
                        if ind2 == 0:
                            # -- (SD/)
                            h[m] += sd[0, ind1 - 4] * cored[4, n_col]
                        elif ind2 < 4:
                            # -- (PD/)
                            idp = int(inddp[ind1 - 4, ind2 - 1])      # 0..14
                            h[m] += ( cored[5, n_col] * dp[idp, 0, 0]
                                     + cored[7, n_col] * (dp[idp, 1, 1] + dp[idp, 2, 2]) )
                        else:
                            # -- (DD/)
                            idd = int(inddd[ind1 - 4, ind2 - 4])      # 0..14
                            h[m] += ( cored[6, n_col] * d_d[idd, 0, 0]
                                     + cored[8, n_col] * (d_d[idd, 1, 1] + d_d[idd, 2, 2])
                                     + cored[9, n_col] * (d_d[idd, 3, 3] + d_d[idd, 4, 4]) )

    add_block(ia, ib, n_col=0)
    add_block(ja, jb, n_col=1)
                            
def rotatd(ni, nj, ci, cj, w, kr, env=None):
    """
    NumPy  rotatd
    Return:
        w (the input array will be written/modified in place)
        enuc (nucleus-nucleus repulsion energy, in eV scale)
    """
    ### transforme atomic number into 0-based 
    ni = ni - 1
    nj = nj - 1 
    
    tore = env.tore
    natorb = env.natorb
    indx = env.indx
    indexd = env.indexd
    inddd = env.inddd
    # --- 1) rotate matrix  ---
    r_ang, sp, pp, sd, dp, d_d = rotmat(nj, ni, ci, cj, env=env)

    # --- 2) 2 center local integrals（22 terms）and G_AB（used in core-core） ---
    ri, gab = reppd(ni, nj, r_ang, env=env)  # 已内部用 a0 归一 , 确定与fortran对齐

    # --- 3) unit transformation: r Bohr ---
    r_bohr = r_ang / a0
    cored = spcore(ni, nj, r_bohr, env=env)                 # (10,2) 确定与fortran对齐

    c = 1.0
    one_mc = 1.0 - c

    # --- 5) 加入 d 相关两中心 & rep(491) ---
    rep, core_d = reppd2(ni, nj, r_bohr, ri, env=env)       # core_d 
    cored = np.array(cored, dtype=float, copy=True)
    if core_d is not None:
        cored[4:, :] = core_d[4:, :]

    # --- 6) feather：smoothen the core  ---
    # right（column 0）：point = -(eV/r)*tore(nj)
    point = -(Ha2eV / r_bohr) * tore[nj]
    idx_add = [0, 2, 3, 6, 8, 9]   # 1,3,4,7,9,10 -> 0-based
    idx_mul = [1, 4, 5, 7]         # 2,5,6,8
    cored[idx_add, 0] = cored[idx_add, 0] * c + one_mc * point
    cored[idx_mul, 0] = cored[idx_mul, 0] * c
    
    # left（column 1）：point = -(eV/r)*tore(ni)
    point = -(Ha2eV / r_bohr) * tore[ni]
    cored[idx_add, 1] = cored[idx_add, 1] * c + one_mc * point
    cored[idx_mul, 1] = cored[idx_mul, 1] * c


    # --- 7) assemble AO-pair ww（Max length 2025=45*45） ---
    ii = int(natorb[ni])  # 1 / 4 / 9
    kk = int(natorb[nj])  # 1 / 4 / 9
    ww = np.zeros(2025, dtype=float)   # temporary ww 
    if ii * kk > 0:
        limij = ii * (ii + 1) // 2
        limkl = kk * (kk + 1) // 2   # 45 
        ww = ww[:limij*limkl]  

        v, logv = tx(ii, kk, rep, sp, pp, sd, dp, d_d, env=env)  # bool(45,45), float(45,45)
        
        met = np.array([
            1,2,3,2,3,3,2,3,3,3, 4,5,5,5,6, 4,5,5,5,6,6, 4,5,5,5,6,6,6,
            4,5,5,5,6,6,6,6, 4,5,5,5, 6,6,6,6,6
        ], dtype=int)  # 45 terms

        # Fortran indw(i,j) (1-based) -> Python 0-based
        def indw_1b(i1b, j1b, kl0b):
            return int(indx[i1b - 1, j1b - 1]) * limkl + kl0b  # 0-based

        # Main loop
        for i1 in range(1, ii + 1):
            for j1 in range(1, i1 + 1):
                ij = int(indexd[i1 - 1, j1 - 1])         # 0..44
                jj = int(indx[i1 - 1, j1 - 1])           # 0..(limij-1)
                mm = int(met[jj])

                for k in range(1, kk + 1):
                    for l in range(1, k + 1):
                        kl = int(indx[k - 1, l - 1])     # 0..(limkl-1)
                        if not logv[ij, kl]:
                            continue
                        wrepp = float(v[ij, kl])

                        if mm == 1:
                            iw = indw_1b(1, 1, kl)
                            ww[iw] = wrepp

                        elif mm == 2:
                            # S-P
                            for i in range(1, 4):
                                iw = indw_1b(i + 1, 1, kl)
                                ww[iw] += sp[i1 - 2, i - 1] * wrepp  # sp(row=i1-2, col=i-1)

                        elif mm == 3:
                            # P-P
                            for i in range(1, 4):
                                cc = pp[i - 1, i1 - 2, j1 - 2]
                                iw = indw_1b(i + 1, i + 1, kl)
                                ww[iw] += cc * wrepp
                                for j in range(1, i):
                                    cc = pp[(1 + i + j) - 1, i1 - 2, j1 - 2]  # 1+i+j -> 0-based
                                    iw = indw_1b(i + 1, j + 1, kl)
                                    ww[iw] += cc * wrepp

                        elif mm == 4:
                            # S-D
                            for i in range(1, 6):
                                iw = indw_1b(i + 4, 1, kl)
                                ww[iw] += sd[i1 - 5, i - 1] * wrepp  # sd(row=i1-5, col=i-1)

                        elif mm == 5:
                            # D-P
                            for i in range(1, 6):
                                for j in range(1, 4):
                                    iw = indw_1b(i + 4, j + 1, kl)
                                    ij1_py = 3 * (i - 1) + (j - 1)     # 0..14
                                    ww[iw] += dp[ij1_py, i1 - 5, j1 - 2] * wrepp

                        elif mm == 6:
                            # D-D
                            for i in range(1, 6):
                                cc = d_d[i - 1, i1 - 5, j1 - 5]
                                iw = indw_1b(i + 4, i + 4, kl)
                                ww[iw] += cc * wrepp
                                for j in range(1, i):
                                    ij1_py = int(inddd[i - 1, j - 1])  # 0..14
                                    cc = d_d[ij1_py, i1 - 5, j1 - 5]
                                    iw = indw_1b(i + 4, j + 4, kl)
                                    ww[iw] += cc * wrepp
    # --- 9) add local ww to global w ---
    iw = (ii * (ii + 1))/ 2
    jw = (kk * (kk + 1))// 2

    kr = w2mat(ww, w, kr, iw, jw)
    enuc = ccrep_pm6(ni, nj, r_bohr, gab, env=env)   

    return kr, float(enuc), sp, sd, pp, dp, d_d, cored

def rotate_np(ni, nj, xi, xj, w_buf, kr, env=None):
    """
    NumPy rotate（0-based）。

    Parameters
    ----------
    ni, nj : int
        1_based atomic number
    xi, xj : array_like shape (3,)
        cartesian coordinates of atom i and j (Å)
    w_buf : np.ndarray (1D)
        global w buffer (1D)
    kr : int
        current position of w_buf to write

    Keyword-only
    ------------
    env : object
        Must provide:
        - natorb : (nZ,)  Number of AOs for each element (1/4/9)
        - And the tensors used by elenuc_fn (sp/sd/pp/dp/d_d, as well as
            env.cored(10,2) for the current atom pair, which is written by rotatd)

    rotatd_fn : callable
        rotatd_fn(ni, nj, xi, xj, w_buf, kr, env) -> (kr_new, enuc)
        Responsible for computing two-electron integrals after transforming from the
        local frame to the molecular frame, and writing the 45×45 = 2025 terms in
        column-major order to w_buf[kr : kr + 2025]. Returns the new kr and enuc (eV).

    elenuc_fn : callable
        elenuc_fn(ia, ib, ja, jb, h_packed, env=env) -> None
        Accumulates the nuclear stabilization terms for the current pair into the
        packed vector h_packed (length sufficient to hold (li+lj)*(li+lj+1)/2, up to 171).
        Requires env.cored to have been written in advance by rotatd_fn.

    Returns
    -------
    kr_new : int
    e1b : (li*(li+1)//2,) ndarray
    e2a : (lj*(lj+1)//2,) ndarray
    enuc : float
    """

    xi = np.asarray(xi, float)
    xj = np.asarray(xj, float)

    dx = xj - xi
    rij2 = np.dot(dx, dx)
    if rij2 < 2.0e-5:  
        # lower limit of the atomic distance: set all w to 0 and do not advance kr (consistent with Fortran)
        # Fortran sets w(2025) = 0 when rij < 2.0e-5; here we clear the 2025 slot (if any)
        if w_buf is not None and w_buf.ndim == 1 and w_buf.size >= kr + 2025:
            w_buf[kr:kr+2025] = 0.0
        li = int(env.natorb[ni])
        lj = int(env.natorb[nj])
        e1b = np.zeros(li*(li+1)//2, dtype=float)
        e2a = np.zeros(lj*(lj+1)//2, dtype=float)
        enuc = 0.0
        return kr, e1b, e2a, enuc

    # 1) 2-electron term generated by rotatd function
    kr_new, enuc, sp, sd, pp, dp, d_d, cored = rotatd(ni, nj, xi, xj, w_buf, kr, env=env)
    # 2) 1-electron core attraction term generated by elenuc
    li = env.natorb[ni-1]
    lj = env.natorb[nj-1]

    # Maximum length, considering li=lj=9 
    h_pack = np.zeros(171, dtype=float) # np.zeros((li + lj) * (li + lj + 1) // 2, dtype=float)

    #   first block : 1..li
    #   second block : li+1 .. li+lj
    elenuc(1, li, li + 1, li + lj, h_pack, sp, sd, pp, dp, d_d, cored, env=env)
    # 3) extract e1b/e2a from h_pack
    # e1b
    e1b = np.empty(li*(li+1)//2, dtype=float)
    t = 0
    for i in range(1, li+1):
        base = i*(i-1)//2
        cnt = i
        e1b[t:t+cnt] = h_pack[base:base+cnt]
        t += cnt

    # e2a: sub-block (li+1..li+lj, li+1..li+lj) of h_pack
    e2a = np.empty(lj*(lj+1)//2, dtype=float)
    t = 0
    for i in range(li+1, li+lj+1):
        base = i*(i-1)//2
        cnt = i - li
        e2a[t:t+cnt] = h_pack[base + (li): base + (li) + cnt]
        t += cnt

    return kr_new, e1b, e2a, enuc

