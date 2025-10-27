import numpy as np
from ase.data import covalent_radii
#### QYZ: 其余分支仍需仔细查对，确保正确 
# ====== 基础工具 ======
def _as_lookup(x):
    """允许 x 是函数或 ndarray；统一成按 Z 查值的可调用。"""
    if callable(x):
        return x
    arr = np.asarray(x)
    return lambda Z: arr[Z]

def _read_xyz(xyz_file):
    """
    读取 .xyz，返回:
      Z:  (N,)   原子序号（int）
      R:  (3,N)  直角坐标，Å（3×N）
    简单解析：假设行格式"El x y z"。
    """
    with open(xyz_file, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
    n = int(lines[0].strip())
    Z = []
    R = np.zeros((3, n), dtype=float)
    # 元素到Z的简单表（可自行扩展或换成你项目里的表）
    _ptable = {'H':1,'C':6,'N':7,'O':8,'F':9,'Si':14,'P':15,'S':16,'Cl':17}
    for i, line in enumerate(lines[2:2+n]):
        toks = line.split()
        sym = toks[0]
        z = _ptable.get(sym, int(sym) if sym.isdigit() else 0)
        Z.append(z)
        R[:, i] = [float(toks[1]), float(toks[2]), float(toks[3])]
    return np.array(Z, dtype=int), R

def distance_euclid(R3N, i, j):
    """欧氏距离；R3N 形状 (3,N)。"""
    d = R3N[:, i] - R3N[:, j]
    return float(np.linalg.norm(d))

def dang(x1, y1, x2, y2):
    """返回 2D 有向夹角 [-pi, pi]。"""
    det = x1 * y2 - y1 * x2
    dot = x1 * x2 + y1 * y2
    return np.arctan2(det, dot)

# ====== 邻接表（无 PBC，3×N 坐标）======
def build_bonds_simple(Z, R3N, radii, scale=1.2, max_nb=12):
    """
    用共价半径阈值构建 nbonds/ibonds（0-based）。
    Z: (N,), R3N: (3,N), radii: (maxZ+1,) 给出每个 Z 的 covalent 半径（Å）
    返回：
      nbonds: (N,)
      ibonds: (max_nb, N)  填不满处为 -1
    """
    Z = np.asarray(Z, dtype=int)
    N = Z.size
    nbonds = np.zeros(N, dtype=int)
    ibonds = -np.ones((max_nb, N), dtype=int)
    for i in range(N):
        Ri = R3N[:, i]
        for j in range(i+1, N):
            Rj = R3N[:, j]
            rij = float(np.linalg.norm(Ri - Rj))
            cutoff = scale * (radii[Z[i]] + radii[Z[j]])
            if rij > 0.0 and rij <= cutoff:
                a = nbonds[i]; b = nbonds[j]
                if a < max_nb:
                    ibonds[a, i] = j; nbonds[i] += 1
                if b < max_nb:
                    ibonds[b, j] = i; nbonds[j] += 1
    return nbonds, ibonds

# ====== C≡C 修正（PM6 系列可用）======
def C_triple_bond_C_pm6(nat, R3N, nbonds, ibonds, numat):
    """
    等价 Fortran C_triple_bond_C() 的 PM6 用法。
    nat: (N,), R3N: (3,N)
    """
    rmin = 1.21
    rmax = 1.33
    param1 = -5.0
    param2 = 25.0

    s = 0.0
    for i in range(numat):
        if nat[i] != 6:  # 仅 C
            continue
        nbi = int(nbonds[i])
        for kk in range(nbi):
            j = int(ibonds[kk, i])
            if j < 0 or j > i:
                continue
            if nat[j] != 6:
                continue
            dvec = R3N[:, i] - R3N[:, j]
            rij2 = float(np.dot(dvec, dvec))
            if rij2 < rmin**2:
                s += 1.0
            elif rmin**2 <= rij2 < rmax**2:
                rab = (np.sqrt(rij2) - rmin) / (rmax - rmin)
                s += (
                    1.0
                    - 10.0 * rab**3
                    + 15.0 * rab**4
                    - 6.0  * rab**5
                    + (param1 + rab * param2)
                      * (rab**3 - 3.0 * rab**4 + 3.0 * rab**5 - rab**6)
                )
    return s * 12.0  # 经验系数

def nsp2_atom_correction(R, n, i, j, k):
    """
    Python 版 nsp2_atom_correction，对齐 Fortran：
      tot = 2π - (θ_ij + θ_ik + θ_jk)
      return -0.5 * exp(-10 * tot)
    说明：
      - R: (3, N) 或 (N, 3) 的坐标数组
      - n 为中心 N 原子的下标；i, j, k 为其三邻居的下标（均为 0-based）
    """
    # R = _as_3xN(R)
    rn = R[:, n]
    ri = R[:, i]
    rj = R[:, j]
    rk = R[:, k]
    def _dist(p, q):
        return np.linalg.norm(p - q)

    def _angle_from_sides(x, y, opp):
        """
        由三角形的两邻边 x、y 与其对边 opp 通过余弦定理求角：
        cos θ = (x^2 + y^2 - opp^2) / (2xy)
        数值上做 clip，避免 arccos 域错误。
        """
        den = 2.0 * x * y
        if den == 0.0:
            return 0.0
        cosv = (x*x + y*y - opp*opp) / den
        cosv = np.clip(cosv, -1.0, 1.0)
        return np.arccos(cosv)
    # 中心到三邻居的距离（a, b, c）
    a = _dist(rn, ri)  # n-i
    b = _dist(rn, rj)  # n-j
    c = _dist(rn, rk)  # n-k

    # 三邻居之间的边（ab, ac, bc）
    ab = _dist(rj, ri)  # j-i
    ac = _dist(rk, ri)  # k-i
    bc = _dist(rj, rk)  # j-k

    # 三个围绕中心原子的夹角（对应 Fortran 中 cosa, cosb, cosc）
    theta_a = _angle_from_sides(b, c, bc)  # 角(j-n-k)
    theta_b = _angle_from_sides(a, c, ac)  # 角(i-n-k)
    theta_c = _angle_from_sides(b, a, ab)  # 角(j-n-i)

    tot = 2.0 * np.pi - (theta_a + theta_b + theta_c)
    return float(-0.5 * np.exp(-10.0 * tot))

# ====== N(sp2) 三配位修正（依赖你的原子级校正函数）======
def nsp2_correction_pm6(nat, R3N, nbonds, ibonds, numat, nsp2_atom_correction):
    """
    仅在 N 原子三配位且 H 个数 < 2 时，调用 nsp2_atom_correction(coord, i, j1, j2, j3)
    要求 nsp2_atom_correction 接受 R3N (3,N) 和四个下标（0-based）。
    """
    corr = 0.0
    for i in range(numat):
        if nat[i] == 7 and int(nbonds[i]) == 3:
            j1, j2, j3 = (int(ibonds[0, i]), int(ibonds[1, i]), int(ibonds[2, i]))
            if j1 < 0 or j2 < 0 or j3 < 0:
                continue
            hcount = int(nat[j1] == 1) + int(nat[j2] == 1) + int(nat[j3] == 1)
            if hcount < 2:
                corr += float(nsp2_atom_correction(R3N, i, j1, j2, j3))
    return corr

# ====== 识别 O=C–N–H，生成 NHCO 二面角列表 + PM6 的 htype ======
def setup_nhco_simple(nat, R3N, numat, keywrd=""):
    """
    返回：nhco(4,n), nnhco, htype, ii_flag
    PM6 的 htype = 2.5（Fortran 固定值）
    搜索阈值：O–C ≤ 1.3 Å，N–C ≤ 1.6 Å，N–H ≤ 1.3 Å，另找 N 的一个邻接 m（≤1.7 Å）
    """
    htype = 2.5000
    ii_flag = 1 if ("NOMM" in (keywrd or "")) else 0

    lst = []
    for j in range(numat):
        if nat[j] != 6:  # C
            continue
        for i in range(numat):
            if nat[i] != 8:  # O
                continue
            if distance_euclid(R3N, i, j) > 1.3:
                continue
            for k in range(numat):
                if nat[k] != 7:  # N
                    continue
                if distance_euclid(R3N, k, j) > 1.6:
                    continue
                for l in range(numat):
                    if nat[l] != 1:  # H
                        continue
                    if distance_euclid(R3N, k, l) > 1.3:
                        continue
                    # 找 N 上另一个配位 m
                    used = False
                    for m in range(numat):
                        if m in (k, l, j):
                            continue
                        if distance_euclid(R3N, m, k) > 1.7:
                            continue
                        if not used:
                            if ii_flag == 0:
                                lst.append([i, j, k, m])
                                lst.append([i, j, k, l])
                            used = True
                            break
    if not lst:
        nhco = np.zeros((4, 0), dtype=int)
        nnhco = 0
    else:
        nhco = np.array(lst, dtype=int).T
        nnhco = nhco.shape[1]
    return nhco, nnhco, htype, ii_flag

# ====== 二面角（无 PBC，3×N）======
def dihed_simple(R3N, i, j, k, l):
    """
    等价 Fortran dihed 在 id==0 的情形（分子，无 PBC）。
    返回角度范围 [0, 2π)。
    """
    r_ik = R3N[:, i] - R3N[:, k]
    r_jk = R3N[:, j] - R3N[:, k]
    r_lk = R3N[:, l] - R3N[:, k]

    xj1, yj1, zj1 = r_jk
    xi1, yi1, zi1 = r_ik
    xl1, yl1, zl1 = r_lk

    dist = float(np.linalg.norm(r_jk))
    cosa = (zj1 / dist) if dist > 0 else 1.0
    cosa = max(-1.0, min(1.0, cosa))
    ddd = 1.0 - cosa**2

    if ddd <= 0.0:
        xi2, yi2 = xi1, yi1
        xl2, yl2 = xl1, yl1
        costh, sinth = cosa, 0.0
    else:
        yxdist = dist * np.sqrt(ddd)
        if yxdist <= 1.0e-6:
            xi2, yi2 = xi1, yi1
            xl2, yl2 = xl1, yl1
            costh, sinth = cosa, 0.0
        else:
            cosph = yj1 / yxdist
            sinph = xj1 / yxdist
            # 绕 z 轴旋转
            xi2 = xi1 * cosph - yi1 * sinph
            xl2 = xl1 * cosph - yl1 * sinph
            yi2 = xi1 * sinph + yi1 * cosph
            yj2 = xj1 * sinph + yj1 * cosph
            yl2 = xl1 * sinph + yl1 * cosph
            # 绕 x 轴旋转
            costh = cosa
            sinth = yj2 / dist if dist > 0 else 0.0

    yi3 = yi2 * costh - zi1 * sinth
    yl3 = yl2 * costh - zl1 * sinth
    ang = dang(xl2, yl3, xi2, yi3)
    if ang < 0.0:
        ang = 2.0*np.pi + ang
    if ang >= 2.0*np.pi:
        ang = 0.0
    return float(ang)

# ====== 顶层：PM6 分子体系（无 PBC）总能修正 atheat ======
def compute_atheat_pm6_mol(
    Z, R3N,
    *,
    eheat_lookup, eisol_lookup, fpc_9=23.060547830619029, scale=1.2, max_nb=12,
    keywrd=""
):
    """
    输入：
      Z: (N,) 原子序
      R3N: (3,N) 坐标（Å）
      eheat_lookup/eisol_lookup: 可函数或数组；按 Z 取值
      fpc_9: 标量
      covalent_radii: (maxZ+1,) 共价半径表（Å）
      keywrd: 只影响 NHCO 的 NOMM 行为
    返回：dict( atheat, eat, ccc_corr, nsp2_corr, sum_dihed )
    """
    Z = np.asarray(Z, dtype=int)
    numat = Z.size

    eH  = _as_lookup(eheat_lookup)
    Ei  = _as_lookup(eisol_lookup)

    # 主项：Σ eheat(Z)
    atheat = float(np.sum([eH(int(z)-1) for z in Z]))

    # 减去 eat * fpc_9
    eat = float(np.sum([Ei(int(z)-1) for z in Z]))
    atheat -= eat * fpc_9

    # 邻接表
    nbonds, ibonds = build_bonds_simple(Z, R3N, radii=covalent_radii, scale=scale, max_nb=max_nb)

    # C≡C 修正
    ccc_corr = C_triple_bond_C_pm6(Z, R3N, nbonds, ibonds, numat)
    atheat += ccc_corr

    # N(sp2) 修正
    nsp2_corr = nsp2_correction_pm6(Z, R3N, nbonds, ibonds, numat, nsp2_atom_correction)
    atheat += nsp2_corr

    # NHCO 二面角项
    nhco, nnhco, htype, _ = setup_nhco_simple(Z, R3N, numat, keywrd=keywrd)
    sum_dihed = 0.0
    for col in range(nnhco):
        i, j, k, l = (int(nhco[r, col]) for r in range(4))
        ang = dihed_simple(R3N, i, j, k, l)  # [0,2π)
        sum_dihed += htype * (np.sin(ang) ** 2)
    atheat += sum_dihed

    return dict(
        atheat=float(atheat),
        eat=float(eat),
        ccc_corr=float(ccc_corr),
        nsp2_corr=float(nsp2_corr),
        sum_dihed=float(sum_dihed),
    )