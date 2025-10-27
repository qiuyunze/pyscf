import numpy as np
import sys
from typing import Optional, TextIO, Callable, Dict
from pyscf.data.elements import ELEMENTS

# —— AO 标签顺序（与 MOPAC 一致）——
_AO_TAGS = (" S", "PX", "PY", "PZ", "X2", "XZ", "Z2", "YZ", "XY")

def _z_to_symbol(z: int) -> str:
    z = int(z)
    if 0 <= z < len(ELEMENTS):
        return ELEMENTS[z]
    return f"Z{z}"
def _pack_index(i: int, j: int) -> int:
            """下三角打包索引（0-basis）。要求 j<=i。"""
            if j > i:
                i, j = j, i
            return i * (i + 1) // 2 + j
def _add_packed_block(h, ia, ib, eblk, scale=1.0):
    """
    将打包的下三角块 eblk（长度 (L*(L+1))//2, L=ib-ia+1）加到 h 的大打包矩阵中。
    把 eblk 的第 r 行（局部）加到全局行 p=ia+r 的列区间 [ia, ia+r]。
    """
    L = ib - ia + 1
    if L <= 0:
        return
    off = 0
    for p in range(ia, ib + 1):
        n = p - ia + 1
        row_vals = eblk[off:off + n]
        # 写到全局 h 的 (p, ia..p)
        base = _pack_index(p, ia)
        h[base: base + n] += scale * row_vals
        off += n

def _packed_diag_index(i: int) -> int:
    """0-based 第 i 行的对角元素在 packed 向量中的索引。"""
    return (i+1)*(i+2)//2 - 1

def packed_to_full(packed, norb):
    """下三角packed -> 全矩阵（对称）"""
    full = np.zeros((norb, norb), dtype=packed.dtype)
    idx = 0
    for i in range(norb):
        full[i, :i+1] = packed[idx:idx+i+1]
        full[:i+1, i] = packed[idx:idx+i+1]
        idx += i+1
    return full

def full_to_packed(full):
    norb = full.shape[0]
    out = np.empty(norb*(norb+1)//2, dtype=full.dtype)
    idx = 0
    for i in range(norb):
        out[idx:idx+i+1] = full[i, :i+1]
        idx += i+1
    return out

def vecprt_h(
    pm6mol,
    stream: Optional[TextIO] = None,
) -> None:
    """
    打印 packed 下三角向量 a（长度 = numm*(numm+1)/2）。
    自动添加原子/轨道抬头，行为尽量贴近 Fortran vecprt。
    """
    out = stream if stream is not None else sys.stdout

    a = np.asarray(pm6mol.h, dtype=float)
    n = int(abs(pm6mol.norbs))
    assert a.size == n*(n+1)//2, f"packed 长度不匹配: got {a.size}, need {n*(n+1)//2}"

    numat = int(pm6mol.numat)
    nfirst = np.asarray(pm6mol.nfirst, dtype=int) if numat else None
    nlast  = np.asarray(pm6mol.nlast,  dtype=int) if numat else None
    nat    = np.asarray(pm6mol.Z,    dtype=int) if numat else None

    # —— 只为打印而缩放对角线（和 Fortran 相同）——
    # sumax 取对角线最大绝对值，但不小于 1.0
    sumax = 1.0
    if n > 0:
        # 最大 |diag|
        dmax = 0.0
        for i in range(n):
            dmax = max(dmax, abs(a[_packed_diag_index(i)]))
        sumax = max(1.0, dmax)
    i10 = int(np.floor(np.log10(sumax))) if sumax > 0 else 0
    if i10 in (1, 2):
        i10 = 0
    fact = 10.0**(-i10)

    # 打印时提示缩放因子
    if abs(fact - 1.0) > 1e-3:
        print(f"Diagonal Terms should be Multiplied by {1.0/fact:12.1f}", file=out)

    # 生成仅用于打印的对角缩放拷贝
    a_print = a.copy()
    if abs(fact - 1.0) > 1e-3:
        for i in range(n):
            ii = _packed_diag_index(i)
            a_print[ii] *= fact

    # —— 抬头标签（按原子 或 按 AO）——
    # itext: AO 标签（S/PX/...）；jtext: 元素符号；natom: 原子序号（1-based）
   # —— 抬头标签（按原子 或 按 AO）——
    itext = np.full(n, "  ", dtype=object)
    jtext = np.full(n, "  ", dtype=object)
    natom = np.arange(1, n+1, dtype=int)

    if numat != 0 and numat == n:
        # 情形：每行一个原子（少见；与 Fortran 相同保留）
        for i in range(numat):
            jtext[i] = _z_to_symbol(int(nat[i])) if nat is not None else "  "
            natom[i] = i + 1

    elif numat != 0 and nlast is not None and nfirst is not None and len(nlast) >= numat:
        # —— 关键修正：自动判定 nfirst/nlast 是 1-based 还是 0-based —— #
        last_raw = int(nlast[-1])
        if last_raw == n:          # Fortran 情形：1-based，nlast(numat) == numm
            base = 1
        elif last_raw == n-1:      # 0-based
            base = 0
        else:
            # 再保险：如果最小的 nfirst>=1 也大概率是 1-based
            base = 1 if int(np.min(nfirst)) >= 1 else 0

        # 逐原子填 AO 标签（S, PX, PY, PZ, X2, XZ, Z2, YZ, XY）
        for i in range(numat):
            jlo0 = int(nfirst[i]) - base        # 统一转成 0-based
            jhi0 = int(nlast[i])  - base        # 统一转成 0-based
            if jlo0 < 0 or jhi0 >= n or jhi0 < jlo0:
                # 防御：越界就跳过该原子（不致崩）
                continue
            Z = int(nat[i]) if nat is not None else 0
            L = jhi0 - jlo0 + 1                 # 该原子的 AO 个数（1/4/9）
            itext[jlo0:jhi0+1] = _AO_TAGS[:L]   # —— MOPAC 的 d 顺序：X2, XZ, Z2, YZ, XY —— #
            jtext[jlo0:jhi0+1] = _z_to_symbol(Z)
            natom[jlo0:jhi0+1] = i + 1
    else:
        # 无法判断：留空
        pass

    # —— 分块打印（每块最多 6 列，从 1 开始）——
    # Fortran 的 fmt 里宽度取决于列数，这里用 Python 简洁输出。
    na = 0  # 0-based 起始列
    dashed = "------"
    while na < n:
        # 当前面板列范围 [na, m]
        width = min(n - na, 6)
        m = na + width - 1
        # 抬头
        header = []
        for col in range(na, m+1):
            header.append(f"{itext[col]:>2} {jtext[col]:>2} {natom[col]:>4}")
        # 行 1：列抬头
        print("\n", file=out)
        print(" " * 12 + "  ".join(header), file=out)
        # 行 2：横线
        print(" " + " ".join([dashed]*(2*width+1)), file=out)

        # 主体：逐行打印
        for i in range(na, n):
            # 该行在本面板中可见的列：从 na 到 min(i, m)（因为是下三角）
            j_end = min(i, m)
            if j_end < na:
                continue
            row_vals = []
            for j in range(na, j_end+1):
                idx = _pack_index(i, j)  # i>=j 保证
                row_vals.append(f"{a_print[idx]:11.6f}")

            # 行标签：itext/jtext/natom（与 Fortran 行首一致）
            row_head = f"{itext[i]:>2} {jtext[i]:>2} {natom[i]:>5}"
            print(f" {row_head}  " + " ".join(row_vals), file=out)

        # 下一块
        na = m + 1

    # 打印完毕后无需恢复 a（我们没改 a 本体）
    return

def printp(i, para, value, txt, iw=None):
    """
    Python/NumPy 版本的 printp。
    - 与 Fortran 一致：NaN 视为 0；|value|<=1e-5 不打印。
    - i、para、txt 原样使用；para 建议给定长度<=7。
    """
    if iw is None:
        iw = sys.stdout

    # 处理 NaN
    v = 0.0 if (value is None or np.isnan(value)) else float(value)

    if abs(v) > 1e-5:
        # Fortran '(I4,A7,2X,F13.8,2X,A)'
        # I4: 右对齐4位；A7: 左对齐7位；F13.8 固定宽度小数；2X 两个空格
        line = f"{int(i):<5d}{str(para):<7}  {v:13.8f}  {txt}"
        iw.write(line + "\n")
### 经过初始化计算，有很多参数是二次计算出来的，存在env中

def prtpar(
    pm6mol, env, iw=None
):
    """
    mol: PM6MOLE class
    env: PM6env class
    """
    nat = pm6mol.Z
    numat = pm6mol.numat
    uss = env.uss6
    upp = env.upp6
    udd = env.udd6
    zs = env.zs6
    zp = env.zp6
    zd = env.zd6
    betas = env.betas6
    betap = env.betap6
    betad = env.betad6
    alp = env.alp6
    gss = env.gss6
    gpp = env.gpp6
    gp2 = env.gp26
    hsp = env.hsp6
    gsp = env.gsp6
    zsn = env.zsn6
    zpn = env.zpn6
    zdn = env.zdn6
    g2sd = env.g2sd6
    ### 以下是二次计算出来的参数，首先是这些要与mopac对齐
    f0dd = env.f0dd
    f2dd = env.f2dd
    f4dd = env.f4dd
    f0sd = env.f0sd
    f0pd = env.f0pd
    f2pd = env.f2pd
    g1pd = env.g1pd
    g3pd = env.g3pd
    ddp = env.ddp
    po = env.po
    tore = env.tore
    eisol = env.eisol
    eheat = env.eheat  # remain to be loaded
    guess1 = env.gues61
    guess2 = env.gues62
    guess3 = env.gues63
    alpb = env.alpb
    xfac = env.xfac
    
    if iw is None:
        iw = sys.stdout

    # 推断元素种类个数 nZ
    nZ = int(len(uss))

    # used 标记：哪些元素类型在 nat[:numat] 中被使用
    used = np.zeros(nZ, dtype=bool)
    if numat > 0:
        idx = np.asarray(nat[:numat], dtype=int)
        # 只标记有效范围
        used_idx = idx[(idx >= 0) & (idx < nZ)]
        used[used_idx-1] = True

    iw.write("\n")
    iw.write("PARAMETER VALUES USED IN THE CALCULATION\n\n")
    iw.write(" NI    TYPE        VALUE     UNIT\n\n")

    # Fortran 循环 1..100，这里取 0..min(100,nZ)-1
    upper = min(100, nZ)

    for i in range(upper):
        if not used[i]:
            continue
        Z = i + 1
        iw.write("\n")
        printp(Z, 'USS',   uss[i],   'EV        ONE-CENTER ENERGY FOR S', iw)
        printp(Z, 'UPP',   upp[i],   'EV        ONE-CENTER ENERGY FOR P', iw)
        printp(Z, 'UDD',   udd[i],   'EV        ONE-CENTER ENERGY FOR D', iw)
        printp(Z, 'ZS',    zs[i],    'AU        ORBITAL EXPONENT  FOR S', iw)
        printp(Z, 'ZP',    zp[i],    'AU        ORBITAL EXPONENT  FOR P', iw)
        printp(Z, 'ZD',    zd[i],    'AU        ORBITAL EXPONENT  FOR D', iw)
        printp(Z, 'BETAS', betas[i], 'EV        BETA PARAMETER    FOR S', iw)
        printp(Z, 'BETAP', betap[i], 'EV        BETA PARAMETER    FOR P', iw)
        printp(Z, 'BETAD', betad[i], 'EV        BETA PARAMETER    FOR D', iw)
        printp(Z, 'ALP',   alp[i],   '(1/A)     ALPHA PARAMETER   FOR CORE', iw)

        printp(Z, 'GSS', gss[i], 'EV        ONE-CENTER INTEGRAL (SS,SS)', iw)
        printp(Z, 'GPP', gpp[i], 'EV        ONE-CENTER INTEGRAL (PP,PP)', iw)
        printp(Z, 'GSP', gsp[i], 'EV        ONE-CENTER INTEGRAL (SS,PP)', iw)
        printp(Z, 'GP2', gp2[i], 'EV        ONE-CENTER INTEGRAL (PP*,PP*)', iw)
        printp(Z, 'HSP', hsp[i], 'EV        ONE-CENTER INTEGRAL (SP,SP)', iw)

        printp(Z, 'ZSN', zsn[i], 'AU        INTERNAL EXPONENT FOR S - (IJ,KL)', iw)
        printp(Z, 'ZPN', zpn[i], 'AU        INTERNAL EXPONENT FOR P - (IJ,KL)', iw)
        printp(Z, 'ZDN', zdn[i], 'AU        INTERNAL EXPONENT FOR D - (IJ,KL)', iw)

        printp(Z, 'F0DD', f0dd[i], 'EV        SLATER-CONDON PARAMETER F0DD', iw)
        printp(Z, 'F2DD', f2dd[i], 'EV        SLATER-CONDON PARAMETER F2DD', iw)
        printp(Z, 'F4DD', f4dd[i], 'EV        SLATER-CONDON PARAMETER F4DD', iw)

        printp(Z, 'F0SD', f0sd[i], 'EV        SLATER-CONDON PARAMETER F0SD', iw)
        printp(Z, 'G2SD', g2sd[i], 'EV        SLATER-CONDON PARAMETER G2SD', iw)

        printp(Z, 'F0PD', f0pd[i], 'EV        SLATER-CONDON PARAMETER F0PD', iw)
        printp(Z, 'F2PD', f2pd[i], 'EV        SLATER-CONDON PARAMETER F2PD', iw)
        printp(Z, 'G1PD', g1pd[i], 'EV        SLATER-CONDON PARAMETER G1PD', iw)
        printp(Z, 'G3PD', g3pd[i], 'EV        SLATER-CONDON PARAMETER G3PD', iw)

        # ddp:aZ ddp(2..6,i) → Python ddp[1..5, i]
        printp(Z, 'DD2',  ddp[1, i], 'BOHR      CHARGE SEPARATION, SP, L=1', iw)
        printp(Z, 'DD3',  ddp[2, i], 'BOHR      CHARGE SEPARATION, PP, L=2', iw)
        printp(Z, '=',    ddp[2, i]/np.sqrt(2.0),
               'BOHR      USING ORIGINAL MNDO PAPER FORMULA', iw)
        printp(Z, 'DD4',  ddp[3, i], 'BOHR      CHARGE SEPARATION, SD, L=2', iw)
        printp(Z, 'DD5',  ddp[4, i], 'BOHR      CHARGE SEPARATION, PD, L=1', iw)
        printp(Z, 'DD6',  ddp[5, i], 'BOHR      CHARGE SEPARATION, DD, L=2', iw)

        # po: n po(1..9,i) → Python po[0..8,i]
        printp(Z, 'PO1', po[0, i], 'BOHR      KLOPMAN-OHNO TERM, SS, L=0', iw)
        printp(Z, 'PO2', po[1, i], 'BOHR      KLOPMAN-OHNO TERM, SP, L=1', iw)
        printp(Z, 'PO3', po[2, i], 'BOHR      KLOPMAN-OHNO TERM, PP, L=2', iw)
        printp(Z, 'PO4', po[3, i], 'BOHR      KLOPMAN-OHNO TERM, SD, L=2', iw)
        printp(Z, 'PO5', po[4, i], 'BOHR      KLOPMAN-OHNO TERM, PD, L=1', iw)
        printp(Z, 'PO6', po[5, i], 'BOHR      KLOPMAN-OHNO TERM, DD, L=2', iw)
        printp(Z, 'PO7', po[6, i], 'BOHR      KLOPMAN-OHNO TERM, PP, L=0', iw)
        printp(Z, 'PO8', po[7, i], 'BOHR      KLOPMAN-OHNO TERM, DD, L=0', iw)
        printp(Z, 'PO9', po[8, i], 'BOHR      KLOPMAN-OHNO TERM, CORE', iw)
        printp(Z, 'CORE',  tore[i],  'E         CORE CHARGE', iw)
        printp(Z, 'EHEAT', eheat[i], 'KCAL/MOL  HEAT OF FORMATION OF THE ATOM (EXP)', iw)
        printp(Z, 'EISOL', eisol[i], 'EV        TOTAL ENERGY OF THE ATOM (CALC)', iw)

        # VdW （最多四项）
        printp(Z, 'FN11', guess1[i, 0], 'CORE-CORE VDW MULTIPLIER 1', iw)
        printp(Z, 'FN21', guess2[i, 0], 'CORE-CORE VDW EXPONENT 1',    iw)
        printp(Z, 'FN31', guess3[i, 0], 'CORE-CORE VDW POSITION 1',    iw)

        printp(Z, 'FN12', guess1[i, 1], 'CORE-CORE VDW MULTIPLIER 2', iw)
        printp(Z, 'FN22', guess2[i, 1], 'CORE-CORE VDW EXPONENT 2',    iw)
        printp(Z, 'FN32', guess3[i, 1], 'CORE-CORE VDW POSITION 2',    iw)

        printp(Z, 'FN13', guess1[i, 2], 'CORE-CORE VDW MULTIPLIER 3', iw)
        printp(Z, 'FN23', guess2[i, 2], 'CORE-CORE VDW EXPONENT 3',    iw)
        printp(Z, 'FN33', guess3[i, 2], 'CORE-CORE VDW POSITION 3',    iw)

        printp(Z, 'FN14', guess1[i, 3], 'CORE-CORE VDW MULTIPLIER 4', iw)
        printp(Z, 'FN24', guess2[i, 3], 'CORE-CORE VDW EXPONENT 4',    iw)
        printp(Z, 'FN34', guess3[i, 3], 'CORE-CORE VDW POSITION 4',    iw)

        # alpb/xfac（NaN→0，且仅在 |alpb|>1e-5 且元素 j 被使用时打印）
        jmax = min(100, nZ)
        # 将 alpb 的 NaN 原地替换为 0（与 Fortran 同步语义）
        # 仅对第 i 行前 jmax 列做一次清理
        row = alpb[i, :jmax]
        nan_mask = np.isnan(row)
        if np.any(nan_mask):
            row[nan_mask] = 0.0
            alpb[i, :jmax] = row  # 写回

        for j in range(jmax):
            if abs(alpb[i, j]) > 1e-5 and used[j]:
                # "(I4,A6,i2,F13.8,2X,A)"
                iw.write(f"{(i+1):<5d}{'ALPB_':<6}{(j+1):2d}{alpb[i,j]:13.8f}  ALPB factor\n")
                iw.write(f"{(i+1):<5d}{'XFAC_':<6}{(j+1):2d}{xfac[i,j]:13.8f}  XFAC factor\n")

def print_title(iw, title: str, leading_blank_lines=2, trailing_blank_lines=1):
    if isinstance(iw, (str, bytes)):
        # 允许传路径；否则当作 file-like
        iw = open(iw, "a", encoding="utf-8")
    # 模仿 Fortran 的 (2/10X,'TITLE' ) 10 个空格缩进
    pre = "\n" * leading_blank_lines + " " * 10
    post = "\n" * trailing_blank_lines
    print(f"{pre}{title}{post}", file=iw)

def _f8_4_block_str(arr) -> str:
    """按 Fortran format(10f8.4) 生成字符串。"""
    out_lines = []
    n = len(arr)
    for i in range(0, n, 10):
        chunk = arr[i:i+10]
        line = "".join(f"{float(x):25.16f}" for x in chunk)
        out_lines.append(line)
    return "\n".join(out_lines)

def vecprt_w(iw, vec: np.ndarray, title: str):
    """打印向量（按 10f8.4）"""
    print_title(iw, title, leading_blank_lines=2, trailing_blank_lines=1)
    if vec.size == 0:
        return
    print(_f8_4_block_str(vec), file=iw)