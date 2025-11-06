import numpy as np
import sys
from typing import Optional, TextIO, Callable, Dict
from pyscf.data.elements import ELEMENTS

# —— AO label——
_AO_TAGS = (" S", "PX", "PY", "PZ", "X2", "XZ", "Z2", "YZ", "XY")

def _z_to_symbol(z: int) -> str:
    z = int(z)
    if 0 <= z < len(ELEMENTS):
        return ELEMENTS[z]
    return f"Z{z}"
def _pack_index(i: int, j: int) -> int:
    if j > i:
        i, j = j, i
    return i * (i + 1) // 2 + j
def _add_packed_block(h, ia, ib, eblk, scale=1.0):
    L = ib - ia + 1
    if L <= 0:
        return
    off = 0
    for p in range(ia, ib + 1):
        n = p - ia + 1
        row_vals = eblk[off:off + n]
        base = _pack_index(p, ia)
        h[base: base + n] += scale * row_vals
        off += n

def _packed_diag_index(i: int) -> int:
    return (i+1)*(i+2)//2 - 1

def packed_to_full(packed, norb):
    full = np.zeros((norb, norb), dtype=packed.dtype)
    i, j = np.tril_indices(norb)
    full[i, j] = packed
    full[j, i] = packed
    return full

def full_to_packed(full):
    norb = full.shape[0]
    # out = np.empty(norb*(norb+1)//2, dtype=full.dtype)
    i, j = np.tril_indices(norb)
    out = full[i, j]
    return out

def vecprt_h(
    pm6mol,
    stream: Optional[TextIO] = None,
) -> None:
    out = stream if stream is not None else sys.stdout

    a = np.asarray(pm6mol.h, dtype=float)
    n = int(abs(pm6mol.norbs))
    assert a.size == n*(n+1)//2, f"inconsistent length: got {a.size}, need {n*(n+1)//2}"

    numat = int(pm6mol.numat)
    nfirst = np.asarray(pm6mol.nfirst, dtype=int) if numat else None
    nlast  = np.asarray(pm6mol.nlast,  dtype=int) if numat else None
    nat    = np.asarray(pm6mol.Z,    dtype=int) if numat else None

    sumax = 1.0
    if n > 0:
        # max |diag|
        dmax = 0.0
        for i in range(n):
            dmax = max(dmax, abs(a[_packed_diag_index(i)]))
        sumax = max(1.0, dmax)
    i10 = int(np.floor(np.log10(sumax))) if sumax > 0 else 0
    if i10 in (1, 2):
        i10 = 0
    fact = 10.0**(-i10)

    if abs(fact - 1.0) > 1e-3:
        print(f"Diagonal Terms should be Multiplied by {1.0/fact:12.1f}", file=out)

    a_print = a.copy()
    if abs(fact - 1.0) > 1e-3:
        for i in range(n):
            ii = _packed_diag_index(i)
            a_print[ii] *= fact

    itext = np.full(n, "  ", dtype=object)
    jtext = np.full(n, "  ", dtype=object)
    natom = np.arange(1, n+1, dtype=int)

    if numat != 0 and numat == n:
        for i in range(numat):
            jtext[i] = _z_to_symbol(int(nat[i])) if nat is not None else "  "
            natom[i] = i + 1

    elif numat != 0 and nlast is not None and nfirst is not None and len(nlast) >= numat:
        last_raw = int(nlast[-1])
        if last_raw == n:       
            base = 1
        elif last_raw == n-1:    
            base = 0
        else:
            base = 1 if int(np.min(nfirst)) >= 1 else 0

        for i in range(numat):
            jlo0 = int(nfirst[i]) - base       
            jhi0 = int(nlast[i])  - base       
            if jlo0 < 0 or jhi0 >= n or jhi0 < jlo0:
                continue
            Z = int(nat[i]) if nat is not None else 0
            L = jhi0 - jlo0 + 1                
            itext[jlo0:jhi0+1] = _AO_TAGS[:L]   # —— MOPAC d sequence：X2, XZ, Z2, YZ, XY —— #
            jtext[jlo0:jhi0+1] = _z_to_symbol(Z)
            natom[jlo0:jhi0+1] = i + 1
    else:
        pass

    na = 0  # 0-based 
    dashed = "------"
    while na < n:
        width = min(n - na, 6)
        m = na + width - 1
        header = []
        for col in range(na, m+1):
            header.append(f"{itext[col]:>2} {jtext[col]:>2} {natom[col]:>4}")
        print("\n", file=out)
        print(" " * 12 + "  ".join(header), file=out)
        print(" " + " ".join([dashed]*(2*width+1)), file=out)

        for i in range(na, n):
            j_end = min(i, m)
            if j_end < na:
                continue
            row_vals = []
            for j in range(na, j_end+1):
                idx = _pack_index(i, j)  # i>=j 
                row_vals.append(f"{a_print[idx]:11.6f}")

            row_head = f"{itext[i]:>2} {jtext[i]:>2} {natom[i]:>5}"
            print(f" {row_head}  " + " ".join(row_vals), file=out)

        na = m + 1
    return

def printp(i, para, value, txt, iw=None):
    if iw is None:
        iw = sys.stdout

    v = 0.0 if (value is None or np.isnan(value)) else float(value)

    if abs(v) > 1e-5:
        # Fortran '(I4,A7,2X,F13.8,2X,A)'
        line = f"{int(i):<5d}{str(para):<7}  {v:13.8f}  {txt}"
        iw.write(line + "\n")

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

    nZ = int(len(uss))

    used = np.zeros(nZ, dtype=bool)
    if numat > 0:
        idx = np.asarray(nat[:numat], dtype=int)
        used_idx = idx[(idx >= 0) & (idx < nZ)]
        used[used_idx-1] = True

    iw.write("\n")
    iw.write("PARAMETER VALUES USED IN THE CALCULATION\n\n")
    iw.write(" NI    TYPE        VALUE     UNIT\n\n")
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

        # VdW 
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

        jmax = min(100, nZ)
        row = alpb[i, :jmax]
        nan_mask = np.isnan(row)
        if np.any(nan_mask):
            row[nan_mask] = 0.0
            alpb[i, :jmax] = row 

        for j in range(jmax):
            if abs(alpb[i, j]) > 1e-5 and used[j]:
                # "(I4,A6,i2,F13.8,2X,A)"
                iw.write(f"{(i+1):<5d}{'ALPB_':<6}{(j+1):2d}{alpb[i,j]:13.8f}  ALPB factor\n")
                iw.write(f"{(i+1):<5d}{'XFAC_':<6}{(j+1):2d}{xfac[i,j]:13.8f}  XFAC factor\n")

def print_title(iw, title: str, leading_blank_lines=2, trailing_blank_lines=1):
    if isinstance(iw, (str, bytes)):
        iw = open(iw, "a", encoding="utf-8")
    pre = "\n" * leading_blank_lines + " " * 10
    post = "\n" * trailing_blank_lines
    print(f"{pre}{title}{post}", file=iw)

def _f8_4_block_str(arr) -> str:
    out_lines = []
    n = len(arr)
    for i in range(0, n, 10):
        chunk = arr[i:i+10]
        line = "".join(f"{float(x):25.16f}" for x in chunk)
        out_lines.append(line)
    return "\n".join(out_lines)

def vecprt_w(iw, vec: np.ndarray, title: str):
    print_title(iw, title, leading_blank_lines=2, trailing_blank_lines=1)
    if vec.size == 0:
        return
    print(_f8_4_block_str(vec), file=iw)