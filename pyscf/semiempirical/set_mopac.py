import numpy as np
from math import exp

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

def aintgs_np(x: float, k: int) -> np.ndarray:
    """
    Fortran:
      c = exp(-x)
      a(1) = c/x
      do i = 1, k
        a(i+1) = (a(i)*i + c)/x
      end do
    Python (0-based):
      a[0] = c/x
      for i in 1..k: a[i] = (a[i-1]*i + c)/x
    """
    if k < 0:
        return np.zeros(0, dtype=float)
    a = np.empty(k + 1, dtype=float)
    c = exp(-x)
    a[0] = c / x
    for i in range(1, k + 1):
        a[i] = (a[i - 1] * i + c) / x
    return a


def bintgs_np(x: float, k: int) -> np.ndarray:
    """
    return  0-based array (k+11), satisfying b(i+1) ↔ b[i]。
    """
    if k < 0:
        return np.zeros(0, dtype=float)
    b = np.empty(k + 1, dtype=float)

    io = 0
    absx = abs(x)

    use_series = False
    if absx > 3.0:
        pass
    elif absx > 2.0:
        if k <= 10:
            pass  
        else:
            last = 15; use_series = True
    elif absx > 1.0:
        if k <= 7:
            pass
        else:
            last = 12; use_series = True
    elif absx > 0.5:
        if k <= 5:
            pass
        else:
            last = 7; use_series = True
    else:
        if absx <= 1e-6:
            for i in range(io, k + 1):
                b[i] = (2 * ((i + 1) % 2)) / (i + 1.0)
            return b
        last = 6; use_series = True

    if not use_series:
        ex = exp(x)
        exm = 1.0 / ex
        b[0] = (ex - exm) / x
        for i in range(1, k + 1):
            b[i] = (i * b[i - 1] + ((-1) ** i) * ex - exm) / x
        return b

    for i in range(io, k + 1):
        y = 0.0
        for m in range(io, last + 1):
            xf = 1.0 if m == 0 else float(fact[m])
            y += ((-x) ** m) * (2 * ((m + i + 1) % 2)) / (xf * (m + i + 1))
        b[i] = y
    return b


def set_np(s1: float, s2: float, na: int, nb: int, rab: float, ii: int):
    """
    Fortran SET :
      - assigning isp/ips and sa/sb 
      - calculate j、alpha、beta、jcall
      - call AINTGS、BINTGS to obtain a(1..jcall+1)、b(1..jcall+1)
    Return：(sa, sb, a_vec, b_vec, isp, ips)
      * a_vec[i] ↔ Fortran a(i+1)
      * b_vec[i] ↔ Fortran b(i+1)
    """
    # 1) isp/ips, sa/sb
    if na <= nb:
        isp, ips = 1, 2
        sa, sb = float(s1), float(s2)
    else:
        isp, ips = 2, 1
        sa, sb = float(s2), float(s1)

    # 2) j and jcall
    j = ii + 2
    if ii > 3:
        j -= 1
    jcall = j - 1 

    # 3) alpha / beta
    alpha = 0.5 * rab * (sa + sb)
    beta  = 0.5 * rab * (sb - sa)

    # 4) a, b array
    a_vec = aintgs_np(alpha, jcall)
    b_vec = bintgs_np(beta,  jcall)

    return sa, sb, a_vec, b_vec, isp, ips