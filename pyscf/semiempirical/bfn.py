import numpy as np

def _factorials_upto(n: int) -> np.ndarray:
    f = np.ones(n + 1, dtype=np.float64)
    for k in range(1, n + 1):
        f[k] = f[k - 1] * k
    return f

def bfn_scalar(x: float) -> np.ndarray:
    """
    Fortran  bf(1..13)
    """
    k = 12
    io = 0
    bf = np.zeros(k + 1, dtype=np.float64)  # bf[0] = B_0, ..., bf[12] = B_12
    absx = abs(x)

    # |x| <= 3 
    if absx <= 3.0:
        if absx > 2.0:
            last = 15
        elif absx > 1.0:
            last = 12
        elif absx > 0.5:
            last = 7
        elif absx <= 1.0e-6:
            # limit x -> 0 ：B_i(0) = 2/(i+1) 
            for i in range(io, k + 1):
                bf[i] = (2 * ((i + 1) % 2)) / (i + 1.0)
            return bf
        else:
            last = 6

        fact = _factorials_upto(last)
        for i in range(io, k + 1):
            y = 0.0
            for m in range(io, last + 1):
                parity = 2 * ((m + i + 1) % 2)  # 2 or 0
                if parity:
                    y += ((-x) ** m) * parity / (fact[m] * (m + i + 1))
            bf[i] = y
        return bf

    # |x| > 3 
    expx = np.exp(x)
    expmx = 1.0 / expx
    bf[0] = (expx - expmx) / x
    for i in range(1, k + 1):
        bf[i] = (i * bf[i - 1] + ((-1) ** i) * expx - expmx) / x
    return bf

def bfn_np(x):
    """
    vectorized wrapper of bfn_scalar：
    - scalar input, return shape (13,);
    - array input, return shape x.shape + (13,).
    """
    x_arr = np.asarray(x)
    if x_arr.ndim == 0:
        return bfn_scalar(float(x_arr))
    flat = x_arr.ravel()
    out = np.empty((flat.size, 13), dtype=np.float64)
    for idx, xv in enumerate(flat):
        out[idx] = bfn_scalar(float(xv))
    return out.reshape(x_arr.shape + (13,))