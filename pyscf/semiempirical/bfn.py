import numpy as np

def _factorials_upto(n: int) -> np.ndarray:
    """返回 [0..n] 的阶乘，float64。n 不大（此处最多 15）。"""
    f = np.ones(n + 1, dtype=np.float64)
    for k in range(1, n + 1):
        f[k] = f[k - 1] * k
    return f

def bfn_scalar(x: float) -> np.ndarray:
    """
    计算 B 函数向量 bf[0..12]，对应 Fortran 的 bf(1..13)。
    完全按原 Fortran 分支与公式实现。
    """
    k = 12
    io = 0
    bf = np.zeros(k + 1, dtype=np.float64)  # bf[0] = B_0, ..., bf[12] = B_12
    absx = abs(x)

    # |x| <= 3 分支（含极小 x 的特例）
    if absx <= 3.0:
        if absx > 2.0:
            last = 15
        elif absx > 1.0:
            last = 12
        elif absx > 0.5:
            last = 7
        elif absx <= 1.0e-6:
            # x -> 0 极限：B_i(0) = 2/(i+1) (i 偶数), 否则 0
            for i in range(io, k + 1):
                bf[i] = (2 * ((i + 1) % 2)) / (i + 1.0)
            return bf
        else:
            last = 6

        # 级数：B_i(x) = Σ_{m=0..last} (-x)^m * 2*mod(m+i+1,2) / (m!*(m+i+1))
        # 注意：mod(m+i+1,2)=1 当 (m+i+1) 为奇数，即 i+m 为偶数（只有此时积分不为 0）
        fact = _factorials_upto(last)
        for i in range(io, k + 1):
            y = 0.0
            for m in range(io, last + 1):
                parity = 2 * ((m + i + 1) % 2)  # 2 或 0
                if parity:
                    y += ((-x) ** m) * parity / (fact[m] * (m + i + 1))
            bf[i] = y
        return bf

    # |x| > 3 分支：解析递推
    expx = np.exp(x)
    expmx = 1.0 / expx
    bf[0] = (expx - expmx) / x
    for i in range(1, k + 1):
        bf[i] = (i * bf[i - 1] + ((-1) ** i) * expx - expmx) / x
    return bf

def bfn_np(x):
    """
    向量化包装：
    - 若 x 为标量，返回形状 (13,) 的 bf；
    - 若 x 为数组，返回形状 x.shape + (13,) 的 bf 堆叠结果。
    """
    x_arr = np.asarray(x)
    if x_arr.ndim == 0:
        return bfn_scalar(float(x_arr))
    flat = x_arr.ravel()
    out = np.empty((flat.size, 13), dtype=np.float64)
    for idx, xv in enumerate(flat):
        out[idx] = bfn_scalar(float(xv))
    return out.reshape(x_arr.shape + (13,))