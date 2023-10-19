import numpy as np


def equal_ratio(base: float, r: float, n: int):
    return [base * r**i for i in range(n)]


def cauchy_nth(n, m, c, scale):
    res = np.zeros(c)
    for _ in range(m):
        x = np.abs(np.random.standard_cauchy(size=n) / scale)
        x = np.sort(x)

        dN = n // c
        res += x[np.array([dN * i for i in range(c)])]

    res /= m
    return res

def create_vec(name: str, ls: list, ctype: str = "double"):
    return f"const auto {name} = std::vector<{ctype}>({{{','.join(map(str, ls))}}})"

def create(*, b_base, b_r, b_n, w0_scale,
           w0_n,
           w0_m,
           w0_c):
    b = equal_ratio(b_base, b_r, b_n)
    _w0 = cauchy_nth(w0_n, w0_m, w0_c, w0_scale)
    w0 = sorted([-x for x in _w0] + [0] + _w0.tolist())
    
    print(create_vec("betas", b) + ";")
    print(create_vec("w0", w0) + ";")
