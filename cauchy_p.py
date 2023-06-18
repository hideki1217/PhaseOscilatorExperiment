import numpy as np

scale = 1.0
N = 100000
M = 100
C = 6

res = np.zeros(C)
for _ in range(M):
    x = np.abs(np.random.standard_cauchy(size=N) / scale)
    x = np.sort(x)

    dN = N // C
    res += x[np.array([dN * i for i in range(C)])]

res /= M

print(",".join([str(round(x, 4)) for x in res]))