import numpy as np


def create_w(N):
    # N-fractional point sequence of Cauchy distribution
    def F_inv(p, x0=0., r=1.): return x0 + r * np.tan(np.pi * (p - 0.5))

    return F_inv(np.array(list(range(1, N+1))) / (N + 1))
