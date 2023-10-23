import window_rate

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    file = Path(__file__)
    data = file.parent / "output" / file.stem
    if not data.exists():
        data.mkdir()

    ndim = 2

    for k in [0, 0.5, 0.99, 1.01, 2.0]:
        K = np.array([k] * (ndim**2)).reshape((ndim, ndim))
        K[np.diag_indices_from(K)] = 0
        K = K.flatten()

        w = np.array([-1., 1])

        for window in [1000, 10000, 30000]:
            R_sampling = window_rate.OrderChain(
                window=window, dt=0.01, ndim=ndim).eval(1000, K, w)

            plt.hist(R_sampling, bins=50, label=f"{window}")
        plt.legend()
        plt.savefig(f"./K={k:.3f}.png")
        plt.close()


if __name__ == "__main__":
    main()
