import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import opypy


def main():
    file = Path(__file__)
    data = file.parent / "output" / file.stem
    if not data.exists():
        data.mkdir()

    ndim = 2
    w = np.array([-1, 1.])
    model = opypy.OrderEvaluator(
        window=30000, epsilon=1e-4, dt=0.01, max_iteration=100000, ndim=ndim)

    def f(K1, K2, w):
        K_ = np.array([0, K1,
                       K2, 0])

        status = model.eval(K_, w)

        if (status.value == 0):
            return model.result()
        else:
            return np.nan

    K_list = np.linspace(0, 3, 100)
    R_map = np.array([[f(K1, K2, w) for K1 in K_list] for K2 in K_list])

    plt.plot(K_list, np.diag(R_map))
    plt.tight_layout()
    plt.savefig(data / "K1=K2_L=3.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.imshow(R_map, vmin=0, vmax=1)
    plt.tight_layout()
    plt.savefig(data / "K1_K2_L=3.png")
    plt.close()


if __name__ == "__main__":
    main()
