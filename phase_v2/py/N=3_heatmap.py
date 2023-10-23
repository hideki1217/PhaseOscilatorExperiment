import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import opypy


def main():
    file = Path(__file__)
    data = file.parent / "output" / file.stem
    if not data.exists():
        data.mkdir()

    ndim = 3
    w = np.array([-1, 0, 1])
    model_2d = opypy.OrderEvaluator(
        window=30000, epsilon=1e-4, dt=0.01, max_iteration=100000, ndim=ndim)

    def f(K1, K2, K3, w):
        K_ = np.array([0, K1, K2,
                       K1, 0, K3,
                       K2, K3, 0])

        status = model_2d.eval(K_, w)

        print(f"{K1}, {K2}")
        if (status.value == 0):
            return model_2d.result()
        else:
            return np.nan

    R_map = np.array([[f(K1, K2, K1, w)for K1 in np.linspace(0, 2 / np.sqrt(2), 100)]
                      for K2 in np.linspace(0, 2, 100)])
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.imshow(R_map, vmin=0, vmax=1)
    plt.tight_layout()
    plt.savefig(data / "K2_K1=K3_L=2.png")
    plt.close()

    R_map = np.array([[f(0, K2, K3, w)for K2 in np.linspace(0, 2, 100)]
                      for K3 in np.linspace(0, 2, 100)])
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.imshow(R_map, vmin=0, vmax=1)
    plt.tight_layout()
    plt.savefig(data / "K2_K3_K1=0_L=2.png")
    plt.close()

    for K2 in [0, 0.1, 0.5, 1.0, 2.0]:
        R_map = np.array([[f(K1, K2, K3, w)for K1 in np.linspace(0, 2, 100)]
                          for K3 in np.linspace(0, 2, 100)])
        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        ax.imshow(R_map, vmin=0, vmax=1)
        plt.tight_layout()
        plt.savefig(data / f"K1_K3_K2={K2:.1f}_L=2.png")
        plt.close()


if __name__ == "__main__":
    main()