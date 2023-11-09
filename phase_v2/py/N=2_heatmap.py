import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import opypy


def experiment(datadir: Path, order: opypy.Orders):
    datadir = datadir / str(order)
    if not datadir.exists():
        datadir.mkdir()
    else:
        return

    ndim = 2
    w = np.array([-1, 1.])
    model = opypy.OrderEvaluator.default(ndim, order=order)

    def f(K1, K2, w):
        print(f"{K1}, {K2}")
        K_ = np.array([0, K1,
                       K2, 0])

        status = model.eval(K_, w)

        if (status.value == 0):
            return model.result()
        else:
            return np.nan

    K_list = np.linspace(0, 3, 100)
    R_map = np.array([[f(K1, K2, w) for K1 in K_list] for K2 in K_list])
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.imshow(R_map, vmin=0, vmax=1)
    plt.tight_layout()
    plt.savefig(datadir / "K1_K2_L=3.png")
    plt.close()

    K_list = np.linspace(0, 3, 1000)
    R_list = np.array([f(K, K, w) for K in K_list])
    plt.plot(K_list, R_list)
    plt.tight_layout()
    plt.savefig(datadir / "K1=K2_L=3.png")
    plt.close()


def main():
    file = Path(__file__)
    data = file.parent / "output" / file.stem
    if not data.exists():
        data.mkdir()

    experiment(data, "kuramoto")
    experiment(data, "relative_kuramoto")
    experiment(data, "freq_rate0")
    experiment(data, "freq_mean0")


if __name__ == "__main__":
    main()
