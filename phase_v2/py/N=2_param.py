import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import opypy


def main():
    file = Path(__file__)
    data = file.parent / "output" / file.stem
    if not data.exists():
        data.mkdir()

    def theoritical_2d(K, w):
        if (K >= w):
            return np.cos(0.5 * np.arcsin(w / K))
        return 0

    def f(K, w, window=30000, sampling_dt=0.01, epsilon=1e-4, max_p=5):
        print([K, w, window, sampling_dt, epsilon])
        ndim = 2
        w = np.array([-w, w])
        K_ = np.array([0, K, K, 0])
        model = opypy.OrderEvaluator(
            window=window, epsilon=epsilon, sampling_dt=sampling_dt, max_iteration=int(window*max_p), ndim=ndim)

        status = model.eval(K_, w)

        if (status.value == 0):
            return model.result()
        else:
            return np.nan

    K_list = np.array([0.5, 0.99, 1.01, 2])
    w = 1.0

    windows = list(range(1000, 10000, 100))
    for T in [500., 1000., 2000.]:
        K = 0.99
        R = theoritical_2d(K, w)
        plt.plot(windows, [f(window=window, K=K, w=w, sampling_dt=T / window) -
                 R for window in windows], label=f"T = {T:.3f}")
    plt.tight_layout()
    plt.legend()
    plt.savefig(data / "window_samplingdt_K=0.99.png")
    plt.close()

    for epsilon in [1e-4, 1e-6, 1e-8]:
        windows = list(range(10000, 100000, 100))
        for K in [0.99, 1.01]:
            R = theoritical_2d(K, w)
            plt.plot(windows, [f(window=window, K=K, w=w, epsilon=epsilon) -
                               R for window in windows], label=f"K = {K:.3f}")
        plt.tight_layout()
        plt.legend()
        plt.savefig(data / f"K_window_epsilon={epsilon}.png")
        plt.close()

    K = 0.99
    R = theoritical_2d(K, w)
    max_ps = np.linspace(2, 5, 100)
    epsilons = np.power(10, np.linspace(-6, -2, 100))
    dR_map = [[abs(f(K=K, w=w, max_p=max_p, epsilon=epsilon) -
               R) for epsilon in epsilons] for max_p in max_ps]
    fig, ax = plt.subplots()
    im = ax.imshow(dR_map, vmin=0)
    fig.colorbar(im, ax=ax)
    plt.savefig(data / f"K_window_max_p.png")
    plt.close()

    for max_p in np.linspace(2.01, 3, 10):
        windows = list(range(10000, 100000, 100))
        for K in [0.99, 1.01]:
            R = theoritical_2d(K, w)
            plt.plot(windows, [f(window=window, K=K, w=w, max_p=max_p) -
                               R for window in windows], label=f"K = {K:.3f}")
        plt.tight_layout()
        plt.legend()
        plt.savefig(data / f"K_window_max_p={max_p}.png")
        plt.close()

    for window in [30000, 50000]:
        sampling_dts = np.power(10, np.linspace(-2, -0.5, 100))
        for K in K_list:
            R = theoritical_2d(K, w)
            plt.plot(sampling_dts, [f(window=window, sampling_dt=sampling_dt, K=K,
                     w=w) - R for sampling_dt in sampling_dts], label=f"K = {K:.3f}")
        plt.tight_layout()
        plt.legend()
        plt.xscale("log")
        plt.savefig(data / f"K_samplingdt_window={window}.png")
        plt.close()

    for window in [30000, 50000]:
        epsilons = np.power(10, np.linspace(-6, -1, 100))
        for K in K_list:
            R = theoritical_2d(K, w)
            plt.plot(epsilons, [f(window=window, epsilon=epsilon, K=K,
                     w=w) - R for epsilon in epsilons], label=f"K = {K:.3f}")
        plt.tight_layout()
        plt.legend()
        plt.xscale("log")
        plt.savefig(data / f"K_epsilon_window={window}.png")
        plt.close()


if __name__ == "__main__":
    main()
