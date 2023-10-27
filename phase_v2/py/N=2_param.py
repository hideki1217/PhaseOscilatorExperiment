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

    def f(K, w, window=30000, sampling_dt=0.01, epsilon=1e-4):
        print([K, w, window, sampling_dt, epsilon])
        ndim = 2
        w = np.array([-w, w])
        K_ = np.array([0, K, K, 0])
        model = opypy.OrderEvaluator(
            window=window, epsilon=epsilon, sampling_dt=sampling_dt, max_iteration=window*3, ndim=ndim)

        status = model.eval(K_, w)

        if (status.value == 0):
            return model.result()
        else:
            return np.nan

    K_list = np.array([0.5, 0.99, 1.01, 2])
    w = 1.0
    sampling_dt = 0.01
    epsilon = 1e-4
    window = 30000

    windows = list(map(lambda T: int(T / sampling_dt), range(100, 1000, 1)))
    for K in K_list:
        R = theoritical_2d(K, w)
        plt.plot(windows, [f(window=window, epsilon=epsilon,
                 sampling_dt=sampling_dt, K=K, w=w) - R for window in windows], label=f"K = {K:.3f}")
    plt.tight_layout()
    plt.legend()
    plt.savefig(data / "K_window.png")
    plt.close()

    sampling_dts = np.power(10, np.linspace(-2, -1, 100))
    for K in K_list:
        R = theoritical_2d(K, w)
        plt.plot(sampling_dts, [f(window=window, epsilon=epsilon,
                 sampling_dt=sampling_dt, K=K, w=w) - R for sampling_dt in sampling_dts], label=f"K = {K:.3f}")
    plt.tight_layout()
    plt.legend()
    plt.xscale("log")
    plt.savefig(data / "K_samplingdt.png")
    plt.close()

    epsilons = np.power(10, np.linspace(-6, -1, 100))
    for K in K_list:
        R = theoritical_2d(K, w)
        plt.plot(epsilons, [f(window=window, epsilon=epsilon,
                 sampling_dt=sampling_dt, K=K, w=w) - R for epsilon in epsilons], label=f"K = {K:.3f}")
    plt.tight_layout()
    plt.legend()
    plt.xscale("log")
    plt.savefig(data / "K_epsilon.png")
    plt.close()


if __name__ == "__main__":
    main()
