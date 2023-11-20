import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def target_model(t, x, K, w):
    assert K.ndim == 2

    x = x.reshape((-1, 1))
    return w + np.sum(K * np.sin(x.T - x), axis=1)


def figure_1(datadir):
    t_span = (0, 5)
    t_list = np.linspace(t_span[0], t_span[1], 1000)

    w = np.array([4., 4.])

    K = np.array([0, 0, 0, 0]).reshape((2, 2))
    sol = integrate.solve_ivp(target_model, t_span,
                              (-np.pi / 4, np.pi / 4), args=(K, w), dense_output=True, atol=1e-10)
    not_sync = sol.sol(t_list)

    K = np.array([0, 0.6, 0.6, 0]).reshape((2, 2))
    sol = integrate.solve_ivp(target_model, t_span,
                              (-np.pi / 4, np.pi / 4), args=(K, w), dense_output=True, atol=1e-10)
    sync = sol.sol(t_list)

    plt.plot(t_list, np.sin(not_sync[0]))
    plt.plot(t_list, np.sin(not_sync[1]))
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.tight_layout()
    plt.savefig(datadir / "figure_1_nsync.png")
    plt.close()

    plt.plot(t_list, np.sin(sync[0]))
    plt.plot(t_list, np.sin(sync[1]))
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.tight_layout()
    plt.savefig(datadir / "figure_1_sync.png")
    plt.close()


def main():
    file = Path(__file__)
    data = file.parent / "output" / file.stem
    if not data.exists():
        data.mkdir()

    figure_1(data)


if __name__ == "__main__":
    main()
