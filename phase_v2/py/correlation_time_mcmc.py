import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import dataclasses
import json
import hashlib

import opypy
import common


@dataclasses.dataclass
class Param:
    ndim: int
    order: opypy.Orders
    threshold: float
    scale: float = 1.19
    beta: float = 1.0
    seed: int = 42

    def __str__(self):
        return f"{self.ndim}_{str(self.order)}_{self.threshold:.3f}_{self.scale:.8f}_{self.beta:.4f}_{self.seed}"


def experiment(datadir: Path, p: Param, MaxWindow: int):
    indentity = f"{p.ndim}_{p.order}_{p.threshold:.3f}_{hashlib.md5(str(p).encode('utf-8')).hexdigest()}"
    datadir = datadir / indentity
    if not datadir.exists():
        datadir.mkdir()

        # Store parameters
        with open(datadir / "param.json", mode="w") as f:
            json.dump(dataclasses.asdict(p), f)

    BurnIn = MaxWindow
    SampleN = MaxWindow * 100

    w = common.create_w(p.ndim)
    K = np.array([[5. * (i != j) for i in range(p.ndim)]
                 for j in range(p.ndim)])

    sampled_K_file = datadir / "sampled_K.npy"
    print(f"Search: {sampled_K_file}")
    if sampled_K_file.exists():
        K_list = np.load(sampled_K_file)
    else:
        mcmc = opypy.MCMC(w, K, p.threshold, p.beta, p.scale, p.seed, p.order)
        print("Start: Burn-in")
        for _ in range(BurnIn):
            mcmc.step()
        print("End: Burn-in")

        print("Start: Sampling")
        K_list = []
        for i in range(SampleN):
            K_list.append(mcmc.state().copy())
            mcmc.step()
        print("end: Sampling")

        K_list = np.stack(K_list)
        np.save(sampled_K_file, K_list)

    mean_K = np.mean(K_list, axis=0)

    w_list = list(range(MaxWindow))
    KK_list = []
    dK_list = K_list - mean_K
    for w in range(MaxWindow):
        KK_list.append(
            np.mean(dK_list[:-w] * dK_list[w:] if w != 0 else dK_list**2))
    KK_list = np.array(KK_list)
    KK_list = KK_list / KK_list[0]

    fig, ax = plt.subplots()
    ax.scatter(w_list, KK_list, s=2)
    plt.grid()
    plt.tight_layout()
    plt.savefig(datadir / "w_KK.png")
    plt.close()


def main():
    file = Path(__file__)
    data = file.parent / "output" / file.stem
    if not data.exists():
        data.mkdir()

    N_per_f = 300

    for ndim in range(2, 10):
        r = N_per_f * (ndim * (ndim - 1)) // 2
        experiment(data, Param(ndim, "kuramoto", 0.78), r)
        experiment(data, Param(ndim, "num_of_avg_freq_mode", 0.9), r)
        experiment(data, Param(ndim, "relative_kuramoto", 0.78), r)


if __name__ == "__main__":
    main()
