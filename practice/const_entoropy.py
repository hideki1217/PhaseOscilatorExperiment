import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

cwd = Path(__file__).absolute().parent

df = pd.read_csv(cwd / "const.csv")

E_samples = -df["E"].to_numpy()
E_samples = np.sort(E_samples)

plt.hist(E_samples, bins=600)
plt.xlim((1, 8))
plt.tight_layout()
plt.savefig(cwd / "const_E.png")
plt.close()

def entropy(E, bins=10, p_use=0.99):
    n = E.shape[0]
    E = E[E <= E[int(p_use * n)]]
    val, bins = np.histogram(E, bins=bins)
    db = bins[1] - bins[0]
    E = bins[:-1] + db / 2
    S = E + np.log(val / n)
    return E, (S - S[0])

for bin in [10, 20, 40, 80, 160, 320, 640, 1280, 2560]:
    E, S = entropy(E_samples, bins=bin)
    plt.plot(E, S, label=f"{bin}")

plt.legend()
plt.tight_layout()
plt.savefig(cwd / "const_entropy.png")
plt.close()
