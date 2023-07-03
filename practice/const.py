import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

cwd = Path(__file__).absolute().parent

df = pd.read_csv(cwd / "const.csv")

x = df.to_numpy().T

_x = np.linspace(-3, 3, 100)
_y = _x * _x + 3.0

plt.plot(_x, _y)
plt.scatter(x[0], x[1], alpha=0.6, label="not mcmc", s=5)
plt.scatter(x[2], x[3], alpha=0.8, c="r", s=5)
plt.legend()
plt.tight_layout()
plt.savefig(cwd / "const.png")
plt.close()
