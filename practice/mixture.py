import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

cwd = Path(__file__).absolute().parent

x = np.loadtxt(cwd / "mixture.csv", delimiter=",", skiprows=1).T

ms = np.array([(1, 1), (-1, -1)]).T

plt.scatter(ms[0], ms[1], c="r")
plt.scatter(x[0], x[1], s=5)
plt.tight_layout()
plt.savefig(cwd / "mixture.png")
plt.close()
