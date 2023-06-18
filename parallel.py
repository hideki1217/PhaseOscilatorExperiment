import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

cwd = Path(__file__).absolute().parent

x = np.loadtxt(cwd / "parallel.csv", delimiter=",", skiprows=1).T
R = x.shape[0]//2

ms = np.array([(1, 1), (-1, -1)]).T

for i in range(R):
    plt.scatter(x[2*i], x[2*i + 1], s=5, label=f"{i}")
# plt.scatter(ms[0], ms[1], c="r")
plt.legend()
plt.tight_layout()
plt.savefig(cwd / "parallel.png")
plt.close()


x = np.loadtxt(cwd / "parallel_swap.csv", delimiter=",").T
index = list(range(x.shape[1]))
plt.plot(index, x[0], label=f"{0}")
plt.plot(index, x[-1], label=f"{x.shape[0]-1}")
plt.legend()
plt.tight_layout()
plt.savefig(cwd / "parallel_swap.png")
plt.close()
