import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import numpy as np
from pathlib import Path
import matplotlib.animation as anim

cwd = Path(__file__).absolute().parent

X_ls = np.loadtxt(cwd / "parallel.csv", delimiter=",", skiprows=1)
D = 2
X_ls = X_ls.reshape((X_ls.shape[0], -1, D))
R = X_ls.shape[1]

ms = np.array([(1, 1), (-1, -1)]).T

swap_ls = np.loadtxt(cwd / "parallel_swap.csv", delimiter=",")
index = list(range(swap_ls.shape[1]))

c = 2
N = X_ls.shape[0]
S_ls = swap_ls[:N]
s_ls = np.where(S_ls == c)[1]
x_ls = np.array([X_ls[i, c] for i, c in enumerate(s_ls)])
cmap = plt.get_cmap("viridis")
event_ls = (s_ls[:-1] != s_ls[1:])
color_ls = (s_ls[:-1] - s_ls[1:]) * s_ls[:-1]
color_ls = (color_ls - color_ls.min()) / (2 * (R-1))

fig = plt.figure()
def plot(idx):
    plt.cla()
    t = idx * 16
    plt.plot(x_ls[t:t+32, 0], x_ls[t:t+32, 1], linewidth=0.3, c=cmap(1-s_ls[t]/(R-1)))
    plt.plot(x_ls[t+32:t+64, 0], x_ls[t+32:t+64, 1], c=cmap(1-s_ls[t+32]/(R-1)))
    plt.xlim((-2, 2))
    plt.ylim((-2, 2))
    # plt.colorbar(mappable=mcm.ScalarMappable(cmap=cmap))nf7tipkvi73j4
    plt.title(f't = {t}, T_level = {s_ls[t+1]}')

ani = anim.FuncAnimation(fig, plot, frames=1100)
ani.save(cwd / "parallel.gif")
plt.close()