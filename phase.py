import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml
from sklearn.cluster import KMeans

cwd = Path(__file__).absolute().parent
with open(cwd / "phase.out") as f:
    data = Path(f.readline())
with open("./phase_param.yaml") as f:
    param = yaml.safe_load(f)

# 交換の図
x = np.loadtxt(data / "phase_swap.csv", delimiter=",").T
index = list(range(x.shape[1]))
plt.plot(index, x[0], label=f"{0}")
plt.plot(index, x[-1], label=f"{x.shape[0]-1}")
plt.legend()
plt.tight_layout()
plt.savefig(data / "phase_swap.png")
plt.close()

# エネルギー分布
x = np.loadtxt(data / "phase_E.csv", delimiter=",").T
for i in range(x.shape[0]):
    plt.hist(x[i], bins=200, alpha=1.0 - (1.0 - 0.5) / x.shape[0] * (x.shape[0] - 1 - i), label=f"{i}")
plt.legend()
plt.tight_layout()
plt.savefig(data / "phase_E.png")
plt.close()

# クラスタの数を計算
C = len(param["beta"])
K = np.loadtxt("./phase.csv", delimiter=",").T
D = K.shape[0] // C
K_top = K[D*(C-1):].T
distortions = []
ns = list(range(1, min(21, K.shape[1])))
for i in ns:
    print(i)
    km = KMeans(n_clusters=i,
                init='k-means++',     # k-means++法によりクラスタ中心を選択
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(K_top)                         # クラスタリングの計算を実行
    distortions.append(km.inertia_) 
plt.plot(ns, distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.savefig(data / "phase_K_cluster.png")
plt.close()