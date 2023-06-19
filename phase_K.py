import sys
import yaml

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

with open("./phase_param.yaml") as f:
    param = yaml.safe_load(f)

C = len(param["beta"])

K = np.loadtxt("./phase.csv", delimiter=",").T
D = K.shape[0] // C

K_top = K[D*(C-1):].T

distortions = []

ns = list(range(1, 21))
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
plt.savefig("./phase_K_cluster.png")
plt.close()
