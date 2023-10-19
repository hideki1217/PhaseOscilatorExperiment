import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import opy

model_2d = opy.OrderEvaluator(
    window=30000, epsilon=1e-4, dt=0.01, max_iteration=100000, ndim=2)

K_list = np.linspace(0, 5, 1000)
w = 1


def f(K, w):
    status = model_2d.eval(np.array([0, K, K, 0]),
                           np.array([-w, w]))

    print(f"{K}, {w}")
    if (status.value == 0):
        return model_2d.result()
    else:
        return np.nan


R_list = [f(K, w) for K in K_list]

plt.plot(K_list, R_list)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "output" / "eval_sample.png")
plt.close()
