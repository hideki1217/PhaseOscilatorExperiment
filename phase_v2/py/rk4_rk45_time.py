import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

import opypy


def main():
    file = Path(__file__)
    data = file.parent / "output" / file.stem
    if not data.exists():
        data.mkdir()

    k = 0.5
    w = 1.0
    K = np.array([[0, k, k], [k, 0, k], [k, k, 0]]).flatten()
    W = [-w, 0, w]

    def measure_time(window=30000, sampling_dt=1e-2, m=10):
        print(f"{window}, {sampling_dt}")
        rk4 = opypy.OrderEvaluator(
            window, 1e-4, sampling_dt, window * 3, 3, method='rk4')
        rk45 = opypy.OrderEvaluator(
            window, 1e-4, sampling_dt, window * 3, 3, method='rk45')

        m = 100

        now = time.time()
        for _ in range(m):
            rk4.eval(K, W)
        rk4_time_s = (time.time() - now)
        print(f"{rk4_time_s / m:.6f}(s) {rk4.result():.6f}")

        now = time.time()
        for _ in range(m):
            rk45.eval(K, W)
        rk45_time_s = (time.time() - now)
        print(f"{rk45_time_s / m:.6f}(s) {rk45.result():.6f}")

        print(f"rk45 / rk4 = {rk45_time_s / rk4_time_s:.4f}")

    measure_time(sampling_dt=1e-2)
    measure_time(sampling_dt=1e-1)
    measure_time(sampling_dt=1e-1, window=50000)
    measure_time(sampling_dt=1e-1, window=80000)


if __name__ == "__main__":
    main()
