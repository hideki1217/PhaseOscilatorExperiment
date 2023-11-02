import opy
import numpy as np
from typing import Literal


class OrderEvaluator:
    def __init__(self,
                 window: int,
                 epsilon: float,
                 sampling_dt: float,
                 max_iteration: int,
                 ndim: int,
                 method: Literal["rk4", "rk45"] = "rk4"):
        # NOTE: hard code
        if method == "rk4":
            self._ = opy.OrderEvaluatorRK4(
                window, epsilon, sampling_dt, max_iteration, ndim, update_dt=0.01)
        elif method == "rk45":
            self._ = opy.OrderEvaluatorRK45(
                window, epsilon, sampling_dt, max_iteration, ndim, start_dt=0.01, max_dt=1.0, atol=1e-3)

    @classmethod
    def default(cls, ndim):
        return OrderEvaluator(window=30000, epsilon=1e-4, sampling_dt=0.1, max_iteration=100000, ndim=ndim, method='rk45')

    def eval(self, K, w):
        status = self._.eval(np.array(K), np.array(w))
        return status

    def result(self):
        return self._.result()
