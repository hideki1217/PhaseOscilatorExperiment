import opy
import numpy as np


class OrderEvaluator:
    def __init__(self, window: int, epsilon: float, dt: float, max_iteration: int, ndim: int):
        self._ = opy.OrderEvaluator(
            window, epsilon, dt, max_iteration, ndim)

    def eval(self, K, w):
        status = self._.eval(np.array(K), np.array(w))
        return status

    def result(self):
        return self._.result()
