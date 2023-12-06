import newopy_impl
import numpy as np
from typing import Literal, Optional
import enum

Orders = Literal["kuramoto", "max_avg_freq_cluster"]


def snake_to_UCC(snake):
    return "".join([key[0].upper() + key[1:] for key in snake.split("_")])


class OrderEvaluator:
    def __init__(self,
                 window: int,
                 epsilon: float,
                 Dt: float,
                 max_iter: int,
                 ndim: int,
                 order: Orders = 'kuramoto'):
        # NOTE: hard code
        assert ndim <= 16
        self._ = getattr(newopy_impl.evaluation, f"{snake_to_UCC(order)}_{ndim}")(
            window=window, epsilon=epsilon, Dt=Dt, max_iter=max_iter)

    @classmethod
    def default(cls, ndim, order: Orders = "kuramoto", window=3000, epsilon=1e-4, Dt=1):
        return OrderEvaluator(window=window, epsilon=epsilon, Dt=Dt, max_iter=window * 4, ndim=ndim, order=order)

    def eval(self, K, w):
        status, _ = self._.eval(np.array(K), np.array(w))
        return status

    def result(self):
        return self._.result()


class MCMCResult(enum.Enum):
    Accepted = 0
    Rejected = 1
    MinusConnection = 2
    SmallOrder = 3
    NotConverged = 4


class MCMC:
    def __init__(self, w, initial_K, threshold, beta, scale, seed, order: Orders = "kuramoto"):
        ndim = w.shape[0]
        assert ndim <= 16
        self._ = getattr(newopy_impl.single_mcmc, f"{snake_to_UCC(order)}_{ndim}")(
            np.array(w), np.array(initial_K), threshold, beta, scale, seed)

    def step(self) -> MCMCResult:
        result = self._.step()
        return MCMCResult(int(result))

    def try_swap(self, lhs: "MCMC") -> bool:
        assert type(self._) == type(lhs._)
        return bool(self._.try_swap(lhs._))

    def state(self) -> np.ndarray:
        return self._.state()

    def energy(self):
        return self._.energy()
