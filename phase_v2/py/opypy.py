import opy
import numpy as np
from typing import Literal
import enum

Orders = Literal["freq_rate0", "kuramoto", "freq_mean0",
                 "relative_kuramoto", "num_of_avg_freq_mode"]


class OrderEvaluator:
    def __init__(self,
                 window: int,
                 epsilon: float,
                 sampling_dt: float,
                 max_iteration: int,
                 ndim: int,
                 method: Literal["rk4", "rk45"] = "rk4",
                 order: Orders = 'kuramoto'):
        # NOTE: hard code
        if method == "rk4":
            if order == "freq_rate0":
                self._ = opy.ZeroFreqRateEvaluatorRK4(
                    window, epsilon, sampling_dt, max_iteration, ndim, update_dt=0.01)
            elif order == "kuramoto":
                self._ = opy.KuramotoOrderEvaluatorRK4(
                    window, epsilon, sampling_dt, max_iteration, ndim, update_dt=0.01)
            elif order == "freq_mean0":
                self._ = opy.ZeroFreqMeanOrderEvaluatorRK4(
                    window, epsilon, sampling_dt, max_iteration, ndim, update_dt=0.01)
            else:
                raise NotImplementedError

        if method == "rk45":
            if order == "freq_rate0":
                self._ = opy.ZeroFreqRateEvaluatorRK45(
                    window, epsilon, sampling_dt, max_iteration, ndim, start_dt=0.01, max_dt=1.0, atol=1e-3)
            elif order == "kuramoto":
                self._ = opy.KuramotoEvaluatorRK45(
                    window, epsilon, sampling_dt, max_iteration, ndim, start_dt=0.01, max_dt=1.0, atol=1e-3)
            elif order == "freq_mean0":
                self._ = opy.ZeroFreqMeanEvaluatorRK45(
                    window, epsilon, sampling_dt, max_iteration, ndim, start_dt=0.01, max_dt=1.0, atol=1e-3)
            elif order == "relative_kuramoto":
                self._ = opy.RelativeKuramotoEvaluatorRK45(
                    window, epsilon, sampling_dt, max_iteration, ndim, start_dt=0.01, max_dt=1.0, atol=1e-3)
            elif order == "num_of_avg_freq_mode":
                self._ = opy.NumOfAvgFreqMode_RK45(
                    window, epsilon, sampling_dt, max_iteration, ndim, start_dt=0.01, max_dt=1.0, atol=1e-3)
            else:
                raise NotImplementedError

    @classmethod
    def default(cls, ndim, order: Orders = "kuramoto", window=3000, epsilon=1e-4, sampling_dt=1):
        return OrderEvaluator(window=window, epsilon=epsilon, sampling_dt=sampling_dt, max_iteration=window * 4, ndim=ndim, method='rk45', order=order)

    def eval(self, K, w):
        status = self._.eval(np.array(K), np.array(w))
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
        if order == "kuramoto":
            self._ = opy.Kuramoto_MCMC(np.array(w), np.array(
                initial_K), threshold, beta, scale, seed)
        elif order == "relative_kuramoto":
            self._ = opy.RelativeKuramoto_MCMC(np.array(w), np.array(
                initial_K), threshold, beta, scale, seed)
        elif order == "num_of_avg_freq_mode":
            self._ = opy.NumOfAvgFreqMode_MCMC(np.array(w), np.array(
                initial_K), threshold, beta, scale, seed)
        else:
            raise NotImplementedError()

    def step(self) -> MCMCResult:
        result = self._.step()
        return MCMCResult(result)

    def try_swap(self, lhs: "MCMC"):
        assert type(self._) == type(lhs._)
        return self._.try_swap(lhs._)

    def state(self) -> np.ndarray:
        return self._.state()

    def energy(self):
        return self._.energy()
