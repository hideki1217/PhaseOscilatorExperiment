import numpy as np
from scipy import integrate


class OscilatorNetwork:
    @staticmethod
    def f(t, s, K, w0):
        s = s.reshape((-1, 1))
        return w0 + np.sum(K * np.sin(s.T - s), axis=1)

    @staticmethod
    def Df(t, x, K, w0):
        res = K * np.cos(x.T - x)
        res[np.diag_indices_from(res)] = -np.sum(res, axis=1)
        return res

    def order_rk4(self, K, w0, burn_in=5000, T=500, N=10000):
        n = len(w0)

        (t0, t1, t2) = (0, burn_in, burn_in + T)

        sampling_times = np.linspace(t2, t1, N)

        sol = integrate.solve_ivp(fun=self.f, t_span=(t0, t2), y0=np.zeros(
            n), args=(K, w0), dense_output=True, max_step=1)
        ss = sol.sol(sampling_times).T
        Rs = np.sqrt(np.sum(np.sin(ss), axis=1)**2 +
                     np.sum(np.cos(ss), axis=1)**2)/n
        return np.mean(Rs), ss

    def order_newton(self, K, w0, iteration=50, x0=None):
        n = len(w0)

        x = np.array([0.] * n) if x0 is None else x0
        for i in range(iteration):
            _Df = self.Df(0, x, K, w0) + np.diag([1e-12]*n)
            _f = self.f(0, x, K, w0)
            r = np.linalg.solve(_Df, _f)
            x -= r
        R = np.sqrt(np.sum(np.sin(x))**2 + np.sum(np.cos(x))**2)/n
        return R
