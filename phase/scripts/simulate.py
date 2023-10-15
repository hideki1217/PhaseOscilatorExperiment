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

    def solve_ivp(self, K, w0, sampling_times):
        t0 = sampling_times.min()
        t2 = sampling_times.max()

        n = K.shape[0]
        sol = integrate.solve_ivp(fun=self.f, t_span=(t0, t2), y0=np.zeros(
            n), args=(K, w0), dense_output=True, max_step=1)
        ss = sol.sol(sampling_times).T
        return ss

    def solve_rk4(self, K, w0, dt=0.01, T=100):
        n = K.shape[0]
        res = list(
            take(int(T / dt), rk4(self.f, np.zeros(n), args=(K, w0), dt=dt)))
        sampling_times = np.array([t for (t, x) in res])
        samlping_states = np.stack([x for (t, x) in res])
        return sampling_times, samlping_states

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


def rk4(f, x0, dt=0.01, t0=0., args=None):
    t = t0
    x = np.array(x0)
    if args is not None:
        def F(t, x): return f(t, x, *args)
    else:
        F = f
    while True:
        k1 = dt * F(t, x)
        k2 = dt * F(t+0.5 * dt, x + 0.5 * k1)
        k3 = dt * F(t+0.5 * dt, x + 0.5 * k2)
        k4 = dt * F(t + dt, x + k3)
        x += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += dt

        yield (t, x.copy())


def take(n, itr):
    for _ in range(n):
        yield next(itr)
