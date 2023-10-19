import collections


class OnlineMean:
    def __init__(self, window_N, burnin_N) -> None:
        self.window_N = window_N
        self.burnin_N = burnin_N

        self.count_input = 0
        self._last_t = None
        self._sum_old = 0.
        self._sum_new = 0.
        self.q_old = collections.deque()
        self.q_new = collections.deque()

    def append(self, v) -> float:
        """
        Register a state and return the difference between current moving average and previous one
        """
        self.count_input += 1
        if self.count_input < self.burnin_N:
            return 1e10

        self.q_new.append(v)
        self._sum_new += v
        if len(self.q_new) > self.window_N:
            left = self.q_new.popleft()
            self._sum_new -= left

            self.q_old.append(left)
            self._sum_old += left

        if len(self.q_old) > self.window_N:
            left = self.q_old.popleft()
            self._sum_old -= left

        return abs(self._sum_new - self._sum_old) / self.window_N

    def value(self):
        return self._sum_new / self.window_N
