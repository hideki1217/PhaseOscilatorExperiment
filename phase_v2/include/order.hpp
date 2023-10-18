#include <algorithm>
#include <cmath>
#include <deque>
#include <vector>

namespace lib {
namespace order {

enum EvalStatus {
  Ok = 0,
  NotConverged = 1,
};

template <typename Real = double>
class OrderEvaluator {
 public:
  const int window;
  const Real epsilon;
  const Real dt;
  const int max_iteration;
  const int ndim;
  OrderEvaluator(int window, Real epsilon, Real dt, int max_iteration, int ndim)
      : window(window),
        epsilon(epsilon),
        dt(dt),
        max_iteration(max_iteration),
        ndim(ndim) {
    cos_q_new.resize(window);
    cos_q_old.resize(window);
    sin_q_new.resize(window);
    sin_q_old.resize(window);

    s.resize(ndim);
    _s.resize(ndim);
    k1.resize(ndim);
    k2.resize(ndim);
    k3.resize(ndim);
    k4.resize(ndim);
  }

  /**
   * Evaluate phase order parameter of a specified oscilator network
   * And return the status flag
   */
  EvalStatus eval(const Real *K, const Real *w) {
    recorder_init();
    std::fill(s.begin(), s.end(), 0);
    Real t = 0;
    int iteration = 0;

    for (int i = 0; i < window * 2; i++) {
      t = advance_dt(t, &s[0], K, w);
      recorder_regist(&s[0]);
    }
    iteration += window * 2;

    while (!recorder_check()) {
      if (iteration > max_iteration) {
        return EvalStatus::NotConverged;
      }

      iteration++;
      t = advance_dt(t, &s[0], K, w);
      recorder_regist(&s[0]);
    }

    return EvalStatus::Ok;
  }

  /**
   * Return evaluation result.
   */
  Real result() { return _R; }

 private:
  Real advance_dt(Real t, Real *s, const Real *K, const Real *w) {
    const Real dt_2 = dt * 0.5;
    const Real dt_6 = dt / 6;

    time_diff(K, w, t, &s[0], &k1[0]);
    for (int i = 0; i < ndim; i++) _s[i] = s[i] + dt_2 * k1[i];
    time_diff(K, w, t + dt_2, &_s[0], &k2[0]);
    for (int i = 0; i < ndim; i++) _s[i] = s[i] + dt_2 * k2[i];
    time_diff(K, w, t + dt_2, &_s[0], &k3[0]);
    for (int i = 0; i < ndim; i++) _s[i] = s[i] + dt * k3[i];
    time_diff(K, w, t + dt, &_s[0], &k4[0]);
    for (int i = 0; i < ndim; i++) {
      s[i] += dt_6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    }

    return t + dt;
  }

  void time_diff(const Real *K, const Real *w, const Real t, const Real *s,
                 Real *ds_dt) {
    for (int i = 0; i < ndim; i++) {
      ds_dt[i] = w[i];
    }

    for (int i = 0; i < ndim; i++) {
      for (int j = 0; j < ndim; j++) {
        ds_dt[i] += K[i * ndim + j] * std::sin(s[j] - s[i]);
      }
    }
  }

  bool recorder_check() {
    const auto cos_mean_new = cos_sum_new / window;
    const auto cos_mean_old = cos_sum_old / window;
    const auto sin_mean_new = sin_sum_new / window;
    const auto sin_mean_old = sin_sum_old / window;

    const auto R_new = _R =
        std::sqrt(cos_mean_new * cos_mean_new + sin_mean_new * sin_mean_new);
    const auto R_old =
        std::sqrt(cos_mean_old * cos_mean_old + sin_mean_old * sin_mean_old);
    return std::abs(R_new - R_old) < epsilon;
  }

  void recorder_regist(const Real *s) {
    Real mean, new_pop, old_pop;

    mean = 0;
    for (int i = 0; i < ndim; i++) {
      mean += std::cos(s[i]);
    }
    mean /= ndim;

    new_pop = cos_q_new.front();
    old_pop = cos_q_old.front();
    cos_q_new.pop_front();
    cos_q_old.pop_front();
    cos_q_new.push_back(mean);
    cos_q_old.push_back(new_pop);
    cos_sum_new += mean - new_pop;
    cos_sum_old += new_pop - old_pop;

    mean = 0;
    for (int i = 0; i < ndim; i++) {
      mean += std::sin(s[i]);
    }
    mean /= ndim;

    new_pop = sin_q_new.front();
    old_pop = sin_q_old.front();
    sin_q_new.pop_front();
    sin_q_old.pop_front();
    sin_q_new.push_back(mean);
    sin_q_old.push_back(new_pop);
    sin_sum_new += mean - new_pop;
    sin_sum_old += new_pop - old_pop;
  }

  void recorder_init() {
    std::fill(cos_q_new.begin(), cos_q_new.end(), 0);
    std::fill(cos_q_old.begin(), cos_q_old.end(), 0);
    std::fill(sin_q_new.begin(), sin_q_new.end(), 0);
    std::fill(sin_q_old.begin(), sin_q_old.end(), 0);
    cos_sum_new = cos_sum_old = sin_sum_new = sin_sum_old = 0;
  }

  std::vector<Real> s;
  std::vector<Real> _s;
  std::vector<Real> k1, k2, k3, k4;

  Real _R = -1;
  std::deque<Real> cos_q_new, cos_q_old;
  std::deque<Real> sin_q_new, sin_q_old;
  Real cos_sum_new = 0, sin_sum_new = 0;
  Real cos_sum_old = 0, sin_sum_old = 0;
};

}  // namespace order
}  // namespace lib
