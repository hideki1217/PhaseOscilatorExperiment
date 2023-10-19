#include <algorithm>
#include <cmath>
#include <collection.hpp>
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
    const auto cos_mean_new = cos_q_new.sum() / window;
    const auto cos_mean_old = cos_q_old.sum() / window;
    const auto sin_mean_new = sin_q_new.sum() / window;
    const auto sin_mean_old = sin_q_old.sum() / window;

    const auto R_new = _R =
        std::sqrt(cos_mean_new * cos_mean_new + sin_mean_new * sin_mean_new);
    const auto R_old =
        std::sqrt(cos_mean_old * cos_mean_old + sin_mean_old * sin_mean_old);
    return std::abs(R_new - R_old) < epsilon;
  }

  void recorder_regist(const Real *s) {
    Real cos_mean = 0;
    for (int i = 0; i < ndim; i++) {
      cos_mean += std::cos(s[i]);
    }
    cos_mean /= ndim;
    cos_q_old.push(cos_q_new.push(cos_mean));

    Real sin_mean = 0;
    for (int i = 0; i < ndim; i++) {
      sin_mean += std::sin(s[i]);
    }
    sin_mean /= ndim;
    sin_q_old.push(sin_q_new.push(sin_mean));
  }

  std::vector<Real> s;
  std::vector<Real> _s;
  std::vector<Real> k1, k2, k3, k4;

  Real _R = -1;
  collection::FixedQueue<Real> cos_q_new, cos_q_old;
  collection::FixedQueue<Real> sin_q_new, sin_q_old;
};

}  // namespace order
}  // namespace lib
