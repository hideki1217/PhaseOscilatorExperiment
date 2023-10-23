#include <algorithm>
#include <cmath>
#include <collection.hpp>
#include <deque>
#include <sim.hpp>
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
        ndim(ndim),
        sim_engine(ndim, dt) {
    cos_q_new.resize(window);
    cos_q_old.resize(window);
    sin_q_new.resize(window);
    sin_q_old.resize(window);

    s.resize(ndim);
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
      t = sim_engine.advance_dt(t, &s[0], K, w);
      recorder_regist(&s[0]);
    }
    iteration += window * 2;

    while (!recorder_check()) {
      if (iteration > max_iteration) {
        return EvalStatus::NotConverged;
      }

      iteration++;
      t = sim_engine.advance_dt(t, &s[0], K, w);
      recorder_regist(&s[0]);
    }

    return EvalStatus::Ok;
  }

  /**
   * Return evaluation result.
   */
  Real result() { return _R; }

 private:
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

  Real _R = -1;
  collection::FixedQueue<Real> cos_q_new, cos_q_old;
  collection::FixedQueue<Real> sin_q_new, sin_q_old;

  std::vector<Real> s;
  sim::RK4<Real> sim_engine;
};

}  // namespace order
}  // namespace lib
