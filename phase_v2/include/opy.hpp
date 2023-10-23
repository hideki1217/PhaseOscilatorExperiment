#include <order.hpp>
#include <sim.hpp>
#include <vector>

namespace lib {

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
        avg_new(window, ndim),
        avg_old(window, ndim),
        sim_engine(ndim, dt) {
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
    const auto R_new = _R = avg_new.value();
    const auto R_old = avg_old.value();
    return std::abs(R_new - R_old) < epsilon;
  }

  void recorder_regist(const Real *s) { avg_old.push(avg_new.push(s)); }

  Real _R = -1;
  order::AverageOrder<Real> avg_new, avg_old;

  std::vector<Real> s;
  sim::RK4<Real> sim_engine;
};
}  // namespace lib