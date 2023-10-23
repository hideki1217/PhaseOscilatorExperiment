#include <algorithm>
#include <cmath>
#include <collection.hpp>
#include <deque>
#include <sim.hpp>
#include <vector>

namespace lib {
namespace order {

template <typename Real>
class AverageOrder {
 private:
  struct InnerUnit {
    Real cos;
    Real sin;

    InnerUnit(Real cos, Real sin) : cos(cos), sin(sin) {}
  };

 public:
  const int ndim;
  AverageOrder(int window, int ndim)
      : ndim(ndim), cos_q(window, 0.), sin_q(window, 0.) {}

  InnerUnit push(const Real *s) {
    Real cos_mean = 0;
    for (int i = 0; i < ndim; i++) {
      cos_mean += std::cos(s[i]);
    }
    cos_mean /= ndim;
    auto cos_pop = cos_q.push(cos_mean);

    Real sin_mean = 0;
    for (int i = 0; i < ndim; i++) {
      sin_mean += std::sin(s[i]);
    }
    sin_mean /= ndim;
    auto sin_pop = sin_q.push(sin_mean);

    return InnerUnit(cos_pop, sin_pop);
  }

  InnerUnit push(const InnerUnit inner) {
    auto cos_pop = cos_q.push(inner.cos);
    auto sin_pop = sin_q.push(inner.sin);
    return InnerUnit(cos_pop, sin_pop);
  }

  Real value() {
    auto cos_mean = cos_q.mean();
    auto sin_mean = sin_q.mean();
    const auto R_new = std::sqrt(cos_mean * cos_mean + sin_mean * sin_mean);
    return R_new;
  }

 private:
  collection::FixedQueue<Real> cos_q;
  collection::FixedQueue<Real> sin_q;
};

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
  AverageOrder<Real> avg_new, avg_old;

  std::vector<Real> s;
  sim::RK4<Real> sim_engine;
};

}  // namespace order
}  // namespace lib
