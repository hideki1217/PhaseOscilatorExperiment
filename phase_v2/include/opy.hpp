#pragma once

#include <cstring>
#include <order.hpp>
#include <sim.hpp>

namespace lib {

enum EvalStatus {
  Ok = 0,
  NotConverged = 1,
};

template <typename Real = double, typename OdeInt = sim::RK4<Real>,
          typename Order = order::Kuramoto<Real>>
class OrderEvaluator {
 public:
  const int window;
  const Real epsilon;
  const Real sampling_dt;
  const int max_iteration;
  const int ndim;
  OrderEvaluator(int window, Real epsilon, Real sampling_dt, int max_iteration,
                 int ndim, OdeInt &&odeint)
      : window(window),
        epsilon(epsilon),
        sampling_dt(sampling_dt),
        max_iteration(max_iteration),
        ndim(ndim),
        avg_new(window, ndim),
        avg_old(window, ndim),
        s(new(std::align_val_t{64}) Real[ndim]),
        s_ini(new(std::align_val_t{64}) Real[ndim]),
        ds_dt(new(std::align_val_t{64}) Real[ndim]),
        sim_engine(std::move(odeint)) {
    // HACK: hard coding
    std::fill_n(&s_ini[0], ndim, Real(0));
  }

  /**
   * Evaluate phase order parameter of a specified oscilator network
   * And return the status flag
   */
  EvalStatus eval(const Real *K, const Real *w) noexcept {
    std::memcpy(&s[0], &s_ini[0], ndim);
    Real t = 0;
    int iteration = 0;

    for (int i = 0; i < window * 2; i++) {
      const auto result =
          sim_engine.advance(sampling_dt, t, &s[0], &K[0], &w[0]);
      t = result.t;
      sim::target_model(ndim, &K[0], &w[0], t, &s[0], &ds_dt[0]);
      recorder_regist(&s[0], &ds_dt[0]);
    }
    iteration += window * 2;

    while (!recorder_check()) {
      if (iteration > max_iteration) {
        return EvalStatus::NotConverged;
      }
      iteration++;

      const auto result =
          sim_engine.advance(sampling_dt, t, &s[0], &K[0], &w[0]);
      t = result.t;
      sim::target_model(ndim, &K[0], &w[0], t, &s[0], &ds_dt[0]);
      recorder_regist(&s[0], &ds_dt[0]);
    }

    return EvalStatus::Ok;
  }

  /**
   * Return evaluation result.
   */
  Real result() const { return _R; }

 private:
  bool recorder_check() {
    const auto R_new = _R = avg_new.value();
    const auto R_old = avg_old.value();
    return std::abs(R_new - R_old) < epsilon;
  }

  void recorder_regist(const Real *s, const Real *ds_dt) {
    avg_old.push(avg_new.push(s, ds_dt));
  }

  Real _R = -1;
  Order avg_new, avg_old;

  std::unique_ptr<Real[]> s;
  std::unique_ptr<Real[]> s_ini;
  std::unique_ptr<Real[]> ds_dt;
  OdeInt sim_engine;
};

template <typename Real = double, typename Order = order::Kuramoto<Real>>
class OrderEvaluatorRK4 : public OrderEvaluator<Real, sim::RK4<Real>, Order> {
 public:
  OrderEvaluatorRK4(int window, Real epsilon, Real sampling_dt,
                    int max_iteration, int ndim, Real update_dt = 0.01)
      : OrderEvaluator<Real, sim::RK4<Real>, Order>(
            window, epsilon, sampling_dt, max_iteration, ndim,
            sim::RK4<Real>(ndim, update_dt)) {}
};

template <typename Real = double, typename Order = order::Kuramoto<Real>>
class OrderEvaluatorRK45
    : public OrderEvaluator<Real, sim::FehlbergRK45<Real>, Order> {
 public:
  OrderEvaluatorRK45(int window, Real epsilon, Real sampling_dt,
                     int max_iteration, int ndim, Real start_dt = 0.01,
                     Real max_dt = 1, Real atol = 1e-3)
      : OrderEvaluator<Real, sim::FehlbergRK45<Real>, Order>(
            window, epsilon, sampling_dt, max_iteration, ndim,
            sim::FehlbergRK45<Real>(ndim, start_dt, max_dt, atol)) {}
};
}  // namespace lib