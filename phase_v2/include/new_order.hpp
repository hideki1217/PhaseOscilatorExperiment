#pragma once

#include <algorithm>
#include <boost/numeric/odeint.hpp>
#include <cmath>
#include <new_collection.hpp>
#include <new_system.hpp>
#include <random>
#include <valarray>

using namespace boost::numeric;

namespace new_lib {
namespace order {
enum EvalStatus {
  Ok = 0,
  NotConverged = 1,
};
template <typename Order>
class Evaluator {
 public:
  using system_t = typename Order::system_t;
  using real_t = typename system_t::real_t;
  using state_t = typename system_t::state_t;
  static constexpr int ndim = system_t::ndim;

 public:
  const int window;
  const real_t epsilon;
  const real_t Dt;
  const int max_iter;

 public:
  Evaluator(int window, real_t epsilon, real_t Dt, int max_iter)
      : window(window),
        epsilon(epsilon),
        Dt(Dt),
        max_iter(max_iter),
        stepper(odeint::make_controlled<odeint::runge_kutta_dopri5<state_t>>(
            1e-4, 1e-6)),
        avg_new(window),
        avg_old(window) {}

  EvalStatus eval(const real_t* K, int Kstride, const real_t* w) noexcept {
    int iteration = 0;

    system_t system(K, Kstride, w);
    std::fill(x.begin(), x.end(), static_cast<real_t>(0));  // HACK: hard coding

    for (int i = 0; i < window * 2; i++) {
      odeint::integrate_const(stepper, system, x, 0.0, Dt, Dt);
      system(x, dx, 0);

      recorder_regist(x, dx);
    }
    iteration += window * 2;

    while (!recorder_check()) {
      if (iteration > max_iter) {
        return EvalStatus::NotConverged;
      }
      iteration++;

      odeint::integrate_const(stepper, system, x, 0.0, Dt, Dt);
      system(x, dx, 0);

      recorder_regist(x, dx);
    }

    return EvalStatus::Ok;
  }

  real_t result() const { return _R; }

 private:
  bool recorder_check() {
    const auto R_new = _R = avg_new.value();
    const auto R_old = avg_old.value();
    return std::abs(R_new - R_old) < epsilon;
  }

  void recorder_regist(const state_t& x, const state_t& dx) {
    avg_old.push(avg_new.first());
    avg_new.push(x, dx);
  }

 private:
  const decltype(odeint::make_controlled<odeint::runge_kutta_dopri5<state_t>>(
      0.0001, 0.0001)) stepper;

  real_t _R = -1;
  Order avg_new, avg_old;

  state_t x;
  state_t dx;
};

template <typename System>
class Kuramoto {
 public:
  using system_t = System;
  using real_t = typename System::real_t;
  using state_t = typename System::state_t;
  static constexpr int ndim = System::ndim;
  static constexpr const char* name = "kuramoto";

  struct UnitRef {
    const real_t cos;
    const real_t sin;
  };

 public:
  Kuramoto(int window) : cos_q(window, 0.), sin_q(window, 0.) {}

  void push(const state_t& x, const state_t& dx) noexcept {
    real_t cos_mean = 0;
    for (int i = 0; i < ndim; i++) {
      cos_mean += std::cos(x[i]);
    }
    cos_mean /= ndim;

    real_t sin_mean = 0;
    for (int i = 0; i < ndim; i++) {
      sin_mean += std::sin(x[i]);
    }
    sin_mean /= ndim;

    push({cos_mean, sin_mean});
  }

  void push(const UnitRef inner) noexcept {
    cos_q.push(inner.cos);
    sin_q.push(inner.sin);
  }

  UnitRef first() noexcept { return {cos_q.first(), sin_q.first()}; }

  real_t value() const noexcept {
    const auto cos_mean = cos_q.mean();
    const auto sin_mean = sin_q.mean();
    const auto R_new = std::sqrt(cos_mean * cos_mean + sin_mean * sin_mean);
    return R_new;
  }

 private:
  collection::FixedQueue<real_t> cos_q;
  collection::FixedQueue<real_t> sin_q;
};

template <typename System>
class MaxAvgFreqCluster {
 public:
  using system_t = System;
  using real_t = typename System::real_t;
  using state_t = typename System::state_t;
  static constexpr int ndim = System::ndim;
  static constexpr const char* name = "max_avg_freq_cluster";

  struct UnitRef {
    const std::valarray<real_t>& freq;
  };

 public:
  MaxAvgFreqCluster(int window) : freq_q(window) {}

  void push(const state_t& x, const state_t& dx) noexcept {
    return freq_q.push(&dx[0]);
  }

  void push(const UnitRef inner) noexcept { freq_q.push(&inner.freq[0]); }

  UnitRef first() { return {freq_q.first()}; }

  real_t value() const noexcept {
    static const real_t eps = 1e-2;
    auto freq_means = std::valarray(freq_q.sum() / freq_q.size());

    std::sort(std::begin(freq_means), std::end(freq_means));
    real_t r = 1, c = 1, prev = std::numeric_limits<real_t>::lowest();
    for (auto freq : freq_means) {
      if (freq - prev < eps) {
        r = std::max(r, ++c);
      } else {
        c = 1;
      }
      prev = freq;
    }
    r = (r - 1) / (ndim - 1);

    return r;
  }

 private:
  collection::FixedQueueVec<real_t, ndim> freq_q;
};
}  // namespace order
}  // namespace new_lib