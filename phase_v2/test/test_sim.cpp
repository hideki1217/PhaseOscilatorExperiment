#include <chrono>
#include <iostream>
#include <order.hpp>
#include <sim.hpp>
#include <tool.hpp>

using namespace lib;
using Real = double;

void measure_time_2d(Real k) {
  const int ndim = 2;
  std::vector<Real> K = {0, k, k, 0};
  std::vector<Real> w = {-1, 1.};

  order::KuramotoFixed<Real> avg(10000, ndim);
  Real sampling_dt = 1;
  const int iteration = 30000;

  std::vector<Real> s(ndim, 0);
  std::vector<Real> ds_dt(ndim, 0);

  {
    Real t = 0;
    auto rk45 = sim::FehlbergRK45<Real>(ndim, 1e-2, 1e0, 1e-3);
    TIMESTAT({
      for (int e = 0; e < 30000; e++) {
        const auto result = rk45.advance(sampling_dt, t, &s[0], &K[0], &w[0]);
        t = result.t;
        sim::target_model(ndim, &K[0], &w[0], t, &s[0], &ds_dt[0]);

        avg.push(&s[0], &ds_dt[0]);
      }
      std::cout << avg.value() << std::endl;
    })
    TIMESTAT({ rk45.advance(iteration * sampling_dt, t, &s[0], &K[0], &w[0]); })
  }
  {
    Real t = 0;
    auto rk4 = sim::RK4<Real>(ndim, 1e-2);
    TIMESTAT({
      for (int e = 0; e < 30000; e++) {
        const auto result = rk4.advance(sampling_dt, t, &s[0], &K[0], &w[0]);
        t = result.t;
        sim::target_model(ndim, &K[0], &w[0], t, &s[0], &ds_dt[0]);

        avg.push(&s[0], &ds_dt[0]);
      }
      std::cout << avg.value() << std::endl;
    })
    TIMESTAT({ rk4.advance(iteration * sampling_dt, t, &s[0], &K[0], &w[0]); })
  }
}

void test_advance_2d(Real k, Real w0) {
  const int ndim = 2;
  std::vector<Real> K = {0, k, k, 0};
  std::vector<Real> w = {-w0, w0};

  Real sampling_dt = 1;
  const int iteration = 4000;
  auto avg = order::KuramotoFixed<Real>(2000, ndim);
  auto theoritical_2d = [](const double K, const double w) -> double {
    if (K >= w) {
      return std::cos(0.5 * std::asin(w / K));
    }
    return 0;
  };

  std::vector<Real> s(ndim, 0);
  std::vector<Real> ds_dt(ndim, 0);

  auto rk45 = sim::FehlbergRK45<Real>(ndim, 1e-2, 1e0, 1e-3);
  Real rk45_order;
  {
    Real t = 0;
    for (int e = 0; e < iteration; e++) {
      const auto result = rk45.advance(sampling_dt, t, &s[0], &K[0], &w[0]);
      t = result.t;
      sim::target_model(ndim, &K[0], &w[0], t, &s[0], &ds_dt[0]);

      avg.push(&s[0], &ds_dt[0]);
    }
    rk45_order = avg.value();
  }
  assert_nearly_eq(rk45_order, theoritical_2d(k, w0), 1e-2);

  auto rk4 = sim::RK4<Real>(ndim, 1e-2);
  Real rk4_order;
  {
    Real t = 0;
    for (int e = 0; e < iteration; e++) {
      const auto result = rk4.advance(sampling_dt, t, &s[0], &K[0], &w[0]);
      t = result.t;
      sim::target_model(ndim, &K[0], &w[0], t, &s[0], &ds_dt[0]);

      avg.push(&s[0], &ds_dt[0]);
    }
    rk4_order = avg.value();
  }
  assert_nearly_eq(rk4_order, theoritical_2d(k, w0), 1e-2);
}

int main() {
  // measure_time_2d(0.99);
  // measure_time_2d(1.01);

  test_advance_2d(0, 1);
  test_advance_2d(0.5, 1);
  test_advance_2d(0.99, 1);
  test_advance_2d(1.01, 1);
  test_advance_2d(1.5, 1);
  test_advance_2d(2, 1);
}