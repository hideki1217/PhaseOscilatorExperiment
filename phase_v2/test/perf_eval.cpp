#include <_common.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <new_lib.hpp>
#include <vector>

using namespace new_lib;

template <typename target_t>
void experiment(int m = 500) {
  constexpr int ndim = target_t::ndim;
  auto evaluator = new_lib::order::Evaluator<target_t>(3000, 1e-4, 1, 10000);
  std::vector<double> K(ndim * ndim, 0.9);
  for (int i = 0; i < ndim; i++) K[i * ndim + i] = 0;
  std::vector<double> w(ndim);
  new_lib::param::create_w(ndim, &w[0]);

  std::valarray<double> ts_ms(m);
  for (int i = 0; i < m; i++) {
    const auto start = std::chrono::system_clock::now();

    { evaluator.eval(K.data(), ndim, w.data()); }

    ts_ms[i] = std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now() - start)
                   .count();
  }
  const double m_time_ms = ts_ms.sum() / ts_ms.size();
  const double std_time_ms = std::sqrt(
      (ts_ms - m_time_ms).apply([](double x) { return x * x; }).sum() /
      ts_ms.size());

  std::printf("ndim = %03d: %.2f ± %.4f (ms)\n", ndim, m_time_ms, std_time_ms);
}

int main() {
  experiment<order::MaxAvgFreqCluster<system::System<2>>>();
  experiment<order::MaxAvgFreqCluster<system::System<3>>>();
  experiment<order::MaxAvgFreqCluster<system::System<4>>>();
  experiment<order::MaxAvgFreqCluster<system::System<5>>>();
  experiment<order::MaxAvgFreqCluster<system::System<6>>>();
  experiment<order::MaxAvgFreqCluster<system::System<7>>>();
  experiment<order::MaxAvgFreqCluster<system::System<8>>>();

  experiment<order::Kuramoto<system::System<2>>>();
  experiment<order::Kuramoto<system::System<3>>>();
  experiment<order::Kuramoto<system::System<4>>>();
  experiment<order::Kuramoto<system::System<5>>>();
  experiment<order::Kuramoto<system::System<6>>>();
  experiment<order::Kuramoto<system::System<7>>>();
  experiment<order::Kuramoto<system::System<8>>>();
  // experiment<order::Kuramoto<system::System<9>>>();
  // experiment<order::Kuramoto<system::System<10>>>();
  // experiment<order::Kuramoto<system::System<11>>>();
  // experiment<order::Kuramoto<system::System<12>>>();
  // experiment<order::Kuramoto<system::System<13>>>();
  // experiment<order::Kuramoto<system::System<14>>>();
  // experiment<order::Kuramoto<system::System<15>>>();
  // experiment<order::Kuramoto<system::System<16>>>();
}