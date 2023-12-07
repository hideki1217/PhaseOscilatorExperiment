#include <_common.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <new_lib.hpp>
#include <vector>

using namespace new_lib;

template <int ndim>
void experiment(int m = 500) {
  auto evaluator = new_lib::order::Evaluator<
      new_lib::order::Kuramoto<new_lib::system::System<ndim>>>(3000, 1e-4, 1,
                                                               10000);
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
  experiment<2>();
  experiment<3>();
  experiment<4>();
  experiment<5>();
  experiment<6>();
  experiment<7>();
  experiment<8>();
  experiment<9>();
  experiment<10>();
  experiment<11>();
  experiment<12>();
  experiment<13>();
  experiment<14>();
  experiment<15>();
  experiment<16>();
}