#include <_common.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <opy.hpp>
#include <vector>

using namespace lib;

int main() {
  const int m = 500;
  for (int ndim = 2; ndim < 20; ndim++) {
    auto model = OrderEvaluatorRK45<double, order::Kuramoto<double>>(
        3000, 1e-4, 1, 10000, ndim);
    std::vector<double> K(ndim * ndim, 0.9);
    for (int i = 0; i < ndim; i++) K[i * ndim + i] = 0;
    std::vector<double> w(ndim);
    param::create_w(ndim, &w[0]);

    std::valarray<double> ts_ms(m);
    for (int i = 0; i < m; i++) {
      const auto start = std::chrono::system_clock::now();

      { model.eval(K.data(), w.data()); }

      ts_ms[i] = std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::system_clock::now() - start)
                     .count();
    }
    const double m_time_ms = ts_ms.sum() / ts_ms.size();
    const double std_time_ms = std::sqrt(
        (ts_ms - m_time_ms).apply([](double x) { return x * x; }).sum() /
        ts_ms.size());

    std::printf("ndim = %03d: %.2f ± %.4f (ms)\n", ndim, m_time_ms,
                std_time_ms);
  }
}