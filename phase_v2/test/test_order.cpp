#include <cassert>
#include <cstdlib>
#include <iostream>
#include <opy.hpp>
#include <vector>

void _assert_eq(double actual, double respect, double acc,
                const char* str_actual, const char* str_respect,
                const char* file, const int line) {
  if (std::abs(actual - respect) > acc) {
    std::cerr << file << ":" << line << ": |" << str_actual << " - "
              << str_respect << "| < " << acc << std::endl;
    std::cerr << "    Actual  : " << actual << std::endl;
    std::cerr << "    Respect : " << respect << std::endl;
    std::exit(1);
  }
}

#define assert_nearly_eq(actual, respect, acc) \
  _assert_eq(actual, respect, acc, #actual, #respect, __FILE__, __LINE__)

double _test_2d(lib::OrderEvaluator<double>& model_2d, double k, double w0) {
  std::vector<double> K = {0, k, k, 0};
  std::vector<double> w = {-w0, w0};
  auto status = model_2d.eval(&K[0], &w[0]);
  assert(status == lib::EvalStatus::Ok);
  return model_2d.result();
}

int main() {
  {
    auto model_2d = lib::OrderEvaluator<double>(30000, 1e-4, 0.01, int(1e6), 2);
    auto theoritical_2d = [](const double K, const double w) -> double {
      if (K >= w) {
        return std::cos(0.5 * std::asin(w / K));
      }
      return 0;
    };
#define test_2d(K) \
  assert_nearly_eq(_test_2d(model_2d, K, 1), theoritical_2d(K, 1), 1e-2);

    test_2d(0);
    test_2d(0.1);
    test_2d(0.5);
    test_2d(0.8);
    test_2d(1.2);
    test_2d(3.);
    test_2d(5);

#undef test_2d
  }
}