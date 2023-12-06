#include <_assert.hpp>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <new_lib.hpp>
#include <vector>

using namespace new_lib;

template <typename Evaluator>
double _test_2d(Evaluator& model_2d, double k, double w0) {
  std::vector<double> K = {0, k, k, 0};
  std::vector<double> w = {-w0, w0};
  auto status = model_2d.eval(&K[0], 2, &w[0]);
  assert(status == order::EvalStatus::Ok);
  return model_2d.result();
}

int main() {
  {
    auto model_2d = order::Evaluator<order::Kuramoto<system::System<2>>>(
        3000, 1e-4, 1, 10000);
    auto theoritical_2d = [](double K, double w) -> double {
      return (K >= w) ? std::cos(0.5 * std::asin(w / K)) : 0;
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
  {
    auto model_2d =
        order::Evaluator<order::MaxAvgFreqCluster<system::System<2>>>(
            3000, 1e-4, 1, 10000);
    auto theoritical_2d = [](double K, double w) -> double {
      return (K >= w) ? 1 : 0;
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