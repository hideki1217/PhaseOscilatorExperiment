#include <chrono>
#include <opy.hpp>
#include <vector>

using namespace lib;

void _main(int m) {
  auto model_2d = OrderEvaluatorRK45<double, order::KuramotoFixed<double>>(
      30000, 1e-4, 0.1, int(1e6), 2);
  std::vector<double> K = {0, 0.9, 0.9, 0};
  std::vector<double> w = {-1, 1};

  for (int i = 0; i < m; i++) {
    model_2d.eval(K.data(), w.data());
  }
}

int main() {
  const int m = 500;

  const auto start = std::chrono::system_clock::now();
  _main(500);
  const auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::system_clock::now() - start)
                           .count();
  std::cout << double(time_ms) / m << "(ms)" << std::endl;
}