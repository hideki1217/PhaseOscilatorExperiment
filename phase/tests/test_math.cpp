
#include <algorithm>
#include <debug_math.hpp>
#include <iostream>
#include <math.hpp>
#include <random>
#include <vector>

void test_solve_symmetric(int n) {
  std::vector<double> A(n * n), b(n), x(n);
  {
    std::mt19937 rng(42);
    std::normal_distribution<> norm(0, 1);
    std::vector<double> buf;
    for (int i = 0; i < n * n + n; i++) buf.push_back(norm(rng));
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        A[i * n + j] = (i > j) ? buf[i * n + j] : buf[j * n + i];
      }
    }
    for (int i = 0; i < n; i++) b[i] = buf[n * n + i];
  }

  solve_symmetric(n, &A[0], &b[0], &x[0]);

  for (int i = 0; i < n; i++) {
    double reg = 0;
    for (int j = 0; j < n; j++) {
      reg += A[i * n + j] * x[j];
    }
    assert(std::abs(reg - b[i]) < 1e-6);
  }
}

int main() {
  test_solve_symmetric(10);
  test_solve_symmetric(50);
  test_solve_symmetric(100);
}
