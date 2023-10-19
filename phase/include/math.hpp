#pragma once

#include <cassert>
#include <cmath>

template <typename T>
int solve_symmetric(int n, const T* A, const T* b, T* x) {
  T d[n];
  T L[n * n];
  T reg;

  for (int i = 0; i < n; i++) {
    // L[i, j]
    for (int j = 0; j < i; j++) {
      reg = A[i * n + j];
      for (int k = 0; k < j; k++) reg -= d[k] * L[i * n + k] * L[j * n + k];
      reg /= d[j];

      L[i * n + j] = L[j * n + i] = reg;
    }

    // d[i]
    reg = A[i * n + i];
    for (int k = 0; k < i; k++) reg -= d[k] * L[i * n + k] * L[i * n + k];
    d[i] = reg;

    assert(d[i] != 0.0);
  }

  // L x = b
  for (int i = 0; i < n; i++) {
    reg = b[i];
    for (int j = 0; j < i; j++) {
      reg -= L[i * n + j] * x[j];
    }
    x[i] = reg;
  }

  // diag(d) L^T x_new = x
  for (int i = n - 1; i >= 0; i--) {
    reg = x[i] / d[i];
    for (int j = i + 1; j < n; j++) {
      reg -= L[i * n + j] * x[j];
    }
    x[i] = reg;
  }

  return 0;
}

template <typename T, typename F, typename DF>
int newton_symmetric(int n, F f, DF Df, T* x0, const T eps = 1e-6,
                     const int maxiter = 50) {
  T _r[n];
  T _f[n];
  T _Df[n * n];

  int count = 0;
  bool conv;
  do {
    f(n, x0, _f);
    Df(n, x0, _Df);
    solve_symmetric(n, _Df, _f, _r);
    for (int i = 0; i < n; i++) x0[i] -= _r[i];

    conv = true;
    for (int i = 0; i < n; i++) {
      conv &= (std::abs(_r[i]) < eps);
    }
  } while (!conv && (count++) < maxiter);

  return count;
}