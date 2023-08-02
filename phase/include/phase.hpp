#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <math.hpp>
#include <vector>

static double _phase_order(int dim, const double *s) {
  auto x = 0.0;
  auto y = 0.0;
  for (int i = 0; i < dim; i++) {
    x += std::cos(s[i]);
    y += std::sin(s[i]);
  }
  x /= dim;
  y /= dim;
  return std::sqrt(x * x + y * y);
}

static double _freq_order(int dim, const double *w, double eps = 0.1) {
  auto c = 0;
  for (int i = 0; i < dim; i++) {
    c += (std::abs(w[i]) < eps);
  }
  return double(c) / dim;
}

static void F(int dim, const double *s, const double *w0, const double *K,
              double *res) {
  for (int i = 0; i < dim; i++) res[i] = w0[i];

  for (int j = 0; j < dim; j++) {
    auto s_j = s[j];
    auto K_j = &K[j * dim];
    for (int i = 0; i < dim; i++) {
      res[i] -= K_j[i] * std::sin(s[i] - s_j);
    }
  }
}

static void DF(int n, const double *s, const double *w0, const double *K,
               double *res) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      res[i * n + j] = (i != j) ? K[i * n + j] * std::cos(s[j] - s[i]) : 0;
    }
  }

  for (int i = 0; i < n; i++) {
    for (int k = 0; k < n; k++) {
      res[i * n + i] -= K[i * n + k] * std::cos(s[k] - s[i]);
    }
  }

  // make it reguler
  for (int i = 0; i < n; i++) {
    res[i * n + i] += 1e-12;
  }
}

class PhaseNewton {
 public:
  const int dim;
  const int maxitr;
  const double eps;
  // const Score score;

  PhaseNewton(const std::vector<double> &w0, const std::vector<double> &s0,
              double eps, int maxitr = 50)
      : w0(w0), s(s0), maxitr(maxitr), eps(eps), dim(w0.size()) {}

  void set_state(const std::vector<double> &s) { this->s = s; }

  void step(const std::vector<double> &K) {
    std::fill(s.begin(), s.end(), 0);
    auto f = [&](int dim, const double *x, double *res) {
      return F(dim, x, &w0[0], &K[0], res);
    };
    auto Df = [&](int dim, const double *x, double *res) {
      return DF(dim, x, &w0[0], &K[0], res);
    };
    auto iter = newton_symmetric(dim, f, Df, &s[0], eps, maxitr);
  }

  double operator()(const std::vector<double> &K) {
    step(K);
    return _phase_order(dim, &s[0]);  // TODO: Hardcode
  }

 private:
  std::vector<double> w0;
  std::vector<double> s;
};

class PhaseRK4 {
 public:
  const int dim;
  const double dt;

  PhaseRK4(const std::vector<double> &w0, const std::vector<double> &s0,
           double dt = 0.01)
      : dim(w0.size()), w0(w0), s(s0), dt(dt) {
    assert(w0.size() == s0.size());

    w.resize(dim);
    k1.resize(dim);
    k2.resize(dim);
    k3.resize(dim);
    _s.resize(dim);
  }

  void set_state(const std::vector<double> &s) { this->s = s; }

  void step(const std::vector<double> &K) {
    assert(K.size() == dim * dim);

    const auto dt2 = dt / 2;
    const auto dt6 = dt / 6;
    // 四次のルンゲクッタ法
    F(dim, s.data(), w0.data(), K.data(), w.data());
    for (int i = 0; i < dim; i++) _s[i] = s[i] + w[i] * dt2;
    F(dim, _s.data(), w0.data(), K.data(), k1.data());
    for (int i = 0; i < dim; i++) _s[i] = s[i] + k1[i] * dt2;
    F(dim, _s.data(), w0.data(), K.data(), k2.data());
    for (int i = 0; i < dim; i++) _s[i] = s[i] + k2[i] * dt;
    F(dim, _s.data(), w0.data(), K.data(), k3.data());
    for (int i = 0; i < dim; i++) k3[i] += w[i] + 2 * k1[i] + 2 * k2[i];
    for (int i = 0; i < dim; i++) s[i] += k3[i] * dt6;
  }

  double phase_order() const { return _phase_order(dim, &s[0]); }
  double freq_order(double eps = 0.1) const { return _freq_order(dim, &w[0]); }

 private:
  std::vector<double> w0;
  std::vector<double> s;

  std::vector<double> w;
  std::vector<double> k1;
  std::vector<double> k2;
  std::vector<double> k3;
  std::vector<double> _s;
};

class PScore {
 public:
  inline double operator()(const PhaseRK4 &model) {
    return model.phase_order();
  }
};

class FScore {
 public:
  inline double operator()(const PhaseRK4 &model) { return model.freq_order(); }
};

template <typename Score = PScore>
class SkipMean {
 private:
  Score score;

 public:
  PhaseRK4 model;
  const int steps_burnin;
  const int steps_eval;

  SkipMean(PhaseRK4 model, int steps_burnin, int steps_eval)
      : model(model), steps_burnin(steps_burnin), steps_eval(steps_eval) {}

  double operator()(const std::vector<double> &K) {
    for (int i = 0; i < steps_burnin; i++) {
      model.step(K);
    }
    double m = 0;
    for (int i = 0; i < steps_eval; i++) {
      model.step(K);
      m += score(model);  // O(1)
    }
    return m / steps_eval;
  }
};

template <typename Score>
class ConvMean {
 private:
  Score score;
  std::vector<double> window;
  double ms[2] = {0, 0};
  int c = 0;

  int reset_count = 0;
  int converge_failure = 0;

 public:
  PhaseRK4 model;
  const double eps;
  const int window_size;
  const int limit;

  ConvMean(PhaseRK4 model, int window, double eps, int limit)
      : model(model), eps(eps), window_size(window), limit(limit) {
    this->window.resize(window * 2, 0.0);
  }

  int loop_count() { return c; }
  double converge_failure_rate() {
    return double(converge_failure) / reset_count;
  }

  void reset() {
    reset_count++;

    c = 0;
    ms[0] = ms[1] = 0;
    std::fill(window.begin(), window.end(), 0);
  }

  bool check(double v) {
    int first = c % window.size();
    int second = (c + window_size) % window.size();

    ms[0] += window[second] - window[first];
    ms[1] += v - window[second];

    window[first] = v;
    c++;
    if (c >= limit) {
      converge_failure++;
      return true;
    }
    return (c <= window.size() + 1)
               ? false
               : std::abs(ms[0] - ms[1]) < eps * window_size;
  }

  double value() { return ms[1] / (window_size); }

  double operator()(const std::vector<double> &K) {
    double res;
    reset();
    do {
      model.step(K);
      res = score(model);
    } while (!check(res));
    return value();
  }
};
