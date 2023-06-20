#include <vector>
#include <cassert>
#include <cmath>

class PhaseRK4 {
 public:
  const int dim;
  const double dt;

  PhaseRK4(std::vector<double> &w0, std::vector<double> &s0, double dt = 0.01)
      : dim(w0.size()), w0(w0), s(s0), dt(dt) {
    assert(w0.size() == s0.size());

    w.resize(dim);
    k1.resize(dim);
    k2.resize(dim);
    k3.resize(dim);
    _s.resize(dim);
  }

  void step(const std::vector<double> &K) {
    assert(K.size() == dim * dim);

    const auto dt2 = dt / 2;
    const auto dt6 = dt / 6;
    // 四次のルンゲクッタ法
    velocity(dim, K.data(), w0.data(), s.data(), w.data());
    for (int i = 0; i < dim; i++) _s[i] = s[i] + w[i] * dt2;
    velocity(dim, K.data(), w0.data(), _s.data(), k1.data());
    for (int i = 0; i < dim; i++) _s[i] = s[i] + k1[i] * dt2;
    velocity(dim, K.data(), w0.data(), _s.data(), k2.data());
    for (int i = 0; i < dim; i++) _s[i] = s[i] + k2[i] * dt;
    velocity(dim, K.data(), w0.data(), _s.data(), k3.data());
    for (int i = 0; i < dim; i++) k3[i] += w[i] + 2 * k1[i] + 2 * k2[i];
    for (int i = 0; i < dim; i++) s[i] += k3[i] * dt6;
  }

  double phase_order() {
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

  double freq_order(double eps = 0.1) {
    auto c = 0;
    for (int i = 0; i < dim; i++) {
      c += (std::abs(w[i]) < eps);
    }
    return double(c) / dim;
  }

 private:
  void velocity(int dim, const double *K, const double *w0, const double *s,
                double *w) {
    // assume: K is symmetric
    assert(std::abs(K[1] - K[dim]) < 1e-8);

    for (int i = 0; i < dim; i++) w[i] = w0[i];

    for (int j = 0; j < dim; j++) {
      auto s_j = s[j];
      auto K_j = &K[j * dim];
      for (int i = 0; i < dim; i++) {
        w[i] -= K_j[i] * std::sin(s[i] - s_j);
      }
    }
  }

  std::vector<double> w0;
  std::vector<double> s;

  std::vector<double> w;
  std::vector<double> k1;
  std::vector<double> k2;
  std::vector<double> k3;
  std::vector<double> _s;
};

class SkipMean {
 public:
  PhaseRK4 model;
  SkipMean(PhaseRK4 model) : model(model) {}

  double operator()(const std::vector<double> &K) {
    // TODO: hardcode
    const static int N_skip = 3000;
    const static int N_mean = 1000;

    for (int i = 0; i < N_skip; i++) {
      model.step(K);
    }
    double m = 0;
    for (int i = 0; i < N_mean; i++) {
      model.step(K);
      m += model.phase_order();
    }
    return m / N_mean;
  }
};