#include <cassert>
#include <cmath>
#include <vector>

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

  double phase_order() const {
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

  double freq_order(double eps = 0.1) const {
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

class PScore {
public:
  inline double operator() (const PhaseRK4& model) { return model.phase_order(); }
};

class FScore {
public:
  inline double operator() (const PhaseRK4& model) { return model.freq_order(); }
};

template<typename Score=PScore>
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
      m += score(model); // O(1)
    }
    return m / steps_eval;
  }
};