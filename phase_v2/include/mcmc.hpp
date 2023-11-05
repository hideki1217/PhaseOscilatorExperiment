#include <algorithm>
#include <cmath>
#include <opy.hpp>
#include <random>
#include <vector>

namespace lib {
using _Real = double;

template <typename T>
class UniformSelector {
 public:
  std::vector<T> items;
  UniformSelector() {}
  void add_item(T item) { items.emplace_back(item); }

  template <typename Rng>
  T &select(Rng &rng) const {
    std::uniform_int_distribution<> dist(item.size());
    return dist(rng);
  }
};

class BolzmanMCMC {
 public:
  const int ndim;
  const _Real threshold;
  const _Real beta;
  OrderEvaluatorRK45<_Real> evaluator;
  std::vector<_Real> K;
  _Real K_sum = 0;
  const std::vector<_Real> w;
  UniformSelector<int> _target_indexs;
  std::mt19937 _rng;

  BolzmanMCMC(int ndim, _Real threshold, const _Real *w, _Real beta,
              int seed = 42)
      : evaluator(30000, 1e-4, 0.1, 100000, ndim),
        threshold(threshold),
        ndim(ndim),
        K(ndim * ndim, 1),
        w(w, w + ndim),
        _rng(seed),
        beta(beta) {
    for (int i = 0; i < ndim; i++) {
      for (int j = i + 1; j < ndim; j++) {
        _target_indexs.add_item(i * ndim + j);
      }
    }

    std::fill(K.begin(), K.end(), 1);
    for (int i = 0; i < ndim; i++) K[i * ndim + i] = 0;
    for (auto k : K) {
      K_sum += k;
    }
  }

  void next() {
    const auto idx = _target_indexs.select(_rng);
    const auto xdi = (idx % ndim) * ndim + (idx / ndim);
    const auto prev_K_idx = K[idx];
    const auto prev_K_sum = K_sum;
    const auto d = std::normal_distribution<>(0, 1)(_rng);

    K[idx] += d;
    K[xdi] += d;
    K_sum += 2 * d;

    const auto result = _try_step(idx, d);

    if (!result) {
      K[idx] = K[xdi] = prev_K_idx;
      K_sum = prev_K_sum;
    }
  }

  bool _try_step(int updated_idx, _Real updated_delta) {
    if (K[updated_idx] < 0) {
      return false;
    }

    const auto status = evaluator.eval(&K[0], &w[0]);
    if (status != EvalStatus::Ok) {
      return false;
    }
    const auto R = evaluator.result();
    if (R < threshold) {
      return false;
    }

    const auto dE = (2 * updated_delta) / ndim;
    if (std::exp(-beta * dE) < std::uniform_real_distribution<>(0, 1)(_rng)) {
      return false;
    }
  }
};
}  // namespace lib
