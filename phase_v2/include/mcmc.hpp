#include <_concurrent.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <opy.hpp>
#include <random>

namespace lib {

namespace utils {
/**
 *  NOTE: Not fast
 */
template <typename T>
class UniformSelector {
 public:
  std::vector<T> items;
  UniformSelector() {}
  void add_item(T item) { items.emplace_back(item); }

  template <typename Rng>
  const T &select(Rng &rng) const {
    std::uniform_int_distribution<> dist(0, items.size() - 1);
    return items[dist(rng)];
  }
};

class Statistics {
 public:
  Statistics(int num_class) : counts(num_class, 0) {}

  void push(int x) {
    assert(x < counts.size());
    counts[x]++;
    total++;
  }

  double rate(int x) {
    assert(x < counts.size());
    return double(counts[x]) / total;
  }

  void reset() {
    total = 0;
    std::fill(counts.begin(), counts.end(), 0);
  }

 private:
  uint total = 0;
  std::vector<uint> counts;
};
}  // namespace utils

namespace mcmc {
enum Result {
  Accepted = 0,
  Rejected,
  MinusConnection,
  SmallOrder,
  NotConverged,
  LENGTH
};
const char *result2str(Result result) {
  static const char *nameTT[] = {"Accepted", "Rejected", "MinusConnection",
                                 "SmallOrder", "NotConverged"};
  return nameTT[result];
}

template <typename Order>
class BolzmanMarkovChain {
  using Real = typename Order::V;

 public:
  const int ndim;
  const Real threshold;
  const Real beta;
  const Real scale;

  BolzmanMarkovChain(int ndim, const Real *w, const Real *K, Real threshold,
                     Real beta, Real scale, int seed)
      : ndim(ndim),
        threshold(threshold),
        beta(beta),
        scale(scale),
        evaluator(3000, 1e-4, 1, 10000, ndim),  // NOTE: Default parameter
        w(new(std::align_val_t{64}) Real[ndim]),
        K(new(std::align_val_t{64}) Real[ndim * ndim]),
        _rng(seed) {
    for (int i = 0; i < ndim; i++) {
      for (int j = i + 1; j < ndim; j++) {
        _target_indexs.add_item(i * ndim + j);
      }
    }

    std::copy_n(w, ndim, &this->w[0]);
    std::copy_n(K, ndim * ndim, &this->K[0]);

    K_sum = 0;
    for (int i = 0; i < ndim * ndim; i++) K_sum += K[i];
  }

  const Real *connection() const { return &K[0]; }
  Real energy() const { return K_sum / ndim; }

  /**
   * reprica exchange
   */
  bool try_swap(BolzmanMarkovChain<Order> &rhs) {
    const Real rate = std::exp((beta - rhs.beta) * (energy() - rhs.energy()));
    if (1 <= rate || std::uniform_real_distribution<>(0, 1)(_rng) < rate) {
      K.swap(rhs.K);
      return true;
    }
    return false;
  }

  Result step() {
    const auto idx = _target_indexs.select(_rng);
    const auto xdi = (idx % ndim) * ndim + (idx / ndim);
    const auto prev_K_idx = K[idx];
    const auto prev_K_sum = K_sum;
    const auto d = std::normal_distribution<>(0, scale)(_rng);

    K[idx] += d;
    K[xdi] += d;
    K_sum += 2 * d;

    const auto result = try_step(idx, d);

    if (result != Result::Accepted) {
      K[idx] = K[xdi] = prev_K_idx;
      K_sum = prev_K_sum;
    }

    return result;
  }

 private:
  Result try_step(int updated_idx, Real updated_delta) {
    if (K[updated_idx] < 0) {
      return Result::MinusConnection;
    }

    const auto status = evaluator.eval(&K[0], &w[0]);
    if (status != EvalStatus::Ok) {
      return Result::NotConverged;
    }
    const auto R = evaluator.result();
    if (R < threshold) {
      return Result::SmallOrder;
    }

    // dE = E(t+1) - E(t)
    // E = K_sum / ndim
    const auto dE = (2 * updated_delta) / ndim;
    if (dE <= 0 ||
        std::uniform_real_distribution<>(0, 1)(_rng) <= std::exp(-beta * dE)) {
      return Result::Accepted;
    } else {
      return Result::Rejected;
    }
  }

  OrderEvaluatorRK45<Real, Order> evaluator;

  std::unique_ptr<Real[]> w;
  std::unique_ptr<Real[]> K;
  Real K_sum;

  utils::UniformSelector<int> _target_indexs;
  std::mt19937 _rng;
};

template <typename Order>
class RepricaMCMC {
  using Real = typename Order::V;

 public:
  const int ndim;
  const Real threshold;
  const int num_reprica;

  RepricaMCMC(int ndim, const Real *w, const Real *K, Real threshold,
              int num_reprica, const Real *betas, const Real *scales, int seed)
      : ndim(ndim), threshold(threshold), num_reprica(num_reprica) {
    for (int i = 0; i < num_reprica; i++) {
      mcmc_list.emplace_back(BolzmanMarkovChain<Order>(
          ndim, w, K, threshold, betas[i], scales[i], seed + i));
    }
  }

  /**
   * ATTENTION: not async, you create multiple threads and join them.
   */
  void step(concurrent::ThreadPool &pool, int n = 1) {
    for (int r = 0; r < num_reprica; r++) {
      pool.post([this, r, n]() {
        for (int i = 0; i < n; i++) mcmc_list[r].step();
      });
    }
    pool.join();
  }

  struct ExchangeResult {
    uint target;
    uint occured;
  };
  /**
   * if (ret & 1<<r) then mcmc[r] <=> mcmc[r+1] is occured.
   */
  ExchangeResult exchange() {
    assert(num_reprica < std::numeric_limits<uint>::digits);

    uint target = 0;
    uint occured = 0;
    for (int r = (c_exchange++ % 2); r + 1 < num_reprica; r += 2) {
      target |= uint(1) << r;
      if (mcmc_list[r].try_swap(mcmc_list[r + 1])) {
        occured |= (uint(1) << r);
      }
    }

    return {target, occured};
  }

  BolzmanMarkovChain<Order> &operator[](int index) {
    assert(0 <= index && index < num_reprica);
    return mcmc_list[index];
  }

 private:
  int c_exchange = 0;
  std::vector<BolzmanMarkovChain<Order>> mcmc_list;
};
}  // namespace mcmc

}  // namespace lib
