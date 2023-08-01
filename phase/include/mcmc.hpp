#pragma once

#include <utils.hpp>

class Swapper {
 public:
  Swapper(Rng &rng) : rng(rng), prob(std::uniform_real_distribution<>(0, 1)) {}

  template <typename Bolzman>
  bool try_swap(Bolzman &lhs, Bolzman &rhs) {
    assert(lhs.dim == rhs.dim);

    auto r = std::exp((lhs.beta() - rhs.beta()) * (lhs.E() - rhs.E()));
    if (r >= 1 || r >= prob(rng)) {
      lhs.swap(rhs);
      return true;
    }
    return false;
  }

 private:
  Rng &rng;
  std::uniform_real_distribution<> prob;
};