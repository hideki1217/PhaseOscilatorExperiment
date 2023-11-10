#include <iostream>
#include <mcmc.hpp>
#include <order.hpp>

int main() {
  double w[] = {-1, 1};
  double K[] = {0, 3, 3, 0};
  auto markov = lib::BolzmanMarkovChain<lib::order::Kuramoto<double>>(
      2, w, K, 0.99, 1.0, 1.0, 42);

  markov.step(100);  // Burn-In

  for (int i = 0; i < 100; i++) {
    markov.step(100);
    std::cout << markov.energy() << std::endl;
  }
}