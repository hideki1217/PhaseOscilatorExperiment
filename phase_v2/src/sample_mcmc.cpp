#include <_common.hpp>
#include <_concurrent.hpp>
#include <iostream>
#include <mcmc.hpp>
#include <order.hpp>
#include <random>

using namespace lib;

template <typename Order>
struct Param {
  using target_t = Order;

  const int ndim;
  const double* w;
  const double* K;
  const double threshold;

  const int num_reprica;
  const double* betas;
  const double* scales;

  const int burn_in;
  const int T;
  const int swap;

  const int seed = 35;
};

struct Result {
  std::vector<double> exchange_rates;
};

lib::concurrent::ThreadPool thread_pool(8);

template <typename target_t>
Result reprica_exchange(const Param<target_t> p) {
  std::mt19937 rng(p.seed);

  std::vector<int> count_total(p.num_reprica - 1, 0);
  std::vector<int> count_accepted(p.num_reprica - 1, 0);

  std::vector<mcmc::BolzmanMarkovChain<target_t>> markovs;
  for (int r = 0; r < p.num_reprica; r++) {
    markovs.emplace_back(mcmc::BolzmanMarkovChain<target_t>(
        p.ndim, p.w, p.K, p.threshold, p.betas[r], p.scales[r], rng()));
  }

  // Burn-in
  for (int r = 0; r < p.num_reprica; r++) {
    thread_pool.post([&markovs, &p, r]() {
      for (int i = 0; i < p.burn_in; i++) markovs[r].step();
    });
  }
  thread_pool.join();

  for (int e = 0; e < 1000; e++) {
    for (int r = 0; r < p.num_reprica; r++) {
      thread_pool.post([&markovs, &p, r]() {
        for (int i = 0; i < p.T * p.swap; i++) markovs[r].step();
      });
    }
    thread_pool.join();

    eprintf("%d's try_swap", e + 1);
    for (int i = e % 2; i < p.num_reprica - 1; i += 2) {
      count_total[i]++;
      if (markovs[i].try_swap(markovs[i + 1])) {
        count_accepted[i]++;
      }
    }
  }

  std::vector<double> exchange_rates(p.num_reprica);
  for (int r = 0; r < p.num_reprica; r++) {
    exchange_rates[r] = double(count_accepted[r]) / count_total[r];
  }
  return {exchange_rates};
}

int main() {
  using target_t = order::Kuramoto<double>;
  const int ndim = 2;
  const double w[] = {-1, 1};
  const double K[] = {0, 5, 5, 0};
  const double threshold = 0.78;

  const double min_beta = 1.0;
  const double max_beta = 100;
  const double target_accept_rate = 0.5;
  const double target_exchange_rate = 0.5;
  const int burn_in = 100;

  double scale;
  {
    std::cout << "==== Measure scale ==== : target_accept_rate = "
              << target_accept_rate << std::endl;

    double scale_l = 1e-10;
    double scale_r = 4.0;
    for (int i = 0; i < 10; i++) {
      scale = (scale_r + scale_l) / 2;

      auto stat = utils::Statistics(mcmc::Result::LENGTH);
      auto markov = mcmc::BolzmanMarkovChain<target_t>(ndim, w, K, threshold,
                                                       min_beta, scale, 42);
      for (int i = 0; i < burn_in; i++) markov.step();  // Burn-In

      for (int i = 0; i < 1000; i++) stat.push(markov.step());

      const auto accept_rate = stat.rate(mcmc::Result::Accepted);
      if (target_accept_rate < accept_rate) {
        scale_l = scale;
      } else {
        scale_r = scale;
      }
    }

    std::cout << "beta = " << min_beta << ", scale = " << scale << std::endl;
  }

  double beta = min_beta;
  Result result;
  std::vector<double> betas({min_beta}), scales({scale});
  {
    std::cout << "==== Measure Beta series ==== : target_exchange_rate = "
              << target_exchange_rate << std::endl;

    int index = 0;
    std::cout << index + 1 << ": " << betas[index] << std::endl;
    while (beta < max_beta) {
      // 高温から順に、二分探索で交換率 target_exchange_rate を目指す
      // 決まったら、それをbetasにpushして max_beta に到達するまで続ける
      double beta_l = betas[betas.size() - 1];
      double beta_r = max_beta * 2;
      for (int i = 0; i < 10; i++) {
        beta = (beta_r + beta_l) / 2;

        betas.push_back(beta);
        scales.push_back(min_beta / beta * scale);

        // betasでレプリカ交換法
        Param<target_t> param = {ndim,
                                 w,
                                 K,
                                 threshold,
                                 static_cast<int>(betas.size()),
                                 &betas[0],
                                 &scales[0],
                                 burn_in,
                                 10,
                                 10};
        result = reprica_exchange(param);

        // 最後の二つの 交換成立回数/交換試行回数 を計測し、結果を見る
        const double exchange_rate = result.exchange_rates[betas.size() - 2];
        if (exchange_rate < target_exchange_rate) {
          beta_r = beta;
        } else {
          beta_l = beta;
        }

        betas.pop_back();
        scales.pop_back();
      }

      betas.push_back(beta);
      scales.push_back(min_beta / beta * scale);
      index++;
      std::cout << index + 1 << ": " << betas[index] << std::endl;
      for (int i = 0; i < betas.size() - 1; i++) {
        std::cout << i << "<->" << i + 1 << " = " << result.exchange_rates[i]
                  << std::endl;
      }
    }
  }

  std::cout << "==== Total result ====" << std::endl;
  std::cout << "const auto base_scale = " << scale << ";" << std::endl;
  std::cout << "const auto betas[] = {";
  for (auto beta : betas) {
    std::cout << beta << ",";
  }
  std::cout << "};" << std::endl;
  std::cout << "const auto scales[] = {";
  for (auto scale : scales) {
    std::cout << scale << ",";
  }
  std::cout << "};" << std::endl;
}