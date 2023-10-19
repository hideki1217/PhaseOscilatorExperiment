// #include <immintrin.h>
#include <omp.h>

// #define NDEBUG
// #define SKIP_BURNIN

#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mcmc.hpp>
#include <phase.hpp>
#include <phase.param.hpp>
#include <random>
#include <sstream>
#include <utils.hpp>
#include <vector>

class BolzmanMP {
 public:
  using State = std::vector<double>;
#if defined(NEWTON)
  using Score = PhaseNewton;
#else
  using Score = ConvMean<S>;
#endif
  const int dim;

  class Energy {
   public:
    const int dim;
    const double threshold;
    Score &score;

    Energy(int dim, double threshold, Score &score)
        : dim(dim), threshold(threshold), score(score) {}

    double operator()(const std::vector<double> &K) const {
      const double inf = 1e10;
      return check_constraint(K) ? unsafe_energy(K) : inf;
    }

    double unsafe_energy(const std::vector<double> &K) const {
      double sum = 0;
      for (auto k : K) sum += k;
      return sum / dim;
    }

    bool check_constraint(const std::vector<double> &K) const {
      for (auto k : K) {
        if (k < 0) return false;
      }
      if (score(K) < threshold) return false;
      return true;
    }
  } H;

  BolzmanMP(int dim, Energy param, double beta, Rng &rng,
            double step_size = 1.0)
      : dim(dim),
        H(param),
        _beta(beta),
        _sq_dim(int(std::sqrt(dim))),
        rng(rng),
        unifr(std::uniform_real_distribution<>(0, 1)),
        norm(std::normal_distribution<>(0, step_size)) {
    assert(beta > 0);
    State s_(dim, 0);
    set_state(s_);

    for (int i = 0; i < _sq_dim; i++) {
      for (int j = 0; j < _sq_dim; j++) {
        if (i == j) continue;

        idx_table.push_back(i * _sq_dim + j);
      }
    }
    unif = std::uniform_int_distribution<>(0, idx_table.size() - 1);
  }

  void set_state(std::vector<double> &s_) { s = s_; }
  const std::vector<double> &state() const { return s; }

  void reset_statistics() {
    num_update = 0;
    num_accept = 0;
  }
  double accept_rate() { return double(num_accept) / num_update; }

  void swap(BolzmanMP &rhs) {
    assert(dim == rhs.dim);
    rhs.s.swap(s);
  }

  double E() const { return H.unsafe_energy(s); }
  double beta() const { return _beta; }

  State &update() {
    num_update++;

    auto idx = idx_table[unif(rng)];
    // auto xdi = _sq_dim * (idx % _sq_dim) + (idx / _sq_dim);

    auto mem = s[idx];
    auto delta = norm(rng);
    s[idx] += delta;
    // s[xdi] += delta;

    // depend on Energy function
    // H(K) = (\forall i, j K_{ij} >=0 \land R(K) >= R^*) \frac{1}{N} \sum_{i, j
    // = 0}^{N-1} K_{ij} ? \infty dH = H(K+dK) - H(K) = (both fulfilled) ?
    // \frac{1}{N} \sum_{ij} dK_{ij} : \infty
    bool accept = false;
    if (s[idx] >= 0 && H.score(s) >= H.threshold) {
      double dE = delta / H.dim;  // H(s_new) - H(s_old)
      if (dE <= 0 || std::exp(-_beta * dE) >= unifr(rng)) {
        accept = true;
      }
    }

    if (accept) {
      num_accept++;
    } else {
      s[idx] = mem;
      //   s[xdi] = mem;
    }

    return s;
  }

  State &update(int T) {
    assert(T > 0);
    for (int i = 0; i < T - 1; i++) {
      update();
    }
    return update();
  }

 private:
  Rng &rng;
  std::uniform_int_distribution<> unif;
  std::uniform_real_distribution<> unifr;
  std::normal_distribution<> norm;

  double _beta;
  State s;
  int _sq_dim;
  std::vector<int> idx_table;
  int num_accept = 0;
  int num_update = 0;
};

template <typename Rng>
std::vector<double> phase_unif(int n, Rng &rng) {
  std::uniform_real_distribution<> unif(0, 2 * Pi);

  std::vector<double> res(n);
  for (int i = 0; i < n; i++) res[i] = unif(rng);
  return res;
}

void run(std::string &base) {
  const auto R = betas.size();
  Rng rng(42);
  std::vector<Rng> rngs;
  for (int i = 0; i < R; i++) rngs.push_back(Rng(rng()));

  // 位相振動子
  const int D_model = w0.size();
  auto s0 = phase_unif(D_model, rng);
#if defined(NEWTON)
  auto dynamics = std::vector(R, PhaseNewton(w0, s0, 1e-7));
#else
  auto dynamics = std::vector(R, ConvMean<S>(PhaseRK4(w0, s0), converge_window,
                                             converge_eps, converge_limit));
#endif

  std::vector<BolzmanMP> reprica;
  {
    // 絶対同期する結合を初期値に
    std::vector<double> K0(D_model * D_model, 10);
    for (int i = 0; i < D_model; i++) K0[i * D_model + i] = 0;

    for (int i = 0; i < R; i++) {
      auto m = BolzmanMP(D_model * D_model,
                         BolzmanMP::Energy(D_model, threshold, dynamics[i]),
                         betas[i], rngs[i], step_size[i]);
      m.set_state(K0);

      reprica.push_back(std::move(m));
    }
  }
  const int D_state = reprica[0].dim;

  auto swapper = Swapper(rng);
  std::uniform_int_distribution<> random_swap_idx(0, R - 2);
  std::vector<int> swapped(R);
  for (int i = 0; i < R; i++) swapped[i] = i;
  auto random_swap = [&](bool sync = true) {
    static int c = 0;
    for (int target = (c++) % 2; target + 1 < R; target += 2) {
      if (swapper.try_swap(reprica[target], reprica[target + 1])) {
        if (sync) std::swap(swapped[target], swapped[target + 1]);
      }
    }
  };

  // hotin
#ifndef SKIP_BURNIN
  for (int i = 0; i < burn_in / T_sampling; i++) {
#pragma omp parallel for
    for (int j = 0; j < R; j++) {
      reprica[j].update(T_sampling);
    }
    if (i % T_swap == 0) random_swap(false);
  }
#endif

  Csv csv((base + "phase.csv").c_str());
  std::vector<int> swap_history(R * N_sample / T_swap);
  std::vector<double> Es(R * N_sample);
  std::vector<double> state(R * D_state);
  for (int i = 0; i < N_sample; i++) {
#pragma omp parallel for
    for (int j = 0; j < R; j++) {
      auto s = reprica[j].update(T_sampling);

      Es[i * R + j] = reprica[j].E();
      for (int k = 0; k < D_state; k++) state[j * D_state + k] = s[k];
    }

    if (i % T_swap == 0) {
      random_swap(true);

      for (int j = 0; j < R; j++) {
        swap_history[(i / T_swap) * R + j] = swapped[j];
      }
    }

    auto row = csv.new_row();
    for (auto s : state) row.content(s);
  }
  csv.close();

  csv.save_mtx((base + "phase_E.csv").c_str(), N_sample, R, Es.data());
  csv.save_mtx((base + "phase_swap.csv").c_str(), N_sample / T_swap, R,
               swap_history.data());

  csv.open((base + "reprica_stat.csv").c_str());
  {
    auto row = csv.new_row();
    for (auto r : reprica) row.content(r.beta());
  }
  {
    auto row = csv.new_row();
    for (auto r : reprica) row.content(r.accept_rate());
  }
#if defined(NEWTON)
#else
  {
    auto row = csv.new_row();
    for (auto r : reprica) row.content(r.H.score.converge_failure_rate());
  }
#endif
  csv.close();

  std::ofstream ofs("./phase.out");
  ofs << base << std::endl;
}

int main(int argc, char **argv) {
  assert(argc == 2);

  std::string base;
  {
    auto now = std::chrono::system_clock::now();
    std::time_t stamp = std::chrono::system_clock::to_time_t(now);
    const std::tm *lt = std::localtime(&stamp);
    std::stringstream ss;
    ss << argv[1] << "/output/asym/" << std::put_time(lt, "%Y-%m-%d %H:%M:%S")
       << "/";
    base = ss.str();
    mkdir(base.c_str(), 0777);
  }
  try {
    save_param((base + "phase_param.yaml").c_str());
    run(base);
  } catch (...) {
    rmdir(base.c_str());
  }
}
