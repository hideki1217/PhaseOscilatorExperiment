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
#include <random>
#include <sstream>
#include <vector>

#include "phase.hpp"
#include "phase.param.hpp"
#include "utils.hpp"

static const double Pi = 3.14159265358979;

using Rng = std::mt19937;

template <typename Energy>
class SymBolzmanMP {
 public:
  using State = std::vector<double>;
  const int dim;
  Energy H;

  SymBolzmanMP(int dim, Energy energy, double beta, Rng &rng,
               double step_size = 1.0)
      : dim(dim),
        H(energy),
        _beta(beta),
        _sq_dim(int(std::sqrt(dim))),
        rng(rng),
        unif(std::uniform_int_distribution<>(
            0, (dim - int(std::sqrt(dim))) / 2 - 1)),
        unifr(std::uniform_real_distribution<>(0, 1)),
        norm(std::normal_distribution<>(0, step_size)) {
    assert(beta > 0);
    State s_(dim, 0);
    set_state(s_);

    // 対角成分のインデックスだけのテーブル
    for (int i = 0; i < _sq_dim; i++) {
      for (int j = i + 1; j < _sq_dim; j++) {
        idx_table.push_back(i * _sq_dim + j);
      }
    }
  }

  void set_state(std::vector<double> &s_) {
    s = s_;
    _E = H(s);
  }
  const std::vector<double> &state() const { return s; }

  void swap(SymBolzmanMP &rhs) {
    assert(dim == rhs.dim);
    rhs.s.swap(s);
    std::swap(rhs._E, _E);
  }

  double E() const { return _E; }
  double beta() const { return _beta; }

  State &update() {
    auto idx = idx_table[unif(rng)];
    auto xdi = _sq_dim * (idx % _sq_dim) + (idx / _sq_dim);

    auto mem = s[idx];
    auto delta = norm(rng);
    s[idx] += delta;
    s[xdi] += delta;

    auto E = H(s);
    if (_E >= E || std::exp(_beta * (_E - E)) >= unifr(rng)) {
      _E = E;
    } else {
      s[idx] = mem;
      s[xdi] = mem;
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

  double _E;
  double _beta;
  State s;
  int _sq_dim;
  std::vector<int> idx_table;
};

class Swapper {
 public:
  Swapper(Rng &rng) : rng(rng), prob(std::uniform_real_distribution<>(0, 1)) {}

  template <typename H>
  bool try_swap(SymBolzmanMP<H> &lhs, SymBolzmanMP<H> &rhs) {
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

class Energy {
 public:
  const int dim;
  const double threshold;
  SkipMean &score;
  Energy(int dim, double threshold, SkipMean &score)
      : dim(dim), threshold(threshold), score(score) {}

  double operator()(const std::vector<double> &K) {
    const double inf = 1e10;
    for (auto k : K) {
      if (k < 0) return inf;
    }

    auto sum = 0.0;
    for (auto k : K) sum += k;

    auto res = score(K);
    return (res > threshold) ? sum / dim : inf;
  }
};

std::vector<double> symmetric(const std::vector<double> &left) {
  // assume (i<j -> left[i] < left[j] && left[i] > 0)

  std::vector<double> out;

  for (int i = left.size() - 1; i >= 0; i--) out.push_back(-left[i]);
  out.push_back(0);
  for (int i = 0; i < left.size(); i++) out.push_back(left[i]);

  return out;
}

template <typename Rng>
std::vector<double> phase_unif(int n, Rng &rng) {
  std::uniform_real_distribution<> unif(0, 2 * Pi);

  std::vector<double> res(n);
  for (int i = 0; i < n; i++) res[i] = unif(rng);
  return res;
}

void run(std::string &base) {
  /**
   * TODO:
   *  1. 収束判定の導入: オーダーが大きい領域で収束をどのように判定するか？
   *  2. 更新則の対称化: 現状更新則は非対称
   * <- hardcodingした
   *
   *  3. 計算の並列化:
   * 現状シングルスレッドで計算している。rngはそれぞれのスレッドに必要
   * <- OpneMPで簡易な並列化を実装
   *
   *  4. オーダーを出力するための再設計:
   * 現状ではオーダーが見えない。構造を変えてよりみやすく
   *  5. 交換が十分に起こっていいない: 温度の間隔を調整しないといけない
   * <- 間隔を細かくした。並列数に対して計算量は線形にも増えない(並列化のおかげ)
   *
   */
  const auto R = betas.size();
  Rng rng(42);
  std::vector<Rng> rngs;
  for (int i = 0; i < R; i++) rngs.push_back(Rng(rng()));

  // 位相振動子
  auto w0 = symmetric(w_left);
  const int D_model = w0.size();
  auto s0 = phase_unif(D_model, rng);
  std::vector<SkipMean> dynamics(R, SkipMean(PhaseRK4(w0, s0)));
  std::vector<Energy> H_list;
  for (auto &p : dynamics) {
    H_list.push_back(Energy(D_model, threshold, p));
  }

  std::vector<SymBolzmanMP<Energy>> reprica;
  {
    // 絶対同期する結合を初期値に
    std::vector<double> K0(D_model * D_model, 10);
    for (int i = 0; i < D_model; i++) K0[i * D_model + i] = 0;

    for (int i = 0; i < R; i++) {
      auto m = SymBolzmanMP(D_model * D_model, H_list[i], betas[i], rngs[i]);
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
  // TODO: ここまでの状態をセーブできると何かと便利

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

  std::ofstream ofs("./phase.out");
  ofs << base << std::endl;
}

int main() {
  std::string base;
  {
    auto now = std::chrono::system_clock::now();
    std::time_t stamp = std::chrono::system_clock::to_time_t(now);
    const std::tm *lt = std::localtime(&stamp);
    std::stringstream ss;
    ss << "./output/phase" << std::put_time(lt, "%c") << "/";
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
