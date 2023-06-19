// #include <immintrin.h>
#include <omp.h>

// #define NDEBUG

#include <cassert>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

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
  std::vector<double> &state() { return s; }

  void swap(SymBolzmanMP &rhs) {
    assert(dim == rhs.dim);
    rhs.s.swap(s);
    std::swap(rhs._E, _E);
  }

  double E() { return _E; }
  double beta() { return _beta; }

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

class Csv {
 private:
  bool is_open = true;

 public:
  std::ofstream ofs;
  Csv(const char *file) : ofs(std::ofstream(file)) {}

  void close() {
    assert(is_open);
    is_open = false;
    ofs.close();
  }

  void open(const char *file) {
    assert(!is_open);
    is_open = true;
    ofs.open(file);
  }

  class Row {
   private:
    Csv &parent;
    bool is_first = true;

   public:
    Row(Csv &parent) : parent(parent) {}
    ~Row() { parent.ofs << std::endl; }

    Row &content(double x) {
      if (is_first) {
        parent.ofs << x;
        is_first = false;
      } else {
        parent.ofs << "," << x;
      }
      return *this;
    }
  };

  Row new_row() { return Row(*this); }

  template<typename Number>
  void save_mtx(const char* filename, int nrow, int ncol, const Number *data) {
    open(filename);
    for (int i = 0; i < nrow; i++) {
      auto row = new_row();
      for (int j = 0; j < ncol; j++) {
        row.content(data[i * ncol + j]);
      }
    }
    close();
  }
};

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
      if (std::abs(w[i]) < eps) c++;
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

  double operator()(std::vector<double> &K) {
    // TODO: hardcode
    const int N_skip = 4000;
    const int N_mean = 1000;

    for (int i = 0; i < N_skip; i++) {
      model.step(K);
    }
    double m = 0;
    for (int i = 0; i < N_mean; i++) {
      model.step(K);
      m += (model.phase_order() - m) / (i + 1);
    }
    return m;
  }
};

class Energy {
 public:
  const int dim;
  const double threshold;
  SkipMean &score;
  Energy(int dim, double threshold, SkipMean &score)
      : dim(dim), threshold(threshold), score(score) {}

  double operator()(std::vector<double> &K) {
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

void save_param(const char *filename, int T_sampling,
                const std::vector<double> &beta, double threshold, int burn_in,
                int T_swap, const std::vector<double> &w0) {
  std::ofstream ofs(filename);
  {
    auto now = std::chrono::system_clock::now();
    std::time_t stamp = std::chrono::system_clock::to_time_t(now);
    ofs << "#" << std::ctime(&stamp) << std::endl;
  }
  ofs << "T_sampling: " << T_sampling << std::endl;

  ofs << "beta: [" << beta[0];
  for (int i = 1; i < beta.size(); i++) ofs << "," << beta[i];
  ofs << "]" << std::endl;

  ofs << "threshold: " << threshold << std::endl;
  ofs << "burn_in: " << burn_in << std::endl;
  ofs << "T_swap: " << T_swap << std::endl;

  ofs << "w0: [" << w0[0];
  for (int i = 1; i < w0.size(); i++) ofs << "," << w0[i];
  ofs << "]" << std::endl;
}

std::vector<double> symmetric(const std::vector<double>& left) {
  // assume (i<j -> left[i] < left[j] && left[i] > 0)

  std::vector<double> out;
  
  for(int i=left.size()-1; i>=0; i--) out.push_back(-left[i]);
  out.push_back(0);
  for(int i=0; i<left.size(); i++) out.push_back(left[i]);

  return out;
}

template<typename Rng>
std::vector<double> phase_unif(int n, Rng& rng) {
  std::uniform_real_distribution<> unif(0, 2 * Pi);

  std::vector<double> res(n);
  for (int i = 0; i < n; i++) res[i] = unif(rng);
  return res;
}

int main() {
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

  // MCMC param
  const auto N_sample = 10;
  const auto T_sampling = 200;  // TODO: 決めないと
  const static int burn_in = 5000;
  const int T_swap = 10;
  const auto betas = std::vector<double>({0.1, 1.0, 10.});
  // Phase param
  const auto w_left =
      std::vector<double>({0.268, 0.5773, 0.9998, 1.7311, 3.7298});
  const double threshold = 0.99;

  save_param("./phase_param.yaml", T_sampling, betas, threshold, burn_in,
             T_swap, w_left);

  const auto R = betas.size();
  Rng rng(42);
  std::vector<Rng> rngs;
  for (int i = 0; i < R; i++) rngs.push_back(Rng(rng()));

  // 位相振動子
  auto w0 = symmetric(w_left);
  const int dim = w0.size();
  auto s0 = phase_unif(dim, rng);
  std::vector<SkipMean> dynamics(R, SkipMean(PhaseRK4(w0, s0)));
  std::vector<Energy> H_list;
  for (auto &p : dynamics) {
    H_list.push_back(Energy(dim, threshold, p));
  }

  std::vector<SymBolzmanMP<Energy>> reprica;
  {
    // 絶対同期する結合を初期値に
    std::vector<double> K0(dim * dim, 10);
    for (int i = 0; i < dim; i++) K0[i * dim + i] = 0;

    for (int i = 0; i < R; i++) {
      auto m = SymBolzmanMP(dim * dim, H_list[i], betas[i], rngs[i]);
      m.set_state(K0);

      reprica.push_back(std::move(m));
    }
  }
  auto swapper = Swapper(rng);
  std::uniform_int_distribution<> swap_idx(0, R - 2);

  // hotin
#pragma omp parallel for
  for (int j = 0; j < R; j++) {
    reprica[j].update(burn_in);
  }
  // TODO: ここまでの状態をセーブできると何かと便利

  Csv csv("./phase.csv");
  std::vector<int> swapped(R);
  for (int i = 0; i < R; i++) swapped[i] = i;
  std::vector<int> swap_history(R * N_sample / T_swap);
  std::vector<double> Es(R * N_sample);
  std::vector<double> state(R * dim);
  for (int i = 0; i < N_sample; i++) {
#pragma omp parallel for
    for (int j = 0; j < R; j++) {
      auto s = reprica[j].update(T_sampling);

      Es[i * R + j] = reprica[j].E();
      for (int k = 0; k < dim; k++) state[j * dim + k] = s[k];
    }

    if (i != 0 && i % T_swap == 0) {
      int target = swap_idx(rng);
      if (swapper.try_swap(reprica[target], reprica[target + 1])) {
        std::swap(swapped[target], swapped[target + 1]);
      }
      for (int j = 0; j < R; j++) swap_history[i * R + j] = swapped[j];
    }

    auto row = csv.new_row();
    for (auto s : state) row.content(s);
  }
  std::cout << std::endl;
  csv.close();

  csv.save_mtx("./phase_E.csv", N_sample, R, Es.data());
  csv.save_mtx("./phase_swap.csv", N_sample / T_swap, R, swap_history.data());
}
