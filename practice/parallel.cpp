#include <cassert>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

using Rng = std::mt19937;

template <typename Energy>
class BolzmanMP {
 public:
  using State = std::vector<double>;
  const int dim;
  const Energy H;

  BolzmanMP(int dim, Energy energy, double beta, Rng &rng,
            double step_size = 1.0, int seed = 12)
      : dim(dim),
        H(energy),
        _beta(beta),
        rng(rng),
        unif(std::uniform_int_distribution<>(0, dim - 1)),
        unifr(std::uniform_real_distribution<>(0, 1)),
        norm(std::normal_distribution<>(0, step_size)) {
    assert(beta > 0);
    State s_(dim, 0);
    set_state(s_);
  }

  void set_state(std::vector<double> &s_) {
    s = s_;
    _E = H(s);
  }
  std::vector<double> &state() { return s; }

  void swap(BolzmanMP &rhs) {
    assert(dim == rhs.dim);
    rhs.s.swap(s);
    std::swap(rhs._E, _E);
  }

  double E() { return _E; }
  double beta() { return _beta; }

  State &update() {
    auto idx = unif(rng);
    auto mem = s[idx];
    s[idx] += norm(rng);

    auto E = H(s);
    if (_E >= E || std::exp(_beta * (_E - E)) >= unifr(rng)) {
      _E = E;
    } else {
      s[idx] = mem;
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
};

class Swapper {
 public:
  Swapper(Rng &rng) : rng(rng), prob(std::uniform_real_distribution<>(0, 1)) {}

  template <typename F>
  bool try_swap(BolzmanMP<F> &lhs, BolzmanMP<F> &rhs) {
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

template <typename OFS>
class CsvWriter {
 public:
  enum class Mode { Header, Content } mode;

  OFS &ofs;

  CsvWriter(OFS &ofs, bool skip_header = false)
      : mode(skip_header ? Mode::Content : Mode::Header), ofs(ofs) {}

  CsvWriter &header(char *x) {
    assert(mode == Mode::Header);
    if (is_first) {
      ofs << x;
      is_first = false;
    } else {
      ofs << "," << x;
    }
    return *this;
  }

  CsvWriter &content(double x) {
    assert(mode == Mode::Content);
    if (is_first) {
      ofs << x;
      is_first = false;
    } else {
      ofs << "," << x;
    }
    return *this;
  }

  CsvWriter &newrow() {
    if (mode == Mode::Header) {
      if (!is_first) {
        ofs << std::endl;
      }

      mode = Mode::Content;
      is_first = true;
    }

    if (mode == Mode::Content) {
      if (!is_first) {
        ofs << std::endl;
      }

      is_first = true;
    }
    return *this;
  }

  void complete() { newrow(); }

 private:
  bool is_first = true;
};

int main() {
  Rng rng(42);

  const auto N = 8000;
  const auto T = 32;

  // 混合ガウス分布
  const double sigma = 0.01;
  constexpr int D = 2;
  constexpr int C = 2;
  const double m[C][D] = {{1, 1}, {-1, -1}};
  auto w = std::vector<double>({1, 1});
  {
    auto sum = 0;
    for (auto x : w) sum += x;
    for (auto &x : w) x /= sum;
  }
  auto H = [&](std::vector<double> &s) -> double {
    auto res = 0.0;
    for (int c = 0; c < C; c++) {
      auto p = 0.0;
      for (int i = 0; i < D; i++) {
        p += std::pow(s[i] - m[c][i], 2);
      }
      res += std::exp(-p / (2 * sigma)) * w[c];
    }
    return -std::log(res);
  };

  auto reprica = std::vector({
      BolzmanMP(D, H, 0.01, rng),
      BolzmanMP(D, H, 0.1, rng),
      BolzmanMP(D, H, 1, rng),
      BolzmanMP(D, H, 10, rng),
      BolzmanMP(D, H, 100, rng),
  });
  const int R = reprica.size();
  auto swapper = Swapper(rng);

  // hotin
  for (auto &model : reprica) {
    model.update(1000);
  }

  std::ofstream ofs("./parallel.csv");
  CsvWriter writer(ofs, true);

  std::vector<int> history(R);
  for(int i=0; i<R; i++)history[i] = i;
  for (int i = 0; i < N; i++) {
    for (auto &model : reprica) {
      auto s = model.update(T);

      for (auto x : s) writer.content(x);
    }
    writer.newrow();

    if (i % 10 == 0) {
      static int count = 0;
      auto idx = (count++) % (R - 1);
      if (swapper.try_swap(reprica[idx], reprica[idx + 1])) {
        std::swap(history[idx], history[idx+1]);
      }
    }
    std::cout << history[0];
    for(int i=1; i<R; i++) std::cout << "," << history[i];
    std::cout << std::endl;
  }
}