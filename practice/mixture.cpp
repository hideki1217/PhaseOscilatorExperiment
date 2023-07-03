#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <functional>
#include <cassert>

template <typename Energy>
class BolzmanMP
{
public:
  using State = std::vector<double>;
  const int dim;
  State s;
  Energy H;


  BolzmanMP(int dim, Energy energy, double step_size = 1.0, int seed = 12)
      : dim(dim),
        H(energy),
        rng(std::mt19937(seed)),
        unif(std::uniform_int_distribution<>(0, dim - 1)),
        unifr(std::uniform_real_distribution<>(0, 1)),
        norm(std::normal_distribution<>(0, step_size))
  {
    State s_(dim, 0);
    set_state(s_);
  }

  void set_state(std::vector<double> &s_)
  {
    s = s_;
    _E = H(s);
  }

  State &update()
  {
    auto idx = unif(rng);
    auto mem = s[idx];
    s[idx] += norm(rng);

    auto E = H(s);
    if (E >= _E || std::exp(E - _E) >= unifr(rng))
    {
      _E = E;
    }
    else
    {
      s[idx] = mem;
    }

    return s;
  }

  State &update(int T) {
    assert(T > 0);
    for(int i=0; i<T-1; i++) {
      update();
    }
    return update();
  }

private:
  std::mt19937 rng;
  std::uniform_int_distribution<> unif;
  std::uniform_real_distribution<> unifr;
  std::normal_distribution<> norm;

  double _E;
};

int main()
{
  auto N = 1600;
  auto T = 32;

  double sigma = 0.01;
  double m[2][2] = {
      {1, 1},
      {-1, -1}};
  int D = 2;
  int C = 2;

  std::mt19937 rng(42);
  std::normal_distribution<> norm(0, 1.0);

  auto H = [&](std::vector<double> &s)
  {
    auto res = 0.0;
    for (int c = 0; c < C; c++)
    {
      auto p = 0.0;
      for (int i = 0; i < D; i++)
      {
        p += (s[i] - m[c][i]) * (s[i] - m[c][i]);
      }
      res += 0.5 * std::exp(- p / (2 * sigma));
    }
    return std::log(res);
  };
  BolzmanMP<decltype(H)> model(D, H);
  model.update(1000); // hotin

  std::ofstream ofs("./mixture.csv");
  ofs << "y[0]"
      << ","
      << "y[1]" << std::endl;
  for (int i = 0; i < N; i++)
  {

    double x[2], y[2];

    for (int j = 0; j < T - 1; j++)
    {
      model.update();
    }
    auto s = model.update();
    y[0] = s[0];
    y[1] = s[1];

    ofs << y[0] << "," << y[1] << std::endl;
  }
}