#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <functional>
#include <cassert>

template<typename Energy>
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
        unifr(std::uniform_real_distribution<>(0, 1)),
        norm(std::normal_distribution<>(0, step_size))
  {
    State s_(dim, 0);
    set_state(s_);
  }

  void set_state(std::vector<double> &s_) { 
    s = s_;
    _E = H(s);
  }

  double E() { return _E; }

  State &update()
  {
    auto idx = (t++) % dim;
    auto mem = s[idx];
    s[idx] += norm(rng);

    auto E = H(s);
    if (E >= _E || std::exp(E - _E) >= unifr(rng)) {
      _E = E;
    } else {
      s[idx] = mem;
    }

    return s;
  }

  State &update(int n) {
    for (int i=0; i<n-1; i++) {
      update();
    }
    return update();
  }

private:
  std::mt19937 rng;
  std::uniform_real_distribution<> unifr;
  std::normal_distribution<> norm;

  double _E;
  uint t = 0;
};


int main()
{
  auto N = 1<<18;
  auto T = 16;

  double sigma[2][2] = {
    {2, 1},
    {1, 3}
  };
  int D = 2;

  double sigma_inv[2][2] = {
    {sigma[1][1], -sigma[1][0]},
    {-sigma[0][1], sigma[0][0]}
  };
  for(int i=0; i<D; i++) {
    for(int j=0; j<D; j++) {
      sigma_inv[i][j] /= 5;
    }
  }

  assert(sigma_inv[0][0] * sigma[0][0] + sigma_inv[0][1] * sigma[1][0] == 1);
  assert(std::abs(sigma_inv[0][0] * sigma[0][1] + sigma_inv[0][1] * sigma[1][1]) <= 0.000001);


  std::mt19937 rng(42);
  std::normal_distribution<> norm(0, 1.0);

  /*
  exp(-1/2 \sum_ij (A^-1)_ij x_i x_j)
  */
  auto H = [&](std::vector<double> &s){
    if (s[0] * s[0] + 3.0 >= s[1]) {
      return -1000000.0;
    }
    auto res = 0.0;
    for(int i=0; i<D; i++) {
      for(int j=0; j<D; j++) {
        res += sigma_inv[i][j] * s[i] * s[j];
      }
    }
    return -res/2;
  };
  BolzmanMP<decltype(H)> model(D, H);
  for(int i=0; i<1000; i++) {
    model.update();
  }


  std::ofstream ofs("./const.csv");
  ofs << "x[0]" << "," << "x[1]" << "," ;
  ofs << "y[0]" << "," << "y[1]" << "," ;
  ofs << "E" << std::endl;
  for (int i=0; i<N; i++) {

    double x[2], y[2];

    auto x0 = norm(rng);
    auto x1 = norm(rng);
    x[0] = sigma[0][0] * x0 + sigma[0][1] * x1;
    x[1] = sigma[1][0] * x0 + sigma[1][1] * x1;

    auto s = model.update(T);
    y[0] = s[0];
    y[1] = s[1];

    ofs << x[0] << "," << x[1] << "," ;
    ofs << y[0] << "," << y[1] << "," ;
    ofs << model.E() << std::endl;
  }
}