#include <memory>

template <typename H>
class Hmc {
 public:
  const int dim;
  Hmc(int dim)
      : dim(dim),
        s(std::make_unique(dim)),
        p(std::make_unique(dim)),
        _s(std::make_unique(dim)) {}

  const double* update() {
    mechanics_step();
    random_step();
  }

 private:
  void mechanics_step() {}

  void random_step() {

  }

      std::unique_ptr<double[]>
          s;
  std::unique_ptr<double[]> p;
  std::unique_ptr<double[]> _s;
};

int main() {
  // 試しに混合ガウス分布でやってみる
}