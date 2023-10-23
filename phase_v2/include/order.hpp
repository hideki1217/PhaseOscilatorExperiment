#include <cmath>
#include <collection.hpp>

namespace lib {
namespace order {

template <typename Real>
class AverageOrder {
 private:
  struct InnerUnit {
    Real cos;
    Real sin;

    InnerUnit(Real cos, Real sin) : cos(cos), sin(sin) {}
  };

 public:
  const int ndim;
  AverageOrder(int window, int ndim)
      : ndim(ndim), cos_q(window, 0.), sin_q(window, 0.) {}

  InnerUnit push(const Real *s) {
    Real cos_mean = 0;
    for (int i = 0; i < ndim; i++) {
      cos_mean += std::cos(s[i]);
    }
    cos_mean /= ndim;
    auto cos_pop = cos_q.push(cos_mean);

    Real sin_mean = 0;
    for (int i = 0; i < ndim; i++) {
      sin_mean += std::sin(s[i]);
    }
    sin_mean /= ndim;
    auto sin_pop = sin_q.push(sin_mean);

    return InnerUnit(cos_pop, sin_pop);
  }

  InnerUnit push(const InnerUnit inner) {
    auto cos_pop = cos_q.push(inner.cos);
    auto sin_pop = sin_q.push(inner.sin);
    return InnerUnit(cos_pop, sin_pop);
  }

  Real value() {
    auto cos_mean = cos_q.mean();
    auto sin_mean = sin_q.mean();
    const auto R_new = std::sqrt(cos_mean * cos_mean + sin_mean * sin_mean);
    return R_new;
  }

 private:
  collection::FixedQueue<Real> cos_q;
  collection::FixedQueue<Real> sin_q;
};

}  // namespace order
}  // namespace lib
