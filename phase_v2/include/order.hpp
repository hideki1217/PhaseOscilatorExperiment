#include <cmath>
#include <collection.hpp>

namespace lib {
namespace order {

template <typename Real>
struct InnerUnit {
  Real cos;
  Real sin;

  InnerUnit(Real cos, Real sin) : cos(cos), sin(sin) {}
};

template <typename Real>
class KuramotoFixed {
 public:
  const int ndim;
  KuramotoFixed(int window, int ndim)
      : ndim(ndim), cos_q(window, 0.), sin_q(window, 0.) {}

  InnerUnit<Real> push(const Real *s) {
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

  InnerUnit<Real> push(const InnerUnit<Real> inner) {
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

template <typename Real>
class FreqFixed {
  struct Unit {
    Real ratio;
  };

 public:
  const int ndim;
  FreqFixed(int window, int ndim) : ndim(ndim), ratio_q(window, 0.) {}

  Unit push(const Real *s) {
    std::abort();  // TODO
    const Real epsilon = 1e-4;
    Real ratio = 0;
    for (int i = 0; i < ndim; i++) {
      ratio += (s[i] < epsilon);
    }
    ratio /= ndim;

    return push({ratio});
  }

  Unit push(const Unit inner) {
    const auto ratio = ratio_q.push(inner.ratio);
    return {ratio};
  }

  Real value() { return ratio_q.mean(); }

 private:
  collection::FixedQueue<Real> ratio_q;
};
}  // namespace order
}  // namespace lib
