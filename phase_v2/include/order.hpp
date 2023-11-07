#pragma once

#include <cmath>
#include <collection.hpp>
#include <valarray>

namespace lib {
namespace order {
/**
 * Estimate Kuramoto Phase parameter
 * $\frac{1}{T} \int_0^T $
 */
template <typename Real>
class KuramotoFixed {
  struct Unit {
    Real cos;
    Real sin;

    Unit(Real cos, Real sin) : cos(cos), sin(sin) {}
  };

 public:
  const int ndim;
  KuramotoFixed(int window, int ndim)
      : ndim(ndim), cos_q(window, 0.), sin_q(window, 0.) {}

  Unit push(const Real *s, const Real *ds_dt) {
    Real cos_mean = 0;
    for (int i = 0; i < ndim; i++) {
      cos_mean += std::cos(s[i]);
    }
    cos_mean /= ndim;

    Real sin_mean = 0;
    for (int i = 0; i < ndim; i++) {
      sin_mean += std::sin(s[i]);
    }
    sin_mean /= ndim;

    return push({cos_mean, sin_mean});
  }

  Unit push(const Unit inner) {
    auto cos_pop = cos_q.push(inner.cos);
    auto sin_pop = sin_q.push(inner.sin);
    return Unit(cos_pop, sin_pop);
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

/**
 * Estimate a probablity that Freqency sample is in [-D, D].
 * D = 0.1
 */
template <typename Real>
class ZeroFreqRateFixed {
  struct Unit {
    Real ratio;
  };

 public:
  const int ndim;
  ZeroFreqRateFixed(int window, int ndim) : ndim(ndim), ratio_q(window, 0.) {}

  Unit push(const Real *s, const Real *ds_dt) {
    const Real epsilon = 1e-1;
    Real ratio = 0;
    for (int i = 0; i < ndim; i++) {
      ratio += (std::abs(ds_dt[i]) < epsilon);
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

/**
 * Estimate mean frequency and calculate freq 0 order.
 * TODO: maybe this is slow because it create valarray every times
 */
template <typename Real>
class ZeroFreqMeanFixed {
  struct Unit {
    std::valarray<Real> freq;
  };

 public:
  const int ndim;
  ZeroFreqMeanFixed(int window, int ndim)
      : ndim(ndim), freq_q(window, std::valarray<Real>(Real(0), ndim)) {}

  Unit push(const Real *s, const Real *ds_dt) {
    return push({std::valarray<Real>(ds_dt, ndim)});
  }

  Unit push(const Unit inner) {
    const auto freq = freq_q.push(inner.freq);
    return {freq};
  }

  Real value() {
    static const Real eps = 1e-2;
    const auto freq_means = freq_q.mean();

    Real r = 0;
    for (auto freq : freq_means) {
      r += (std::abs(freq) < eps);
    }
    r /= ndim;

    return r;
  }

 private:
  collection::FixedQueue<std::valarray<Real>> freq_q;
};
}  // namespace order
}  // namespace lib
