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
class Kuramoto {
  struct Unit {
    Real cos;
    Real sin;

    Unit(Real cos, Real sin) : cos(cos), sin(sin) {}
  };

 public:
  const int ndim;
  Kuramoto(int window, int ndim)
      : ndim(ndim), cos_q(window, 0.), sin_q(window, 0.) {}

  Unit push(const Real *s, const Real *ds_dt) noexcept {
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

  Unit push(const Unit inner) noexcept {
    auto cos_pop = cos_q.push(inner.cos);
    auto sin_pop = sin_q.push(inner.sin);
    return Unit(cos_pop, sin_pop);
  }

  Real value() const noexcept {
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
 * \sum_{i > 1} e^{i(\theta_i(t) - \theta_0)}
 */
template <typename Real>
class RelativeKuramoto {
  struct Unit {
    Real cos;
    Real sin;

    Unit(Real cos, Real sin) : cos(cos), sin(sin) {}
  };

 public:
  const int ndim;
  RelativeKuramoto(int window, int ndim)
      : ndim(ndim), cos_q(window, 0.), sin_q(window, 0.) {}

  Unit push(const Real *s, const Real *ds_dt) noexcept {
    Real cos_mean = 0;
    for (int i = 1; i < ndim; i++) {
      cos_mean += std::cos(s[i] - s[0]);
    }
    cos_mean /= ndim - 1;

    Real sin_mean = 0;
    for (int i = 1; i < ndim; i++) {
      sin_mean += std::sin(s[i] - s[0]);
    }
    sin_mean /= ndim - 1;

    return push({cos_mean, sin_mean});
  }

  Unit push(const Unit inner) noexcept {
    auto cos_pop = cos_q.push(inner.cos);
    auto sin_pop = sin_q.push(inner.sin);
    return Unit(cos_pop, sin_pop);
  }

  Real value() const noexcept {
    const auto cos_mean = cos_q.mean();
    const auto sin_mean = sin_q.mean();
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
class ZeroFreqRate {
  struct Unit {
    Real ratio;
  };

 public:
  const int ndim;
  ZeroFreqRate(int window, int ndim) : ndim(ndim), ratio_q(window, 0.) {}

  Unit push(const Real *s, const Real *ds_dt) noexcept {
    const Real epsilon = 1e-2;
    Real ratio = 0;
    for (int i = 0; i < ndim; i++) {
      ratio += (std::abs(ds_dt[i]) < epsilon);
    }
    ratio /= ndim;

    return push({ratio});
  }

  Unit push(const Unit inner) noexcept {
    const auto ratio = ratio_q.push(inner.ratio);
    return {ratio};
  }

  Real value() const noexcept { return ratio_q.mean(); }

 private:
  collection::FixedQueue<Real> ratio_q;
};

/**
 * Estimate mean frequency and calculate freq 0 order.
 * TODO: maybe this is slow because it create valarray every times
 */
template <typename Real>
class ZeroFreqMean {
  struct Unit {
    std::valarray<Real> freq;
  };

 public:
  const int ndim;
  ZeroFreqMean(int window, int ndim)
      : ndim(ndim), freq_q(window, std::valarray<Real>(Real(0), ndim)) {}

  Unit push(const Real *s, const Real *ds_dt) noexcept {
    return push({std::valarray<Real>(ds_dt, ndim)});
  }

  Unit push(const Unit inner) noexcept {
    const auto freq = freq_q.push(inner.freq);
    return {freq};
  }

  Real value() const noexcept {
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
