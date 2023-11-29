#pragma once

#ifdef SIM_AVX2
#include <ia32intrin.h>
#include <immintrin.h>
#endif

#include <_math.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <memory>

namespace lib {
namespace sim {

template <typename Real>
void target_model(int ndim, const Real *K, int Kstride, const Real *w,
                  const Real t, const Real *s, Real *ds_dt) noexcept {
  for (int i = 0; i < ndim; i++) {
    ds_dt[i] = w[i];
  }

  for (int i = 0; i < ndim; i++) {
    for (int j = 0; j < ndim; j++) {
      ds_dt[i] += K[i * Kstride + j] * std::sin(s[j] - s[i]);
    }
  }
}

#ifdef SIM_AVX2

template <>
void target_model(int ndim, const double *K, int Kstride, const double *w,
                  const double t, const double *s, double *ds_dt) noexcept {
  constexpr int simd_size = 4;
  const int simd_iteration = ndim / simd_size;
  const int remainder = ndim % simd_size;

  for (int i = 0; i < ndim; i += simd_size) {
    __m256d ds_dt_i = _mm256_load_pd(&w[i]);
    __m256d s_i = _mm256_load_pd(&s[i]);
    for (int j = 0; j < ndim; j++) {
      __m256d sin_diff =
          _mm256_sin_pd(_mm256_sub_pd(_mm256_set1_pd(s[j]), s_i));
      __m256d K_ = _mm256_load_pd(&K[j * Kstride + i]);
      ds_dt_i = _mm256_add_pd(ds_dt_i, _mm256_mul_pd(K_, sin_diff));
    }

    switch (ndim - i) {
      case 3: {
        __m256i mask = _mm256_setr_epi64x(~0, ~0, ~0, 0);
        _mm256_maskstore_pd(&ds_dt[i], mask, ds_dt_i);
        break;
      }
      case 2: {
        __m256i mask = _mm256_setr_epi64x(~0, ~0, 0, 0);
        _mm256_maskstore_pd(&ds_dt[i], mask, ds_dt_i);
        break;
      }
      case 1: {
        __m256i mask = _mm256_setr_epi64x(~0, 0, 0, 0);
        _mm256_maskstore_pd(&ds_dt[i], mask, ds_dt_i);
        break;
      }
      default:
        _mm256_store_pd(&ds_dt[i], ds_dt_i);
        break;
    }
  }
}

#endif

template <typename Real>
class RK4 {
  struct Result {
    Real t;
  };

 public:
  const int ndim;
  const Real max_dt;

  RK4(const int ndim, const Real max_dt)
      : ndim(ndim),
        max_dt(max_dt),
        _s(new(std::align_val_t{64}) Real[ndim]),
        k1(new(std::align_val_t{64}) Real[ndim]),
        k2(new(std::align_val_t{64}) Real[ndim]),
        k3(new(std::align_val_t{64}) Real[ndim]),
        k4(new(std::align_val_t{64}) Real[ndim]) {}

  Result advance(Real T, Real t, Real *s, const Real *K, int Kstride,
                 const Real *w) noexcept {
    const Real t_max = t + T;

    while (t < t_max) {
      const Real h = (t + max_dt <= t_max) ? max_dt : t_max - t;
      t = _advance_dt(h, t, s, K, Kstride, w);
    }
    assert(std::abs(t - t_max) < 1e-6);

    return {t};
  }

 private:
  Real _advance_dt(const Real dt, const Real t, Real *s, const Real *K,
                   int Kstride, const Real *w) {
    const Real dt_2 = dt * 0.5;
    const Real dt_6 = dt / 6;

    target_model(ndim, K, Kstride, w, t, s, &k1[0]);
    sumofp(ndim, &_s[0], s, dt_2, &k1[0]);
    target_model(ndim, K, Kstride, w, t + dt_2, &_s[0], &k2[0]);
    sumofp(ndim, &_s[0], s, dt_2, &k2[0]);
    target_model(ndim, K, Kstride, w, t + dt_2, &_s[0], &k3[0]);
    sumofp(ndim, &_s[0], s, dt, &k3[0]);
    target_model(ndim, K, Kstride, w, t + dt, &_s[0], &k4[0]);
    sumofp(ndim, s, s, dt_6, &k1[0], dt_6 * 2, &k2[0], dt_6 * 2, &k3[0], dt_6,
           &k4[0]);

    return t + dt;
  }

  std::unique_ptr<Real[]> _s;
  std::unique_ptr<Real[]> k1, k2, k3, k4;
};

template <typename Real>
class FehlbergRK45 {
  struct Result {
    Real t;
    int iteration;
  };

 public:
  const Real atol;
  const Real first_h, min_h, max_h;
  const int ndim;
  FehlbergRK45(int ndim, Real first_h, Real max_h, Real atol = 1e-3,
               Real min_h = 1e-6)
      : atol(atol),
        first_h(first_h),
        min_h(min_h),
        max_h(max_h),
        ndim(ndim),
        tmp(new(std::align_val_t{64}) Real[ndim]),
        k0(new(std::align_val_t{64}) Real[ndim]),
        k1(new(std::align_val_t{64}) Real[ndim]),
        k2(new(std::align_val_t{64}) Real[ndim]),
        k3(new(std::align_val_t{64}) Real[ndim]),
        k4(new(std::align_val_t{64}) Real[ndim]),
        k5(new(std::align_val_t{64}) Real[ndim]) {}

  /**
   * advance T time from (t, s) on the model.
   */
  Result advance(Real T, Real t, Real *s, const Real *K, int Kstride,
                 const Real *w) noexcept {
    const Real t_max = t + T;
    h = first_h;

    int iteration = 0;
    while (t + min_h < t_max) {
      iteration++;
      const auto h = try_advance(t, s, K, Kstride, w, t_max);
      if (h > 0) {
        t += h;
      }
    }

    return {t, iteration};
  }

 private:
  Real norm(int size, const Real *v) { return *std::max_element(v, v + size); }
  Real try_advance(Real t, Real *s, const Real *K, int Kstride, const Real *w,
                   Real t_max) {
    /**
     * reference: https://slpr.sakura.ne.jp/qp/runge-kutta-ex/#rkf45hauto
     */
    h = std::min(h, max_h);
    h = std::min(h, t_max - t);
    assert(h > 0);
    assert(h <= t_max - t);

    static constexpr Real c_0 = 1. / 4;
    static constexpr Real a_00 = 1. / 4;

    static constexpr Real c_1 = 3. / 8;
    static constexpr Real a_10 = 3. / 32;
    static constexpr Real a_11 = 9. / 32;

    static constexpr Real c_2 = 12. / 13;
    static constexpr Real a_20 = 1932. / 2197;
    static constexpr Real a_21 = -7200. / 2197;
    static constexpr Real a_22 = 7296. / 2197;

    static constexpr Real c_3 = 1.;
    static constexpr Real a_30 = 439. / 216;
    static constexpr Real a_31 = -8.;
    static constexpr Real a_32 = 3680. / 513;
    static constexpr Real a_33 = -845. / 4104;

    static constexpr Real c_4 = 1. / 2;
    static constexpr Real a_40 = -8. / 27;
    static constexpr Real a_41 = 2;
    static constexpr Real a_42 = -3544. / 2565;
    static constexpr Real a_43 = 1859. / 4104;
    static constexpr Real a_44 = -11. / 40;

    static constexpr Real b_0 = 1. / 360;
    static constexpr Real b_1 = -128. / 4275;
    static constexpr Real b_2 = -2197. / 75240;
    static constexpr Real b_3 = 1. / 50;
    static constexpr Real b_4 = 2. / 55;

    static constexpr Real d_0 = 25. / 216;
    static constexpr Real d_1 = 1408. / 2565;
    static constexpr Real d_2 = 2197. / 4104;
    static constexpr Real d_3 = -1. / 5;

    target_model(ndim, K, Kstride, w, t, s, &k0[0]);

    sumofp(ndim, &tmp[0], s, h * a_00, &k0[0]);
    target_model(ndim, K, Kstride, w, t + c_0 * h, &tmp[0], &k1[0]);

    sumofp(ndim, &tmp[0], s, h * a_10, &k0[0], h * a_11, &k1[0]);
    target_model(ndim, K, Kstride, w, t + c_1 * h, &tmp[0], &k2[0]);

    sumofp(ndim, &tmp[0], s, h * a_20, &k0[0], h * a_21, &k1[0], h * a_22,
           &k2[0]);
    target_model(ndim, K, Kstride, w, t + c_2 * h, &tmp[0], &k3[0]);

    sumofp(ndim, &tmp[0], s, h * a_30, &k0[0], h * a_31, &k1[0], h * a_32,
           &k2[0], h * a_33, &k3[0]);
    target_model(ndim, K, Kstride, w, t + c_3 * h, &tmp[0], &k4[0]);

    sumofp(ndim, &tmp[0], s, h * a_40, &k0[0], h * a_41, &k1[0], h * a_42,
           &k2[0], h * a_43, &k3[0], h * a_44, &k4[0]);
    target_model(ndim, K, Kstride, w, t + c_4 * h, &tmp[0], &k5[0]);

    sumofp(ndim, &tmp[0], b_0, &k0[0], b_1, &k2[0], b_2, &k3[0], b_3, &k4[0],
           b_4, &k5[0]);
    const auto R = norm(ndim, &tmp[0]) + std::numeric_limits<Real>::epsilon();
    const auto R_base = atol;

    // Try to update state
    Real dt = 0;
    if (R < R_base) {
      t += (dt = h);
      sumofp(ndim, s, s, h * d_0, &k0[0], h * d_1, &k2[0], h * d_2, &k3[0],
             h * d_3, &k4[0]);
    }

    const auto delta = std::pow(atol / (2 * R), 0.25);
    h *= (delta <= 0.1) ? 0.1 : (delta >= 4) ? 4 : delta;

    reliability = R;
    return dt;
  }
  Real h;
  Real reliability;
  std::unique_ptr<Real[]> tmp;
  std::unique_ptr<Real[]> k0;
  std::unique_ptr<Real[]> k1;
  std::unique_ptr<Real[]> k2;
  std::unique_ptr<Real[]> k3;
  std::unique_ptr<Real[]> k4;
  std::unique_ptr<Real[]> k5;
};
}  // namespace sim
}  // namespace lib