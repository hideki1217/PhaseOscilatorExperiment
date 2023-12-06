#pragma once

#include <array>
#include <cassert>

#ifdef SIM_AVX2
#include <ia32intrin.h>
#include <immintrin.h>
#endif

namespace new_lib {
namespace system {

template <int Ndim>
struct System {
 public:
  using real_t = double;
  using state_t = std::array<real_t, Ndim>;
  static const int ndim = Ndim;

 public:
  const real_t* K;
  const int Kstride;
  const real_t* w;

 public:
  System(const real_t* K, int Kstride, const real_t* w)
      : K(K), Kstride(Kstride), w(w) {
    assert(K[0 * Kstride + 1] == K[1 * Kstride + 0]);
  }
  void operator()(const state_t& x, state_t& dx, double t) {
    system_impl(K, Kstride, w, t, &x[0], &dx[0]);
  }

 private:
  template <typename real_t>
  void system_impl(const real_t* K, int Kstride, const real_t* w,
                   const real_t t, const real_t* s, real_t* ds_dt) noexcept {
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
  void system_impl(const double* K, int Kstride, const double* w,
                   const double t, const double* s, double* ds_dt) noexcept {
    constexpr int simd_size = 4;

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
};
}  // namespace system
}  // namespace new_lib
