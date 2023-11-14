#pragma once

template <typename Real>
void sumofp(int ndim, Real *out, Real v1, const Real *x1) noexcept {
  for (int i = 0; i < ndim; i++) {
    out[i] = v1 * x1[i];
  }
}
template <typename Real>
void sumofp(int ndim, Real *out, Real v1, const Real *x1, Real v2,
            const Real *x2) noexcept {
  for (int i = 0; i < ndim; i++) {
    out[i] = v1 * x1[i] + v2 * x2[i];
  }
}
template <typename Real>
void sumofp(int ndim, Real *out, Real v1, const Real *x1, Real v2,
            const Real *x2, Real v3, const Real *x3) noexcept {
  for (int i = 0; i < ndim; i++) {
    out[i] = v1 * x1[i] + v2 * x2[i] + v3 * x3[i];
  }
}
template <typename Real>
void sumofp(int ndim, Real *out, Real v1, const Real *x1, Real v2,
            const Real *x2, Real v3, const Real *x3, Real v4,
            const Real *x4) noexcept {
  for (int i = 0; i < ndim; i++) {
    out[i] = v1 * x1[i] + v2 * x2[i] + v3 * x3[i] + v4 * x4[i];
  }
}
template <typename Real>
void sumofp(int ndim, Real *out, Real v1, const Real *x1, Real v2,
            const Real *x2, Real v3, const Real *x3, Real v4, const Real *x4,
            Real v5, const Real *x5) noexcept {
  for (int i = 0; i < ndim; i++) {
    out[i] = v1 * x1[i] + v2 * x2[i] + v3 * x3[i] + v4 * x4[i] + v5 * x5[i];
  }
}

template <typename Real>
void sumofp(int ndim, Real *out, Real *x, Real v1, const Real *x1) noexcept {
  for (int i = 0; i < ndim; i++) {
    out[i] = x[i] + v1 * x1[i];
  }
}
template <typename Real>
void sumofp(int ndim, Real *out, const Real *x, Real v1, const Real *x1,
            Real v2, const Real *x2) noexcept {
  for (int i = 0; i < ndim; i++) {
    out[i] = x[i] + v1 * x1[i] + v2 * x2[i];
  }
}
template <typename Real>
void sumofp(int ndim, Real *out, const Real *x, Real v1, const Real *x1,
            Real v2, const Real *x2, Real v3, const Real *x3) noexcept {
  for (int i = 0; i < ndim; i++) {
    out[i] = x[i] + v1 * x1[i] + v2 * x2[i] + v3 * x3[i];
  }
}
template <typename Real>
void sumofp(int ndim, Real *out, const Real *x, Real v1, const Real *x1,
            Real v2, const Real *x2, Real v3, const Real *x3, Real v4,
            const Real *x4) noexcept {
  for (int i = 0; i < ndim; i++) {
    out[i] = x[i] + v1 * x1[i] + v2 * x2[i] + v3 * x3[i] + v4 * x4[i];
  }
}
template <typename Real>
void sumofp(int ndim, Real *out, const Real *x, Real v1, const Real *x1,
            Real v2, const Real *x2, Real v3, const Real *x3, Real v4,
            const Real *x4, Real v5, const Real *x5) noexcept {
  for (int i = 0; i < ndim; i++) {
    out[i] =
        x[i] + v1 * x1[i] + v2 * x2[i] + v3 * x3[i] + v4 * x4[i] + v5 * x5[i];
  }
}

#ifdef MATH_AVX2
#include <immintrin.h>

template <>
void sumofp(int ndim, double *out, double v1, const double *x1) noexcept {
  for (int i = 0; i < ndim; i += 4) {
    __m256d out_i = _mm256_setzero_pd();
    out_i = _mm256_add_pd(
        out_i, _mm256_mul_pd(_mm256_set1_pd(v1), _mm256_load_pd(&x1[i])));

    switch (ndim - i) {
      case 3: {
        __m256i mask = _mm256_setr_epi64x(~0, ~0, ~0, 0);
        _mm256_maskstore_pd(&out[i], mask, out_i);
        break;
      }
      case 2: {
        __m256i mask = _mm256_setr_epi64x(~0, ~0, 0, 0);
        _mm256_maskstore_pd(&out[i], mask, out_i);
        break;
      }
      case 1: {
        __m256i mask = _mm256_setr_epi64x(~0, 0, 0, 0);
        _mm256_maskstore_pd(&out[i], mask, out_i);
        break;
      }
      default:
        _mm256_store_pd(&out[i], out_i);
        break;
    }
  }
}
template <>
void sumofp(int ndim, double *out, double v1, const double *x1, double v2,
            const double *x2) noexcept {
  for (int i = 0; i < ndim; i += 4) {
    __m256d out_i = _mm256_setzero_pd();
    out_i = _mm256_add_pd(
        out_i, _mm256_mul_pd(_mm256_set1_pd(v1), _mm256_load_pd(&x1[i])));
    out_i = _mm256_add_pd(
        out_i, _mm256_mul_pd(_mm256_set1_pd(v2), _mm256_load_pd(&x2[i])));

    switch (ndim - i) {
      case 3: {
        __m256i mask = _mm256_setr_epi64x(~0, ~0, ~0, 0);
        _mm256_maskstore_pd(&out[i], mask, out_i);
        break;
      }
      case 2: {
        __m256i mask = _mm256_setr_epi64x(~0, ~0, 0, 0);
        _mm256_maskstore_pd(&out[i], mask, out_i);
        break;
      }
      case 1: {
        __m256i mask = _mm256_setr_epi64x(~0, 0, 0, 0);
        _mm256_maskstore_pd(&out[i], mask, out_i);
        break;
      }
      default:
        _mm256_store_pd(&out[i], out_i);
        break;
    }
  }
}
template <>
void sumofp(int ndim, double *out, double v1, const double *x1, double v2,
            const double *x2, double v3, const double *x3) noexcept {
  for (int i = 0; i < ndim; i += 4) {
    __m256d out_i = _mm256_setzero_pd();
    out_i = _mm256_add_pd(
        out_i, _mm256_mul_pd(_mm256_set1_pd(v1), _mm256_load_pd(&x1[i])));
    out_i = _mm256_add_pd(
        out_i, _mm256_mul_pd(_mm256_set1_pd(v2), _mm256_load_pd(&x2[i])));
    out_i = _mm256_add_pd(
        out_i, _mm256_mul_pd(_mm256_set1_pd(v3), _mm256_load_pd(&x3[i])));

    switch (ndim - i) {
      case 3: {
        __m256i mask = _mm256_setr_epi64x(~0, ~0, ~0, 0);
        _mm256_maskstore_pd(&out[i], mask, out_i);
        break;
      }
      case 2: {
        __m256i mask = _mm256_setr_epi64x(~0, ~0, 0, 0);
        _mm256_maskstore_pd(&out[i], mask, out_i);
        break;
      }
      case 1: {
        __m256i mask = _mm256_setr_epi64x(~0, 0, 0, 0);
        _mm256_maskstore_pd(&out[i], mask, out_i);
        break;
      }
      default:
        _mm256_store_pd(&out[i], out_i);
        break;
    }
  }
}
template <>
void sumofp(int ndim, double *out, double v1, const double *x1, double v2,
            const double *x2, double v3, const double *x3, double v4,
            const double *x4) noexcept {
  for (int i = 0; i < ndim; i += 4) {
    __m256d out_i = _mm256_setzero_pd();
    out_i = _mm256_add_pd(
        out_i, _mm256_mul_pd(_mm256_set1_pd(v1), _mm256_load_pd(&x1[i])));
    out_i = _mm256_add_pd(
        out_i, _mm256_mul_pd(_mm256_set1_pd(v2), _mm256_load_pd(&x2[i])));
    out_i = _mm256_add_pd(
        out_i, _mm256_mul_pd(_mm256_set1_pd(v3), _mm256_load_pd(&x3[i])));
    out_i = _mm256_add_pd(
        out_i, _mm256_mul_pd(_mm256_set1_pd(v4), _mm256_load_pd(&x4[i])));

    switch (ndim - i) {
      case 3: {
        __m256i mask = _mm256_setr_epi64x(~0, ~0, ~0, 0);
        _mm256_maskstore_pd(&out[i], mask, out_i);
        break;
      }
      case 2: {
        __m256i mask = _mm256_setr_epi64x(~0, ~0, 0, 0);
        _mm256_maskstore_pd(&out[i], mask, out_i);
        break;
      }
      case 1: {
        __m256i mask = _mm256_setr_epi64x(~0, 0, 0, 0);
        _mm256_maskstore_pd(&out[i], mask, out_i);
        break;
      }
      default:
        _mm256_store_pd(&out[i], out_i);
        break;
    }
  }
}
template <>
void sumofp(int ndim, double *out, double v1, const double *x1, double v2,
            const double *x2, double v3, const double *x3, double v4,
            const double *x4, double v5, const double *x5) noexcept {
  for (int i = 0; i < ndim; i += 4) {
    __m256d out_i = _mm256_setzero_pd();
    out_i = _mm256_add_pd(
        out_i, _mm256_mul_pd(_mm256_set1_pd(v1), _mm256_load_pd(&x1[i])));
    out_i = _mm256_add_pd(
        out_i, _mm256_mul_pd(_mm256_set1_pd(v2), _mm256_load_pd(&x2[i])));
    out_i = _mm256_add_pd(
        out_i, _mm256_mul_pd(_mm256_set1_pd(v3), _mm256_load_pd(&x3[i])));
    out_i = _mm256_add_pd(
        out_i, _mm256_mul_pd(_mm256_set1_pd(v4), _mm256_load_pd(&x4[i])));
    out_i = _mm256_add_pd(
        out_i, _mm256_mul_pd(_mm256_set1_pd(v5), _mm256_load_pd(&x5[i])));

    switch (ndim - i) {
      case 3: {
        __m256i mask = _mm256_setr_epi64x(~0, ~0, ~0, 0);
        _mm256_maskstore_pd(&out[i], mask, out_i);
        break;
      }
      case 2: {
        __m256i mask = _mm256_setr_epi64x(~0, ~0, 0, 0);
        _mm256_maskstore_pd(&out[i], mask, out_i);
        break;
      }
      case 1: {
        __m256i mask = _mm256_setr_epi64x(~0, 0, 0, 0);
        _mm256_maskstore_pd(&out[i], mask, out_i);
        break;
      }
      default:
        _mm256_store_pd(&out[i], out_i);
        break;
    }
  }
}
template <>
void sumofp(int ndim, double *out, const double *x, double v1, const double *x1,
            double v2, const double *x2) noexcept {
  for (int i = 0; i < ndim; i += 4) {
    __m256d out_i = _mm256_load_pd(&x[i]);
    out_i = _mm256_add_pd(
        out_i, _mm256_mul_pd(_mm256_set1_pd(v1), _mm256_load_pd(&x1[i])));
    out_i = _mm256_add_pd(
        out_i, _mm256_mul_pd(_mm256_set1_pd(v2), _mm256_load_pd(&x2[i])));

    switch (ndim - i) {
      case 3: {
        __m256i mask = _mm256_setr_epi64x(~0, ~0, ~0, 0);
        _mm256_maskstore_pd(&out[i], mask, out_i);
        break;
      }
      case 2: {
        __m256i mask = _mm256_setr_epi64x(~0, ~0, 0, 0);
        _mm256_maskstore_pd(&out[i], mask, out_i);
        break;
      }
      case 1: {
        __m256i mask = _mm256_setr_epi64x(~0, 0, 0, 0);
        _mm256_maskstore_pd(&out[i], mask, out_i);
        break;
      }
      default:
        _mm256_store_pd(&out[i], out_i);
        break;
    }
  }
}
template <>
void sumofp(int ndim, double *out, const double *x, double v1, const double *x1,
            double v2, const double *x2, double v3, const double *x3) noexcept {
  for (int i = 0; i < ndim; i += 4) {
    __m256d out_i = _mm256_load_pd(&x[i]);
    out_i = _mm256_add_pd(
        out_i, _mm256_mul_pd(_mm256_set1_pd(v1), _mm256_load_pd(&x1[i])));
    out_i = _mm256_add_pd(
        out_i, _mm256_mul_pd(_mm256_set1_pd(v2), _mm256_load_pd(&x2[i])));
    out_i = _mm256_add_pd(
        out_i, _mm256_mul_pd(_mm256_set1_pd(v3), _mm256_load_pd(&x3[i])));

    switch (ndim - i) {
      case 3: {
        __m256i mask = _mm256_setr_epi64x(~0, ~0, ~0, 0);
        _mm256_maskstore_pd(&out[i], mask, out_i);
        break;
      }
      case 2: {
        __m256i mask = _mm256_setr_epi64x(~0, ~0, 0, 0);
        _mm256_maskstore_pd(&out[i], mask, out_i);
        break;
      }
      case 1: {
        __m256i mask = _mm256_setr_epi64x(~0, 0, 0, 0);
        _mm256_maskstore_pd(&out[i], mask, out_i);
        break;
      }
      default:
        _mm256_store_pd(&out[i], out_i);
        break;
    }
  }
}
template <>
void sumofp(int ndim, double *out, const double *x, double v1, const double *x1,
            double v2, const double *x2, double v3, const double *x3, double v4,
            const double *x4) noexcept {
  for (int i = 0; i < ndim; i += 4) {
    __m256d out_i = _mm256_load_pd(&x[i]);
    out_i = _mm256_add_pd(
        out_i, _mm256_mul_pd(_mm256_set1_pd(v1), _mm256_load_pd(&x1[i])));
    out_i = _mm256_add_pd(
        out_i, _mm256_mul_pd(_mm256_set1_pd(v2), _mm256_load_pd(&x2[i])));
    out_i = _mm256_add_pd(
        out_i, _mm256_mul_pd(_mm256_set1_pd(v3), _mm256_load_pd(&x3[i])));
    out_i = _mm256_add_pd(
        out_i, _mm256_mul_pd(_mm256_set1_pd(v4), _mm256_load_pd(&x4[i])));

    switch (ndim - i) {
      case 3: {
        __m256i mask = _mm256_setr_epi64x(~0, ~0, ~0, 0);
        _mm256_maskstore_pd(&out[i], mask, out_i);
        break;
      }
      case 2: {
        __m256i mask = _mm256_setr_epi64x(~0, ~0, 0, 0);
        _mm256_maskstore_pd(&out[i], mask, out_i);
        break;
      }
      case 1: {
        __m256i mask = _mm256_setr_epi64x(~0, 0, 0, 0);
        _mm256_maskstore_pd(&out[i], mask, out_i);
        break;
      }
      default:
        _mm256_store_pd(&out[i], out_i);
        break;
    }
  }
}
template <>
void sumofp(int ndim, double *out, const double *x, double v1, const double *x1,
            double v2, const double *x2, double v3, const double *x3, double v4,
            const double *x4, double v5, const double *x5) noexcept {
  for (int i = 0; i < ndim; i += 4) {
    __m256d out_i = _mm256_load_pd(&x[i]);
    out_i = _mm256_add_pd(
        out_i, _mm256_mul_pd(_mm256_set1_pd(v1), _mm256_load_pd(&x1[i])));
    out_i = _mm256_add_pd(
        out_i, _mm256_mul_pd(_mm256_set1_pd(v2), _mm256_load_pd(&x2[i])));
    out_i = _mm256_add_pd(
        out_i, _mm256_mul_pd(_mm256_set1_pd(v3), _mm256_load_pd(&x3[i])));
    out_i = _mm256_add_pd(
        out_i, _mm256_mul_pd(_mm256_set1_pd(v4), _mm256_load_pd(&x4[i])));
    out_i = _mm256_add_pd(
        out_i, _mm256_mul_pd(_mm256_set1_pd(v5), _mm256_load_pd(&x5[i])));

    switch (ndim - i) {
      case 3: {
        __m256i mask = _mm256_setr_epi64x(~0, ~0, ~0, 0);
        _mm256_maskstore_pd(&out[i], mask, out_i);
        break;
      }
      case 2: {
        __m256i mask = _mm256_setr_epi64x(~0, ~0, 0, 0);
        _mm256_maskstore_pd(&out[i], mask, out_i);
        break;
      }
      case 1: {
        __m256i mask = _mm256_setr_epi64x(~0, 0, 0, 0);
        _mm256_maskstore_pd(&out[i], mask, out_i);
        break;
      }
      default:
        _mm256_store_pd(&out[i], out_i);
        break;
    }
  }
}

#endif