#pragma once

#include <array>
#include <cassert>

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
    system_impl(ndim, K, Kstride, w, t, &x[0], &dx[0]);
  }

 private:
  template <typename real_t>
  void system_impl(int ndim, const real_t* K, int Kstride, const real_t* w,
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
};
}  // namespace system
}  // namespace new_lib
