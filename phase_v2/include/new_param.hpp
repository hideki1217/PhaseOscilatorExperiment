#pragma once

#include <cmath>

namespace new_lib {
namespace param {
template <typename real_t>
void create_w(int ndim, real_t *w) {
  const real_t x0 = 0, r = 1.0;
  for (int i = 0; i < ndim; i++) {
    const real_t p = real_t(i + 1) / (ndim + 1);
    w[i] = x0 + r * std::tan(M_PI * (p - 0.5));
  }
}
}  // namespace param
}  // namespace new_lib