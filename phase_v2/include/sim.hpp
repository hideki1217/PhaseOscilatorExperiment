#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace lib {
namespace sim {

template <typename Real>
static void target_model(int ndim, const Real *K, const Real *w, const Real t,
                         const Real *s, Real *ds_dt) {
  for (int i = 0; i < ndim; i++) {
    ds_dt[i] = w[i];
  }

  for (int i = 0; i < ndim; i++) {
    for (int j = 0; j < ndim; j++) {
      ds_dt[i] += K[i * ndim + j] * std::sin(s[j] - s[i]);
    }
  }
}

template <typename Real>
class RK4 {
  struct Result {
    Real t;
  };

 public:
  const int ndim;
  const Real max_dt;

  RK4(const int ndim, const Real max_dt) : ndim(ndim), max_dt(max_dt) {
    _s.resize(ndim);
    k1.resize(ndim);
    k2.resize(ndim);
    k3.resize(ndim);
    k4.resize(ndim);
  }

  Result advance(Real T, Real t, Real *s, const Real *K, const Real *w) {
    const Real t_max = t + T;

    while (t < t_max) {
      const Real h = (t + max_dt <= t_max) ? max_dt : t_max - t;
      t = _advance_dt(h, t, s, K, w);
    }
    assert(std::abs(t - t_max) < 1e-6);

    return {t};
  }

 private:
  Real _advance_dt(const Real dt, const Real t, Real *s, const Real *K,
                   const Real *w) {
    const Real dt_2 = dt * 0.5;
    const Real dt_6 = dt / 6;

    target_model(ndim, K, w, t, &s[0], &k1[0]);
    for (int i = 0; i < ndim; i++) _s[i] = s[i] + dt_2 * k1[i];
    target_model(ndim, K, w, t + dt_2, &_s[0], &k2[0]);
    for (int i = 0; i < ndim; i++) _s[i] = s[i] + dt_2 * k2[i];
    target_model(ndim, K, w, t + dt_2, &_s[0], &k3[0]);
    for (int i = 0; i < ndim; i++) _s[i] = s[i] + dt * k3[i];
    target_model(ndim, K, w, t + dt, &_s[0], &k4[0]);
    for (int i = 0; i < ndim; i++) {
      s[i] += dt_6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    }

    return t + dt;
  }

  std::vector<Real> _s;
  std::vector<Real> k1, k2, k3, k4;
};

template <typename Real>
class FehlbergRK45 {
  struct Result {
    Real t;
    int iteration;
  };

 public:
  const Real atol;
  const Real first_h, max_h;
  const int ndim;
  FehlbergRK45(int ndim, Real first_h, Real max_h, Real atol = 1e-3)
      : atol(atol),
        first_h(first_h),
        max_h(max_h),
        ndim(ndim),
        tmp(ndim),
        _k(ndim * 6) {
    for (int i = 0; i < 6; i++) k[i] = &_k[i * ndim];
  }

  /**
   * advance T time from (t, s) on the model.
   */
  Result advance(Real T, Real t, Real *s, const Real *K, const Real *w) {
    const Real t_max = t + T;
    h = first_h;

    int iteration = 0;
    while (t < t_max) {
      iteration++;
      const auto h = try_advance(t, s, K, w, t_max);
      if (h > 0) {
        t += h;
      }
    }

    return {t, iteration};
  }

 private:
  Real norm(int size, const Real *v) { return *std::max_element(v, v + size); }
  Real try_advance(Real t, Real *s, const Real *K, const Real *w, Real t_max) {
    /**
     * reference: https://slpr.sakura.ne.jp/qp/runge-kutta-ex/#rkf45hauto
     */
    assert(t < t_max);
    assert(h > 0);

    static constexpr Real c_0 = 1. / 4, a_0[] = {1. / 4};
    static constexpr Real c_1 = 3. / 8, a_1[] = {3. / 32, 9. / 32};
    static constexpr Real c_2 = 12. / 13,
                          a_2[] = {1932. / 2197, -7200. / 2197, 7296. / 2197};
    static constexpr Real c_3 = 1.,
                          a_3[] = {439. / 216, -8., 3680. / 513, -845. / 4104};
    static constexpr Real c_4 = 1. / 2, a_4[] = {-8. / 27, 2, -3544. / 2565,
                                                 1859. / 4104, -11. / 40};
    static constexpr Real b[] = {1. / 360, -128. / 4275, -2197. / 75240,
                                 1. / 50, 2. / 55};
    static constexpr Real d[] = {25. / 216, 1408. / 2565, 2197. / 4104,
                                 -1. / 5};

    target_model(ndim, K, w, t, s, k[0]);

    for (int i = 0; i < ndim; i++) tmp[i] = s[i] + h * (a_0[0] * k[0][i]);
    target_model(ndim, K, w, t + c_0 * h, &tmp[0], k[1]);

    for (int i = 0; i < ndim; i++)
      tmp[i] = s[i] + h * (a_1[0] * k[0][i] + a_1[1] * k[1][i]);
    target_model(ndim, K, w, t + c_1 * h, &tmp[0], k[2]);

    for (int i = 0; i < ndim; i++)
      tmp[i] =
          s[i] + h * (a_2[0] * k[0][i] + a_2[1] * k[1][i] + a_2[2] * k[2][i]);
    target_model(ndim, K, w, t + c_2 * h, &tmp[0], k[3]);

    for (int i = 0; i < ndim; i++)
      tmp[i] = s[i] + h * (a_3[0] * k[0][i] + a_3[1] * k[1][i] +
                           a_3[2] * k[2][i] + a_3[3] * k[3][i]);
    target_model(ndim, K, w, t + c_3 * h, &tmp[0], k[4]);

    for (int i = 0; i < ndim; i++)
      tmp[i] =
          s[i] + h * (a_4[0] * k[0][i] + a_4[1] * k[1][i] + a_4[2] * k[2][i] +
                      a_4[3] * k[3][i] + a_4[4] * k[4][i]);
    target_model(ndim, K, w, t + c_4 * h, &tmp[0], k[5]);

    for (int i = 0; i < ndim; i++) {
      tmp[i] = b[0] * k[0][i] + b[1] * k[1][i] + b[2] * k[2][i] +
               b[3] * k[3][i] + b[4] * k[4][i];
    }
    const auto R = norm(ndim, &tmp[0]);
    const auto R_base = atol;

    // Try to update state
    Real dt = 0;
    // std::cout << "    R = " << R << " h =" << h << " t = " << t << std::endl;
    if (R < R_base) {
      t += (dt = h);
      for (int i = 0; i < ndim; i++) {
        // NOTE: this index is misaligned but intended.
        s[i] += h * (d[0] * k[0][i] + d[1] * k[2][i] + d[2] * k[3][i] +
                     d[3] * k[4][i]);
      }
    }

    const auto delta = std::pow(atol / (2 * R), 0.25);
    h *= (delta <= 0.1) ? 0.1 : (delta >= 4) ? 4 : delta;
    h = std::min(h, max_h);
    if (t + h > t_max) {
      h = t_max - t;
    }

    reliability = R;
    return dt;
  }
  Real h;
  Real reliability;
  std::vector<Real> tmp;
  std::vector<Real> _k;
  Real *k[6];
};
}  // namespace sim
}  // namespace lib