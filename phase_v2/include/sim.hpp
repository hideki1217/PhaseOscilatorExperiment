#include <vector>

namespace lib {
namespace sim {

template <typename Real>
class RK4 {
 public:
  const int ndim;
  const Real dt;

  RK4(const int ndim, const Real dt) : ndim(ndim), dt(dt) {
    _s.resize(ndim);
    k1.resize(ndim);
    k2.resize(ndim);
    k3.resize(ndim);
    k4.resize(ndim);
  }

  Real advance_dt(const Real t, Real *s, const Real *K, const Real *w) {
    const Real dt_2 = dt * 0.5;
    const Real dt_6 = dt / 6;

    time_diff(K, w, t, &s[0], &k1[0]);
    for (int i = 0; i < ndim; i++) _s[i] = s[i] + dt_2 * k1[i];
    time_diff(K, w, t + dt_2, &_s[0], &k2[0]);
    for (int i = 0; i < ndim; i++) _s[i] = s[i] + dt_2 * k2[i];
    time_diff(K, w, t + dt_2, &_s[0], &k3[0]);
    for (int i = 0; i < ndim; i++) _s[i] = s[i] + dt * k3[i];
    time_diff(K, w, t + dt, &_s[0], &k4[0]);
    for (int i = 0; i < ndim; i++) {
      s[i] += dt_6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    }

    return t + dt;
  }

  void time_diff(const Real *K, const Real *w, const Real t, const Real *s,
                 Real *ds_dt) {
    for (int i = 0; i < ndim; i++) {
      ds_dt[i] = w[i];
    }

    for (int i = 0; i < ndim; i++) {
      for (int j = 0; j < ndim; j++) {
        ds_dt[i] += K[i * ndim + j] * std::sin(s[j] - s[i]);
      }
    }
  }

 private:
  std::vector<Real> _s;
  std::vector<Real> k1, k2, k3, k4;
};

}  // namespace sim
}  // namespace lib