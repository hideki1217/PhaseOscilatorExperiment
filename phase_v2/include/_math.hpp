

template <typename Real>
void sumofp(int ndim, Real *out, Real v1, const Real *x1) {
  for (int i = 0; i < ndim; i++) {
    out[i] = v1 * x1[i];
  }
}
template <typename Real>
void sumofp(int ndim, Real *out, Real v1, const Real *x1, Real v2,
            const Real *x2) {
  for (int i = 0; i < ndim; i++) {
    out[i] = v1 * x1[i] + v2 * x2[i];
  }
}
template <typename Real>
void sumofp(int ndim, Real *out, Real v1, const Real *x1, Real v2,
            const Real *x2, Real v3, const Real *x3) {
  for (int i = 0; i < ndim; i++) {
    out[i] = v1 * x1[i] + v2 * x2[i] + v3 * x3[i];
  }
}
template <typename Real>
void sumofp(int ndim, Real *out, Real v1, const Real *x1, Real v2,
            const Real *x2, Real v3, const Real *x3, Real v4, const Real *x4) {
  for (int i = 0; i < ndim; i++) {
    out[i] = v1 * x1[i] + v2 * x2[i] + v3 * x3[i] + v4 * x4[i];
  }
}
template <typename Real>
void sumofp(int ndim, Real *out, Real v1, const Real *x1, Real v2,
            const Real *x2, Real v3, const Real *x3, Real v4, const Real *x4,
            Real v5, const Real *x5) {
  for (int i = 0; i < ndim; i++) {
    out[i] = v1 * x1[i] + v2 * x2[i] + v3 * x3[i] + v4 * x4[i] + v5 * x5[i];
  }
}

template <typename Real>
void sumofp(int ndim, Real *out, Real *x, Real v1, const Real *x1) {
  for (int i = 0; i < ndim; i++) {
    out[i] = x[i] + v1 * x1[i];
  }
}
template <typename Real>
void sumofp(int ndim, Real *out, const Real *x, Real v1, const Real *x1,
            Real v2, const Real *x2) {
  for (int i = 0; i < ndim; i++) {
    out[i] = x[i] + v1 * x1[i] + v2 * x2[i];
  }
}
template <typename Real>
void sumofp(int ndim, Real *out, const Real *x, Real v1, const Real *x1,
            Real v2, const Real *x2, Real v3, const Real *x3) {
  for (int i = 0; i < ndim; i++) {
    out[i] = x[i] + v1 * x1[i] + v2 * x2[i] + v3 * x3[i];
  }
}
template <typename Real>
void sumofp(int ndim, Real *out, const Real *x, Real v1, const Real *x1,
            Real v2, const Real *x2, Real v3, const Real *x3, Real v4,
            const Real *x4) {
  for (int i = 0; i < ndim; i++) {
    out[i] = x[i] + v1 * x1[i] + v2 * x2[i] + v3 * x3[i] + v4 * x4[i];
  }
}
template <typename Real>
void sumofp(int ndim, Real *out, const Real *x, Real v1, const Real *x1,
            Real v2, const Real *x2, Real v3, const Real *x3, Real v4,
            const Real *x4, Real v5, const Real *x5) {
  for (int i = 0; i < ndim; i++) {
    out[i] =
        x[i] + v1 * x1[i] + v2 * x2[i] + v3 * x3[i] + v4 * x4[i] + v5 * x5[i];
  }
}