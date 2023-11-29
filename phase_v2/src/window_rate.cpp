#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <order.hpp>
#include <sim.hpp>

namespace py = pybind11;
using namespace lib;

template <typename Real = double>
class OrderChain {
 public:
  const int window;
  const Real dt;
  const int ndim;
  OrderChain(int window, Real dt, int ndim)
      : window(window),
        dt(dt),
        ndim(ndim),
        avg(window, ndim),
        s(ndim),
        ds_dt(ndim),
        sim_engine(ndim, dt) {}

  void eval(int iteration, Real *out, const Real *K, const Real *w) {
    std::fill(s.begin(), s.end(), 0);
    Real t = 0;

    for (int e = 0; e < iteration; e++) {
      for (int i = 0; i < window; i++) {
        const auto result = sim_engine.advance(dt, t, &s[0], K, ndim, w);
        t = result.t;
        sim::target_model(ndim, K, ndim, w, t, &s[0], &ds_dt[0]);
        avg.push(&s[0], &ds_dt[0]);
      }

      out[e] = avg.value();
    }
  }

 private:
  order::Kuramoto<Real> avg;

  std::vector<Real> s;
  std::vector<Real> ds_dt;
  sim::RK4<Real> sim_engine;
};

PYBIND11_MODULE(window_rate, m) {
  using Real = double;
  py::class_<OrderChain<Real>>(m, "OrderChain")
      .def(py::init<int, Real, int>(), py::arg("window"), py::arg("dt"),
           py::arg("ndim"))
      .def(
          "eval",
          [](OrderChain<Real> &model, int iteration, py::array_t<Real> K,
             py::array_t<Real> w) -> py::array_t<Real> {
            assert(model.ndim * model.ndim == K.size());
            assert(model.ndim == w.size());
            py::array_t<Real> res(iteration);
            model.eval(iteration, res.mutable_data(), K.data(), w.data());
            return res;
          },
          "Evaluate phase order parameter of a specified oscilator "
          "network\nAnd return the status flag",
          py::arg("iteration"), py::arg("K"), py::arg("w"));
}
