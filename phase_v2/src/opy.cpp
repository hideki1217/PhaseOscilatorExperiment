#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <opy.hpp>

namespace py = pybind11;
using namespace lib;

PYBIND11_MODULE(opy, m) {
  using Real = double;
  py::enum_<EvalStatus>(m, "EvalStatus")
      .value("Ok", EvalStatus::Ok)
      .value("NotConverged", EvalStatus::NotConverged);

  py::class_<OrderEvaluatorRK4<Real>>(m, "OrderEvaluatorRK4")
      .def(py::init<int, Real, Real, int, int, Real>(), py::arg("window"),
           py::arg("epsilon"), py::arg("sampling_dt"), py::arg("max_iteration"),
           py::arg("ndim"), py::arg("update_dt"))
      .def(
          "eval",
          [](OrderEvaluatorRK4<Real>& model, py::array_t<Real> K,
             py::array_t<Real> w) {
            assert(model.ndim * model.ndim == K.size());
            assert(model.ndim == w.size());
            return model.eval(K.data(), w.data());
          },
          "Evaluate phase order parameter of a specified oscilator "
          "network\nAnd return the status flag",
          py::arg("K"), py::arg("w"))
      .def("result", &OrderEvaluatorRK4<Real>::result)
      .def_readonly("window", &OrderEvaluatorRK4<Real>::window)
      .def_readonly("epsilon", &OrderEvaluatorRK4<Real>::epsilon)
      .def_readonly("sampling_dt", &OrderEvaluatorRK4<Real>::sampling_dt)
      .def_readonly("max_iteration", &OrderEvaluatorRK4<Real>::max_iteration)
      .def_readonly("ndim", &OrderEvaluatorRK4<Real>::ndim);

  py::class_<OrderEvaluatorRK45<Real>>(m, "OrderEvaluatorRK45")
      .def(py::init<int, Real, Real, int, int, Real, Real, Real>(),
           py::arg("window"), py::arg("epsilon"), py::arg("sampling_dt"),
           py::arg("max_iteration"), py::arg("ndim"), py::arg("start_dt"),
           py::arg("max_dt"), py::arg("atol"))
      .def(
          "eval",
          [](OrderEvaluatorRK4<Real>& model, py::array_t<Real> K,
             py::array_t<Real> w) {
            assert(model.ndim * model.ndim == K.size());
            assert(model.ndim == w.size());
            return model.eval(K.data(), w.data());
          },
          "Evaluate phase order parameter of a specified oscilator "
          "network\nAnd return the status flag",
          py::arg("K"), py::arg("w"))
      .def("result", &OrderEvaluatorRK45<Real>::result)
      .def_readonly("window", &OrderEvaluatorRK45<Real>::window)
      .def_readonly("epsilon", &OrderEvaluatorRK45<Real>::epsilon)
      .def_readonly("sampling_dt", &OrderEvaluatorRK45<Real>::sampling_dt)
      .def_readonly("max_iteration", &OrderEvaluatorRK45<Real>::max_iteration)
      .def_readonly("ndim", &OrderEvaluatorRK45<Real>::ndim);
}