#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <opy.hpp>

namespace py = pybind11;
using namespace lib;

int add(int a, int b) { return a + b; }

PYBIND11_MODULE(opy, m) {
  using Real = double;
  py::class_<OrderEvaluator<Real>>(m, "OrderEvaluator")
      .def(py::init<int, Real, Real, int, int>(), py::arg("window"),
           py::arg("epsilon"), py::arg("dt"), py::arg("max_iteration"),
           py::arg("ndim"))
      .def(
          "eval",
          [](OrderEvaluator<Real>& model, py::array_t<Real> K,
             py::array_t<Real> w) {
            assert(model.ndim * model.ndim == K.size());
            assert(model.ndim == w.size());
            return model.eval(K.data(), w.data());
          },
          "Evaluate phase order parameter of a specified oscilator "
          "network\nAnd return the status flag",
          py::arg("K"), py::arg("w"))
      .def("result", &OrderEvaluator<Real>::result)
      .def_readonly("window", &OrderEvaluator<Real>::window)
      .def_readonly("epsilon", &OrderEvaluator<Real>::epsilon)
      .def_readonly("dt", &OrderEvaluator<Real>::dt)
      .def_readonly("max_iteration", &OrderEvaluator<Real>::max_iteration)
      .def_readonly("ndim", &OrderEvaluator<Real>::ndim);
  py::enum_<EvalStatus>(m, "EvalStatus")
      .value("Ok", EvalStatus::Ok)
      .value("NotConverged", EvalStatus::NotConverged);

  m.def("add", &add, "add two numbers");
}