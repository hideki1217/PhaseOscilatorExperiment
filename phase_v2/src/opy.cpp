#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <order.hpp>

namespace py = pybind11;
using namespace lib;

int add(int a, int b) { return a + b; }

PYBIND11_MODULE(opy, m) {
  using Real = double;
  py::class_<order::OrderEvaluator<Real>>(m, "OrderEvaluator")
      .def(py::init<int, Real, Real, int, int>(), py::arg("window"),
           py::arg("epsilon"), py::arg("dt"), py::arg("max_iteration"),
           py::arg("ndim"))
      .def(
          "eval",
          [](order::OrderEvaluator<Real>& model, py::array_t<Real> K,
             py::array_t<Real> w) {
            assert(model.ndim * model.ndim == K.size());
            assert(model.ndim == w.size());
            return model.eval(K.data(), w.data());
          },
          "Evaluate phase order parameter of a specified oscilator "
          "network\nAnd return the status flag",
          py::arg("K"), py::arg("w"))
      .def("result", &order::OrderEvaluator<Real>::result)
      .def_readonly("window", &order::OrderEvaluator<Real>::window)
      .def_readonly("epsilon", &order::OrderEvaluator<Real>::epsilon)
      .def_readonly("dt", &order::OrderEvaluator<Real>::dt)
      .def_readonly("max_iteration",
                    &order::OrderEvaluator<Real>::max_iteration)
      .def_readonly("ndim", &order::OrderEvaluator<Real>::ndim);
  py::enum_<order::EvalStatus>(m, "EvalStatus")
      .value("Ok", order::EvalStatus::Ok)
      .value("NotConverged", order::EvalStatus::NotConverged);

  m.def("add", &add, "add two numbers");
}