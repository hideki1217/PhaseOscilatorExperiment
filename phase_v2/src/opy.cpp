#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <opy.hpp>
#include <order.hpp>

namespace py = pybind11;
using namespace lib;

PYBIND11_MODULE(opy, m) {
  using Real = double;
  py::enum_<EvalStatus>(m, "EvalStatus")
      .value("Ok", EvalStatus::Ok)
      .value("NotConverged", EvalStatus::NotConverged);

#define TARGET OrderEvaluatorRK4<Real, order::KuramotoFixed<Real>>
#define TARGET_NAME "KuramotoOrderEvaluatorRK4"
  py::class_<TARGET>(m, TARGET_NAME)
      .def(py::init<int, Real, Real, int, int, Real>(), py::arg("window"),
           py::arg("epsilon"), py::arg("sampling_dt"), py::arg("max_iteration"),
           py::arg("ndim"), py::arg("update_dt"))
      .def(
          "eval",
          [](TARGET& model, py::array_t<Real> K, py::array_t<Real> w) {
            assert(model.ndim * model.ndim == K.size());
            assert(model.ndim == w.size());
            return model.eval(K.data(), w.data());
          },
          "Evaluate phase order parameter of a specified oscilator "
          "network\nAnd return the status flag",
          py::arg("K"), py::arg("w"))
      .def("result", &TARGET::result)
      .def_readonly("window", &TARGET::window)
      .def_readonly("epsilon", &TARGET::epsilon)
      .def_readonly("sampling_dt", &TARGET::sampling_dt)
      .def_readonly("max_iteration", &TARGET::max_iteration)
      .def_readonly("ndim", &TARGET::ndim);
#undef TARGET
#undef TARGET_NAME

#define TARGET OrderEvaluatorRK4<Real, order::ZeroFreqRateFixed<Real>>
#define TARGET_NAME "ZeroFreqRateEvaluatorRK4"
  py::class_<TARGET>(m, TARGET_NAME)
      .def(py::init<int, Real, Real, int, int, Real>(), py::arg("window"),
           py::arg("epsilon"), py::arg("sampling_dt"), py::arg("max_iteration"),
           py::arg("ndim"), py::arg("update_dt"))
      .def(
          "eval",
          [](TARGET& model, py::array_t<Real> K, py::array_t<Real> w) {
            assert(model.ndim * model.ndim == K.size());
            assert(model.ndim == w.size());
            return model.eval(K.data(), w.data());
          },
          "Evaluate phase order parameter of a specified oscilator "
          "network\nAnd return the status flag",
          py::arg("K"), py::arg("w"))
      .def("result", &TARGET::result)
      .def_readonly("window", &TARGET::window)
      .def_readonly("epsilon", &TARGET::epsilon)
      .def_readonly("sampling_dt", &TARGET::sampling_dt)
      .def_readonly("max_iteration", &TARGET::max_iteration)
      .def_readonly("ndim", &TARGET::ndim);
#undef TARGET
#undef TARGET_NAME
#define TARGET OrderEvaluatorRK4<Real, order::ZeroFreqMeanFixed<Real>>
#define TARGET_NAME "ZeroFreqMeanOrderEvaluatorRK4"
  py::class_<TARGET>(m, TARGET_NAME)
      .def(py::init<int, Real, Real, int, int, Real>(), py::arg("window"),
           py::arg("epsilon"), py::arg("sampling_dt"), py::arg("max_iteration"),
           py::arg("ndim"), py::arg("update_dt"))
      .def(
          "eval",
          [](TARGET& model, py::array_t<Real> K, py::array_t<Real> w) {
            assert(model.ndim * model.ndim == K.size());
            assert(model.ndim == w.size());
            return model.eval(K.data(), w.data());
          },
          "Evaluate phase order parameter of a specified oscilator "
          "network\nAnd return the status flag",
          py::arg("K"), py::arg("w"))
      .def("result", &TARGET::result)
      .def_readonly("window", &TARGET::window)
      .def_readonly("epsilon", &TARGET::epsilon)
      .def_readonly("sampling_dt", &TARGET::sampling_dt)
      .def_readonly("max_iteration", &TARGET::max_iteration)
      .def_readonly("ndim", &TARGET::ndim);
#undef TARGET
#undef TARGET_NAME

#define TARGET OrderEvaluatorRK45<Real, order::KuramotoFixed<Real>>
#define TARGET_NAME "KuramotoOrderEvaluatorRK45"
  py::class_<TARGET>(m, TARGET_NAME)
      .def(py::init<int, Real, Real, int, int, Real, Real, Real>(),
           py::arg("window"), py::arg("epsilon"), py::arg("sampling_dt"),
           py::arg("max_iteration"), py::arg("ndim"), py::arg("start_dt"),
           py::arg("max_dt"), py::arg("atol"))
      .def(
          "eval",
          [](TARGET& model, py::array_t<Real> K, py::array_t<Real> w) {
            assert(model.ndim * model.ndim == K.size());
            assert(model.ndim == w.size());
            return model.eval(K.data(), w.data());
          },
          "Evaluate phase order parameter of a specified oscilator "
          "network\nAnd return the status flag",
          py::arg("K"), py::arg("w"))
      .def("result", &TARGET::result)
      .def_readonly("window", &TARGET::window)
      .def_readonly("epsilon", &TARGET::epsilon)
      .def_readonly("sampling_dt", &TARGET::sampling_dt)
      .def_readonly("max_iteration", &TARGET::max_iteration)
      .def_readonly("ndim", &TARGET::ndim);
#undef TARGET
#undef TARGET_NAME
#define TARGET OrderEvaluatorRK45<Real, order::ZeroFreqRateFixed<Real>>
#define TARGET_NAME "ZeroFreqRateEvaluatorRK45"
  py::class_<TARGET>(m, TARGET_NAME)
      .def(py::init<int, Real, Real, int, int, Real, Real, Real>(),
           py::arg("window"), py::arg("epsilon"), py::arg("sampling_dt"),
           py::arg("max_iteration"), py::arg("ndim"), py::arg("start_dt"),
           py::arg("max_dt"), py::arg("atol"))
      .def(
          "eval",
          [](TARGET& model, py::array_t<Real> K, py::array_t<Real> w) {
            assert(model.ndim * model.ndim == K.size());
            assert(model.ndim == w.size());
            return model.eval(K.data(), w.data());
          },
          "Evaluate phase order parameter of a specified oscilator "
          "network\nAnd return the status flag",
          py::arg("K"), py::arg("w"))
      .def("result", &TARGET::result)
      .def_readonly("window", &TARGET::window)
      .def_readonly("epsilon", &TARGET::epsilon)
      .def_readonly("sampling_dt", &TARGET::sampling_dt)
      .def_readonly("max_iteration", &TARGET::max_iteration)
      .def_readonly("ndim", &TARGET::ndim);
#undef TARGET
#undef TARGET_NAME
#define TARGET OrderEvaluatorRK45<Real, order::ZeroFreqMeanFixed<Real>>
#define TARGET_NAME "ZeroFreqMeanOrderEvaluatorRK45"
  py::class_<TARGET>(m, TARGET_NAME)
      .def(py::init<int, Real, Real, int, int, Real, Real, Real>(),
           py::arg("window"), py::arg("epsilon"), py::arg("sampling_dt"),
           py::arg("max_iteration"), py::arg("ndim"), py::arg("start_dt"),
           py::arg("max_dt"), py::arg("atol"))
      .def(
          "eval",
          [](TARGET& model, py::array_t<Real> K, py::array_t<Real> w) {
            assert(model.ndim * model.ndim == K.size());
            assert(model.ndim == w.size());
            return model.eval(K.data(), w.data());
          },
          "Evaluate phase order parameter of a specified oscilator "
          "network\nAnd return the status flag",
          py::arg("K"), py::arg("w"))
      .def("result", &TARGET::result)
      .def_readonly("window", &TARGET::window)
      .def_readonly("epsilon", &TARGET::epsilon)
      .def_readonly("sampling_dt", &TARGET::sampling_dt)
      .def_readonly("max_iteration", &TARGET::max_iteration)
      .def_readonly("ndim", &TARGET::ndim);
#undef TARGET
#undef TARGET_NAME
}