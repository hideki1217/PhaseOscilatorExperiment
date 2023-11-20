#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <mcmc.hpp>
#include <opy.hpp>
#include <order.hpp>

namespace py = pybind11;
using namespace lib;
using Real = double;

template <typename Order, typename Module>
void export_rk45(Module& m, const char* name) {
  using TARGET = OrderEvaluatorRK45<Real, Order>;
  py::class_<TARGET>(m, name)
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
}

template <typename Order, typename Module>
void export_mcmc(Module& m, const char* name) {
  using TARGET = mcmc::BolzmanMarkovChain<Order>;
  py::class_<TARGET>(m, name)
      .def(py::init([](py::array_t<Real> w, py::array_t<Real> K, Real threshold,
                       Real beta, Real scale, int seed) {
        assert(K.size() == w.size() * w.size());
        const int ndim = w.size();
        return TARGET(ndim, w.data(), K.data(), threshold, beta, scale, seed);
      }))
      .def("step", [](TARGET& self) { return static_cast<int>(self.step()); })
      .def("state",
           [](TARGET& self) {
             return py::array_t<Real>(self.ndim * self.ndim, self.connection());
           })
      .def("energy", &TARGET::energy)
      .def("try_swap", &TARGET::try_swap);
}

template <typename Order, typename Module>
void export_reprica_mcmc(Module& m, const char* name) {
  using TARGET = mcmc::RepricaMCMC<Order>;
  py::class_<TARGET>(m, name)
      .def(py::init([](py::array_t<Real> w, py::array_t<Real> K, Real threshold,
                       py::array_t<Real> betas, py::array_t<Real> scales,
                       int seed) {
        assert(K.size() == w.size() * w.size());
        assert(betas.size() == scales.size());

        const int ndim = w.size();
        const int num_reprica = betas.size();
        return TARGET(ndim, w.data(), K.data(), threshold, num_reprica,
                      betas.data(), scales.data(), seed);
      }))
      .def("step", &TARGET::step)
      .def("exchange",
           [](TARGET& self) {
             const auto result = self.exchange();
             return py::make_tuple(result.target, result.occured);
           })
      .def("__getitem__", &TARGET::operator[]);
}

PYBIND11_MODULE(opy, m) {
  py::enum_<EvalStatus>(m, "EvalStatus")
      .value("Ok", EvalStatus::Ok)
      .value("NotConverged", EvalStatus::NotConverged);

#define TARGET OrderEvaluatorRK4<Real, order::Kuramoto<Real>>
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

#define TARGET OrderEvaluatorRK4<Real, order::ZeroFreqRate<Real>>
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
#define TARGET OrderEvaluatorRK4<Real, order::ZeroFreqMean<Real>>
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

  export_rk45<order::Kuramoto<Real>>(m, "KuramotoEvaluatorRK45");
  export_rk45<order::ZeroFreqRate<Real>>(m, "ZeroFreqRateEvaluatorRK45");
  export_rk45<order::ZeroFreqMean<Real>>(m, "ZeroFreqMeanEvaluatorRK45");
  export_rk45<order::RelativeKuramoto<Real>>(m,
                                             "RelativeKuramotoEvaluatorRK45");
  export_rk45<order::NumOfAvgFreqMode<Real>>(m, "NumOfAvgFreqMode_RK45");

  export_mcmc<order::Kuramoto<Real>>(m, "Kuramoto_MCMC");
  export_mcmc<order::RelativeKuramoto<Real>>(m, "RelativeKuramoto_MCMC");
  export_mcmc<order::NumOfAvgFreqMode<Real>>(m, "NumOfAvgFreqMode_MCMC");

  export_reprica_mcmc<order::Kuramoto<Real>>(m, "Kuramoto_RepricaMCMC");
  export_reprica_mcmc<order::RelativeKuramoto<Real>>(
      m, "RelativeKuramoto_RepricaMCMC");
  export_reprica_mcmc<order::NumOfAvgFreqMode<Real>>(
      m, "NumOfAvgFreqMode_RepricaMCMC");
}