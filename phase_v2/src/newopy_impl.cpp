#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <new_lib.hpp>

using namespace new_lib;
namespace py = pybind11;

using real_t = system::System<2>::real_t;

template <int ndim>
using Kuramoto = order::Kuramoto<system::System<ndim>>;

template <int ndim>
using MaxAvgFreqCluster = order::MaxAvgFreqCluster<system::System<ndim>>;

#define export_evaluator(Order, Ndim)                                         \
  do {                                                                        \
    using TARGET = order::Evaluator<Order<Ndim>>;                             \
    py::class_<TARGET>(m, #Order "_" #Ndim)                                   \
        .def(py::init<int, real_t, real_t, int>(), py::arg("window"),         \
             py::arg("epsilon"), py::arg("Dt"), py::arg("max_iter"))          \
        .def(                                                                 \
            "eval",                                                           \
            [](TARGET& model, py::array_t<real_t> K, py::array_t<real_t> w) { \
              assert(model.ndim* model.ndim == K.size());                     \
              assert(model.ndim == w.size());                                 \
              const auto status = model.eval(K.data(), model.ndim, w.data()); \
              return py::make_tuple(status, model.result());                  \
            },                                                                \
            "Evaluate phase order parameter of a specified oscilator "        \
            "network\nAnd return the status flag",                            \
            py::arg("K"), py::arg("w"))                                       \
        .def("result", &TARGET::result);                                      \
  } while (0)

#define ndim_set(EXPORTER, ORDER) \
  EXPORTER(ORDER, 2);             \
  EXPORTER(ORDER, 3);             \
  EXPORTER(ORDER, 4);             \
  EXPORTER(ORDER, 5);             \
  EXPORTER(ORDER, 6);             \
  EXPORTER(ORDER, 7);             \
  EXPORTER(ORDER, 8);             \
  EXPORTER(ORDER, 9);             \
  EXPORTER(ORDER, 10);            \
  EXPORTER(ORDER, 11);            \
  EXPORTER(ORDER, 12);            \
  EXPORTER(ORDER, 13);            \
  EXPORTER(ORDER, 14);            \
  EXPORTER(ORDER, 15);            \
  EXPORTER(ORDER, 16)

#define export_single_mcmc(Order, Ndim)                                    \
  do {                                                                     \
    using TARGET = mcmc::BolzmanMarkovChain<Order<Ndim>>;                  \
    py::class_<TARGET>(m, #Order "_" #Ndim)                                \
        .def(py::init([](py::array_t<real_t> w, py::array_t<real_t> K,     \
                         real_t threshold, real_t beta, real_t scale,      \
                         int seed) {                                       \
          assert(K.size() == Ndim * Ndim);                                 \
          assert(w.size() == Ndim);                                        \
          return TARGET(w.data(), K.data(), threshold, beta, scale, seed); \
        }))                                                                \
        .def("step", &TARGET::step)                                        \
        .def("state",                                                      \
             [](TARGET& self) {                                            \
               return py::array_t<real_t>(Ndim * Ndim, self.connection()); \
             })                                                            \
        .def("energy", &TARGET::energy)                                    \
        .def("try_swap", &TARGET::try_swap);                               \
  } while (0)

#define export_reprica_mcmc(Order, Ndim)                               \
  do {                                                                 \
    using TARGET = mcmc::RepricaMCMC<Order<Ndim>>;                     \
    py::class_<TARGET>(m, #Order "_" #Ndim)                            \
        .def(py::init([](py::array_t<real_t> w, py::array_t<real_t> K, \
                         real_t threshold, py::array_t<real_t> betas,  \
                         py::array_t<real_t> scales, int seed) {       \
          assert(K.size() == Ndim * Ndim);                             \
          assert(w.size() == Ndim);                                    \
          assert(betas.size() == scales.size());                       \
                                                                       \
          const int num_reprica = betas.size();                        \
          return TARGET(w.data(), K.data(), threshold, num_reprica,    \
                        betas.data(), scales.data(), seed);            \
        }))                                                            \
        .def("step", &TARGET::step)                                    \
        .def("exchange",                                               \
             [](TARGET& self) {                                        \
               const auto result = self.exchange();                    \
               return py::make_tuple(result.target, result.occured);   \
             })                                                        \
        .def("__getitem__", &TARGET::operator[]);                      \
  } while (0)

template <typename Module>
void bind_evaluator(Module m) {
  py::enum_<order::EvalStatus>(m, "EvalStatus")
      .value("Ok", order::EvalStatus::Ok)
      .value("NotConverged", order::EvalStatus::NotConverged);
  ndim_set(export_evaluator, Kuramoto);
  ndim_set(export_evaluator, MaxAvgFreqCluster);
}

template <typename Module>
void bind_single_mcmc(Module m) {
  py::enum_<mcmc::Result>(m, "StepStatus")
      .value("Accepted", mcmc::Result::Accepted)
      .value("Rejected", mcmc::Result::Rejected)
      .value("MinusConnection", mcmc::Result::MinusConnection)
      .value("NotConverged", mcmc::Result::NotConverged)
      .value("SmallOrder", mcmc::Result::SmallOrder);
  ndim_set(export_single_mcmc, Kuramoto);
  ndim_set(export_single_mcmc, MaxAvgFreqCluster);
}

template <typename Module>
void bind_reprica_mcmc(Module m) {
  ndim_set(export_reprica_mcmc, Kuramoto);
  ndim_set(export_reprica_mcmc, MaxAvgFreqCluster);
}

PYBIND11_MODULE(newopy_impl, m) {
  bind_evaluator(m.def_submodule("evaluation"));
  bind_single_mcmc(m.def_submodule("single_mcmc"));
  bind_reprica_mcmc(m.def_submodule("reprica_mcmc"));
}