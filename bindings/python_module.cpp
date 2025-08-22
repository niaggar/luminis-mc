#include <luminis/core/simulation.hpp>
#include <luminis/sample/phase.hpp>
#include <luminis/log/logger.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace luminis::core;
using namespace luminis::sample;
using namespace luminis::log;

PYBIND11_MODULE(luminis_mc, m) {
  m.doc() = "Python bindings for the luminis-mc Monte Carlo core";

  py::class_<PhaseFunction>(m, "PhaseFunction");

  py::class_<UniformPhaseFunction, PhaseFunction>(m, "UniformPhaseFunction")
      .def(py::init<>())
      .def("sample", &UniformPhaseFunction::Sample, py::arg("x"), "Sample the phase function");

  py::class_<RayleighPhaseFunction, PhaseFunction>(m, "RayleighPhaseFunction")
      .def(py::init<int, double, double>(),
           py::arg("nDiv"), py::arg("minVal"), py::arg("maxVal"))
      .def("sample", &RayleighPhaseFunction::Sample, py::arg("x"))
      .def("pdf", &RayleighPhaseFunction::PDF, py::arg("x"));

  py::class_<HenyeyGreensteinPhaseFunction, PhaseFunction>(m, "HenyeyGreensteinPhaseFunction")
      .def(py::init<double>(), py::arg("g"))
      .def("sample", &HenyeyGreensteinPhaseFunction::Sample, py::arg("x"));

  py::class_<RayleighDebyePhaseFunction, PhaseFunction>(m, "RayleighDebyePhaseFunction")
      .def(py::init<double, double, int, double, double>(),
           py::arg("wavelength"), py::arg("radius"), py::arg("nDiv"), py::arg("minVal"), py::arg("maxVal"))
      .def("sample", &RayleighDebyePhaseFunction::Sample, py::arg("x"))
      .def("pdf", &RayleighDebyePhaseFunction::PDF, py::arg("x"));




  py::class_<SimConfig>(m, "SimConfig")
      .def(py::init<>())
      .def_readwrite("seed", &SimConfig::seed)
      .def_readwrite("n_photons", &SimConfig::n_photons)
      .def_readwrite("max_scatter", &SimConfig::max_scatter)
      .def_readwrite("world_z_min", &SimConfig::world_z_min)
      .def_readwrite("world_z_max", &SimConfig::world_z_max);

  py::class_<SimStats>(m, "SimStats")
      .def_readonly("emitted", &SimStats::emitted)
      .def_readonly("detected", &SimStats::detected)
      .def_property_readonly("rate", &SimStats::detection_rate);

  py::class_<Material>(m, "Material")
      .def(py::init<double>(), py::arg("mean_free_path"));

  py::class_<PlaneDetector>(m, "PlaneDetector")
      .def(py::init<double, double>(), py::arg("z_plane"), py::arg("radius"))
      .def_readonly("hits", &PlaneDetector::hits);

  py::class_<Simulation>(m, "Simulation")
      .def(py::init<SimConfig, Material, PlaneDetector>())
      .def(
          "run",
          [](Simulation &self) {
            // Release the GIL while the C++ loop runs
            py::gil_scoped_release release;
            return self.run();
          },
          "Run the simulation and return SimStats")
      .def_property_readonly("detector", &Simulation::detector,
                             py::return_value_policy::reference_internal);

  // Convenience one-shot function
  m.def(
      "run_once",
      [](std::size_t n_photons, double mfp, double z_plane, double radius,
         std::uint64_t seed) {
        SimConfig cfg;
        cfg.n_photons = n_photons;
        cfg.seed = seed;
        Material mat(mfp);
        PlaneDetector det(z_plane, radius);
        Simulation sim(cfg, mat, det);
        py::gil_scoped_release release;
        return sim.run();
      },
      py::arg("n_photons"), py::arg("mfp"), py::arg("z_plane"),
      py::arg("radius"), py::arg("seed") = 0);
}
