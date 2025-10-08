#include <luminis/core/detector.hpp>
#include <luminis/core/laser.hpp>
#include <luminis/core/medium.hpp>
#include <luminis/core/photon.hpp>
#include <luminis/core/simulation.hpp>
#include <luminis/core/absortion.hpp>
#include <luminis/log/logger.hpp>
#include <luminis/math/rng.hpp>
#include <luminis/sample/phase.hpp>

#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace luminis::core;
using namespace luminis::sample;
using namespace luminis::log;

PYBIND11_MODULE(luminis_mc, m) {
  m.doc() = "Python bindings for the luminis-mc Monte Carlo core";

  // Rng bindings
  py::class_<Rng>(m, "Rng")
      .def(py::init<uint64_t>(), py::arg("seed") = std::random_device{}(),
           "Initialize the RNG with an optional seed")
      .def("uniform", &Rng::uniform,
           "Generate a uniform random number in [0, 1)")
      .def("normal", &Rng::normal, py::arg("mean"), py::arg("stddev"),
           "Generate a normally distributed random number with given mean and "
           "stddev");

  // Phase function bindings
  py::class_<PhaseFunction>(m, "PhaseFunction")
      .def("sample", &PhaseFunction::sample_cos, py::arg("x"),
           "Sample the cosine of the scattering angle using a uniform random number x in [0, 1]")
      .def("sample_theta", &PhaseFunction::sample_theta, py::arg("x"),
           "Sample the scattering angle theta using a uniform random number x in [0, 1]")
      .def("sample_phi", &PhaseFunction::sample_phi, py::arg("x"),
           "Sample the azimuthal angle phi using a uniform random number x in [0, 1]")
      .def("sample_phi_conditional", &PhaseFunction::sample_phi_conditional,
           py::arg("theta"), py::arg("S"), py::arg("E"), py::arg("k"),
           py::arg("rng"),
           "Sample the azimuthal angle phi conditioned on theta")
      .def("get_anisotropy_factor", &PhaseFunction::get_anisotropy_factor,
           py::arg("rng"), py::arg("n_samples") = 200000,
           "Estimate the anisotropy factor g using Monte Carlo sampling");

  py::class_<UniformPhaseFunction, PhaseFunction>(m, "UniformPhaseFunction")
      .def(py::init<>());

  py::class_<RayleighPhaseFunction, PhaseFunction>(m, "RayleighPhaseFunction")
      .def(py::init<int, double, double>(), py::arg("nDiv"), py::arg("minVal"),
           py::arg("maxVal"))
      .def("pdf", &RayleighPhaseFunction::PDF, py::arg("x"));

  py::class_<HenyeyGreensteinPhaseFunction, PhaseFunction>(
      m, "HenyeyGreensteinPhaseFunction")
      .def(py::init<double>(), py::arg("g"));

  py::class_<RayleighDebyePhaseFunction, PhaseFunction>(
      m, "RayleighDebyePhaseFunction")
      .def(py::init<double, double, int, double, double>(),
           py::arg("wavelength"), py::arg("radius"), py::arg("nDiv"),
           py::arg("minVal"), py::arg("maxVal"))
      .def("pdf", &RayleighDebyePhaseFunction::PDF, py::arg("x"));

  py::class_<RayleighDebyeEMCPhaseFunction, PhaseFunction>(
      m, "RayleighDebyeEMCPhaseFunction")
      .def(py::init<double, double, int, double, double>(),
           py::arg("wavelength"), py::arg("radius"), py::arg("nDiv"),
           py::arg("minVal"), py::arg("maxVal"))
      .def("pdf", &RayleighDebyeEMCPhaseFunction::PDF, py::arg("x"));

  py::class_<DrainePhaseFunction, PhaseFunction>(m, "DrainePhaseFunction")
      .def(py::init<double, double, int, double, double>(), py::arg("g"),
           py::arg("a"), py::arg("nDiv"), py::arg("minVal"), py::arg("maxVal"))
      .def("pdf", &DrainePhaseFunction::PDF, py::arg("x"));

  // Photon bindings
  py::class_<Photon>(m, "Photon")
      .def(py::init<>())
      .def(py::init<Vec3, Vec3, Vec3, Vec3, double>(),
           py::arg("position"), py::arg("direction"), py::arg("m"),
           py::arg("n"), py::arg("wavelength"))
      .def_readwrite("prev_pos", &Photon::prev_pos)
      .def_readwrite("pos", &Photon::pos)
      .def_readwrite("dir", &Photon::dir)
      .def_readwrite("m", &Photon::m)
      .def_readwrite("n", &Photon::n)
      .def_readwrite("events", &Photon::events)
      .def_readwrite("alive", &Photon::alive)
      .def_readwrite("wavelength_nm", &Photon::wavelength_nm)
      .def_readwrite("opticalpath", &Photon::opticalpath)
      .def_readwrite("weight", &Photon::weight)
      .def_readwrite("polarized", &Photon::polarized)
      .def_readwrite("polarization", &Photon::polarization)
      .def_readonly("k", &Photon::k)
      .def("set_polarization", &Photon::set_polarization, py::arg("pol1"),
           py::arg("pol2"), "Set the polarization state of the photon")
      .def("get_stokes_parameters", &Photon::get_stokes_parameters,
           "Get the Stokes parameters of the photon");

  // Laser Bindings
  py::enum_<LaserSource>(m, "LaserSource")
      .value("Point", LaserSource::Point)
      .value("Uniform", LaserSource::Uniform)
      .value("Gaussian", LaserSource::Gaussian)
      .export_values();

  py::class_<Laser>(m, "Laser")
      .def(
          py::init<Vec3, Vec3, Vec3, Vec3, CVec2, double, double, LaserSource>(),
          py::arg("position"), py::arg("direction"), py::arg("local_m"),
          py::arg("local_n"), py::arg("polarization"), py::arg("wavelength"),
          py::arg("sigma"), py::arg("source_type"))
      .def("emit_photon", &Laser::emit_photon, py::arg("rng"),
           "Emit a photon from the laser source");

  // Detector bindings
  py::class_<Detector>(m, "Detector")
      .def(py::init<Vec3, Vec3, Vec3, Vec3>(), py::arg("origin"),
           py::arg("normal"), py::arg("n_polarization"),
           py::arg("m_polarization"))
      .def_readonly("hits", &Detector::hits)
      .def_readonly("origin", &Detector::origin)
      .def_readonly("normal", &Detector::normal)
      .def_readonly("recorded_photons", &Detector::recorded_photons)
      .def("record_hit", &Detector::record_hit, py::arg("photon"),
           "Record a photon hit on the detector")
      .def("compute_events_histogram", &Detector::compute_events_histogram, py::arg("min_theta"),
           py::arg("max_theta"),
           "Get a histogram of photon hits based on the number of scattering events")
      .def("compute_speckle",
           &Detector::compute_speckle, py::arg("n_theta") = 1125,
           py::arg("n_phi") = 360,
           "Get the angular speckle distribution of photon hits")
      .def("compute_spatial_intensity", &Detector::compute_spatial_intensity, py::arg("max_theta"),
           py::arg("n_x") = 1125, py::arg("n_y") = 1125,
           py::arg("x_max") = 10.0, py::arg("y_max") = 10.0,
           "Get the spatial intensity distribution of photon hits")
      .def("compute_angular_intensity", &Detector::compute_angular_intensity, py::arg("max_theta"),
           py::arg("max_phi"), py::arg("n_theta") = 360,
           py::arg("n_phi") = 360,
           "Get the angular intensity distribution of photon hits");

  py::class_<AngularIntensity>(m, "AngularSpeckle")
      .def_readonly("Ix", &AngularIntensity::Ix)
      .def_readonly("Iy", &AngularIntensity::Iy)
      .def_readonly("I", &AngularIntensity::I)
      .def_readonly("N_theta", &AngularIntensity::N_theta)
      .def_readonly("N_phi", &AngularIntensity::N_phi)
      .def_readonly("theta_max", &AngularIntensity::theta_max)
      .def_readonly("phi_max", &AngularIntensity::phi_max);

  py::class_<SpatialIntensity>(m, "SpatialIntensity")
      .def_readonly("Ix", &SpatialIntensity::Ix)
      .def_readonly("Iy", &SpatialIntensity::Iy)
      .def_readonly("I", &SpatialIntensity::I)
      .def_readonly("N_x", &SpatialIntensity::N_x)
      .def_readonly("N_y", &SpatialIntensity::N_y)
      .def_readonly("x_max", &SpatialIntensity::x_max)
      .def_readonly("y_max", &SpatialIntensity::y_max)
      .def_readonly("dx", &SpatialIntensity::dx)
      .def_readonly("dy", &SpatialIntensity::dy);

  // Medium bindings
  py::class_<Medium>(m, "Medium")
      .def_readonly("mu_a", &Medium::mu_absorption)
      .def_readonly("mu_s", &Medium::mu_scattering)
      .def_readonly("mu_t", &Medium::mu_attenuation)
      .def_readonly("phase_function", &Medium::phase_function)
      .def_readwrite("absorption", &Medium::absorption)
      .def("sample_free_path", &Medium::sample_free_path, py::arg("rng"),
           "Sample the free path length in the medium")
      .def("sample_scattering_angle", &Medium::sample_scattering_angle,
           py::arg("rng"), "Sample the scattering angle in the medium")
      .def("sample_azimuthal_angle", &Medium::sample_azimuthal_angle,
           py::arg("rng"), "Sample the azimuthal angle in the medium")
      .def("scattering_matrix", &Medium::scattering_matrix, py::arg("theta"),
           py::arg("phi"), py::arg("k"),
           "Get the scattering matrix for given angles and wavenumber");

  py::class_<SimpleMedium, Medium>(m, "SimpleMedium")
      .def(py::init<double, double, PhaseFunction *, double, double>(),
           py::arg("absorption"), py::arg("scattering"), py::arg("phase_func"),
           py::arg("mfp"), py::arg("radius"))
      .def_readonly("mean_free_path", &SimpleMedium::mean_free_path)
      .def_readonly("radius", &SimpleMedium::radius);

  // Absorption bindings
  py::class_<Absorption>(m, "Absorption")
      .def(py::init<double, double, double, double>(), py::arg("radius"), py::arg("depth"), py::arg("d_r"), py::arg("d_z"))
      .def_readonly("radius", &Absorption::radius)
      .def_readonly("depth", &Absorption::depth)
      .def_readonly("d_r", &Absorption::d_r)
      .def_readonly("d_z", &Absorption::d_z)
      .def_readonly("absorption_values", &Absorption::absorption_values)
      .def("record_absorption", &Absorption::record_absorption,
           py::arg("photon"), py::arg("d_weight"),
           "Record absorption from a photon at its current position")
      .def("get_absorption_image", &Absorption::get_absorption_image, py::arg("n_photons"),
           "Get the 2D absorption image");

  py::class_<AbsorptionTimeDependent>(m, "AbsortionTimeDependent")
      .def(py::init<double, double, double, double, double, double>(), py::arg("radius"), py::arg("depth"), py::arg("d_r"), py::arg("d_z"), py::arg("d_t"), py::arg("t_max"))
      .def_readonly("radius", &AbsorptionTimeDependent::radius)
      .def_readonly("depth", &AbsorptionTimeDependent::depth)
      .def_readonly("d_r", &AbsorptionTimeDependent::d_r)
      .def_readonly("d_z", &AbsorptionTimeDependent::d_z)
      .def_readonly("d_t", &AbsorptionTimeDependent::d_t)
      .def_readonly("time_slices", &AbsorptionTimeDependent::time_slices)
      .def("record_absorption", &AbsorptionTimeDependent::record_absorption,
           py::arg("photon"), py::arg("d_weight"),
           "Record absorption from a photon at its current position and time")
      .def("get_absorption_image", &AbsorptionTimeDependent::get_absorption_image, py::arg("n_photons"), py::arg("time_index"),
           "Get the 2D absorption image for a specific time slice");

  // Simulation bindings
  py::class_<SimConfig>(m, "SimConfig")
      .def(py::init<std::size_t>(), py::arg("n_photons"))
      .def(py::init<std::uint64_t, std::size_t>(), py::arg("seed"),
           py::arg("n_photons"))
      .def_readwrite("seed", &SimConfig::seed)
      .def_readwrite("n_photons", &SimConfig::n_photons);

  m.def(
      "run_simulation",
      [](SimConfig &config, Medium &medium, Detector &detector, Laser &laser) {
        py::gil_scoped_release release;
        run_simulation(config, medium, detector, laser);
      },
      py::arg("config"), py::arg("medium"), py::arg("detector"),
      py::arg("laser"),
      "Run the Monte Carlo simulation with the given configuration, medium, "
      "detector, and laser");

  // Logger bindings
  py::enum_<Level>(m, "LogLevel")
      .value("debug", Level::debug)
      .value("info", Level::info)
      .value("warn", Level::warn)
      .value("error", Level::error)
      .value("off", Level::off)
      .export_values();

  m.def(
      "set_log_level", [](Level level) { Logger::instance().set_level(level); },
      py::arg("level"), "Set the logging level for the luminis-mc module");
}
