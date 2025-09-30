#include <luminis/core/simulation.hpp>
#include <luminis/core/laser.hpp>
#include <luminis/core/photon.hpp>
#include <luminis/core/detector.hpp>
#include <luminis/core/medium.hpp>
#include <luminis/sample/phase.hpp>
#include <luminis/math/rng.hpp>
#include <luminis/log/logger.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace luminis::core;
using namespace luminis::sample;
using namespace luminis::log;

PYBIND11_MODULE(luminis_mc, m)
{
    m.doc() = "Python bindings for the luminis-mc Monte Carlo core";

    // Rng bindings
    py::class_<Rng>(m, "Rng")
        .def(py::init<uint64_t>(), py::arg("seed") = std::random_device{}(), "Initialize the RNG with an optional seed")
        .def("uniform", &Rng::uniform, "Generate a uniform random number in [0, 1)")
        .def("normal", &Rng::normal, py::arg("mean"), py::arg("stddev"), "Generate a normally distributed random number with given mean and stddev");

    // Phase function bindings
    py::class_<PhaseFunction>(m, "PhaseFunction")
        .def("sample", &PhaseFunction::sample_cos, py::arg("x"), "Sample the cosine of the scattering angle using a uniform random number x in [0, 1]")
        .def("sample_theta", &PhaseFunction::sample_theta, py::arg("x"), "Sample the scattering angle theta using a uniform random number x in [0, 1]");

    py::class_<UniformPhaseFunction, PhaseFunction>(m, "UniformPhaseFunction")
        .def(py::init<>());

    py::class_<RayleighPhaseFunction, PhaseFunction>(m, "RayleighPhaseFunction")
        .def(py::init<int, double, double>(),
             py::arg("nDiv"), py::arg("minVal"), py::arg("maxVal"))
        .def("pdf", &RayleighPhaseFunction::PDF, py::arg("x"));

    py::class_<HenyeyGreensteinPhaseFunction, PhaseFunction>(m, "HenyeyGreensteinPhaseFunction")
        .def(py::init<double>(), py::arg("g"));

    py::class_<RayleighDebyePhaseFunction, PhaseFunction>(m, "RayleighDebyePhaseFunction")
        .def(py::init<double, double, int, double, double>(),
             py::arg("wavelength"), py::arg("radius"), py::arg("nDiv"), py::arg("minVal"), py::arg("maxVal"))
        .def("pdf", &RayleighDebyePhaseFunction::PDF, py::arg("x"));

    py::class_<DrainePhaseFunction, PhaseFunction>(m, "DrainePhaseFunction")
        .def(py::init<double, double, int, double, double>(),
             py::arg("g"), py::arg("a"), py::arg("nDiv"), py::arg("minVal"), py::arg("maxVal"))
        .def("pdf", &DrainePhaseFunction::PDF, py::arg("x"));

    // Photon bindings
    py::class_<Photon>(m, "Photon")
        .def(py::init<>())
        .def(py::init<Vec3, Vec3, double>(), py::arg("position"), py::arg("direction"), py::arg("wavelength_nm"))
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
        .def("get_stokes_parameters", &Photon::get_stokes_parameters, "Get the Stokes parameters of the photon");

    // Laser Bindings
    py::enum_<LaserSource>(m, "LaserSource")
        .value("Point", LaserSource::Point)
        .value("Uniform", LaserSource::Uniform)
        .value("Gaussian", LaserSource::Gaussian)
        .export_values();

    py::class_<Laser>(m, "Laser")
        .def(py::init<Vec3, Vec3, Vec2, double, double, LaserSource>(),
             py::arg("position"), py::arg("direction"), py::arg("polarization"),
             py::arg("wavelength"), py::arg("sigma"), py::arg("source_type"))
        .def("emit_photon", &Laser::emit_photon, py::arg("rng"), "Emit a photon from the laser source");

    // Detector bindings
    py::class_<Detector>(m, "Detector")
        .def(py::init<Vec3, Vec3>(), py::arg("origin"), py::arg("normal"))
        .def_readonly("hits", &Detector::hits)
        .def_readonly("origin", &Detector::origin)
        .def_readonly("normal", &Detector::normal)
        .def_readonly("recorded_photons", &Detector::recorded_photons)
        .def("record_hit", &Detector::record_hit, py::arg("photon"), "Record a photon hit on the detector");

    // Medium bindings
    py::class_<Medium>(m, "Medium")
        .def_readonly("mu_a", &Medium::mu_a)
        .def_readonly("mu_s", &Medium::mu_s)
        .def_readonly("phase_function", &Medium::phase_function)
        .def("sample_free_path", &Medium::sample_free_path, py::arg("rng"), "Sample the free path length in the medium")
        .def("sample_scattering_angle", &Medium::sample_scattering_angle, py::arg("rng"), "Sample the scattering angle in the medium")
        .def("sample_azimuthal_angle", &Medium::sample_azimuthal_angle, py::arg("rng"), "Sample the azimuthal angle in the medium")
        .def("scattering_matrix", &Medium::scattering_matrix, py::arg("theta"), py::arg("phi"), py::arg("k"), "Get the scattering matrix for given angles and wavenumber");

    py::class_<SimpleMedium, Medium>(m, "SimpleMedium")
        .def(py::init<double, double, PhaseFunction*, double, double>(),
             py::arg("absorption"), py::arg("scattering"), py::arg("phase_func"), py::arg("mfp"), py::arg("radius"))
        .def_readonly("mean_free_path", &SimpleMedium::mean_free_path)
        .def_readonly("radius", &SimpleMedium::radius);

    // Simulation bindings
    py::class_<SimConfig>(m, "SimConfig")
        .def(py::init<std::size_t>(), py::arg("n_photons"))
        .def(py::init<std::uint64_t, std::size_t>(), py::arg("seed"), py::arg("n_photons"))
        .def_readwrite("seed", &SimConfig::seed)
        .def_readwrite("n_photons", &SimConfig::n_photons);

    m.def("run_simulation",
        [](SimConfig &config, Medium &medium, Detector &detector, Laser &laser)
        {
          py::gil_scoped_release release;
          run_simulation(config, medium, detector, laser);
        },
        py::arg("config"), py::arg("medium"), py::arg("detector"), py::arg("laser"),
        "Run the Monte Carlo simulation with the given configuration, medium, detector, and laser");


    // Logger bindings
    py::enum_<Level>(m, "LogLevel")
        .value("debug", Level::debug)
        .value("info", Level::info)
        .value("warn", Level::warn)
        .value("error", Level::error)
        .value("off", Level::off)
        .export_values();

    m.def(
        "set_log_level",
        [](Level level)
        {
            Logger::instance().set_level(level);
        },
        py::arg("level"),
        "Set the logging level for the luminis-mc module");
}
