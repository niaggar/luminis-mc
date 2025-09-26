#include <luminis/core/simulation.hpp>
#include <luminis/core/laser.hpp>
#include <luminis/core/photon.hpp>
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
        .def_readwrite("previous_step", &Photon::previous_step)
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
