#include <luminis/core/detector.hpp>
#include <luminis/core/laser.hpp>
#include <luminis/core/medium.hpp>
#include <luminis/core/photon.hpp>
#include <luminis/core/simulation.hpp>
#include <luminis/core/absortion.hpp>
#include <luminis/core/core.hpp>
#include <luminis/log/logger.hpp>
#include <luminis/math/rng.hpp>
#include <luminis/sample/phase.hpp>
#include <luminis/sample/meanfreepath.hpp>

#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace luminis::core;
using namespace luminis::sample;
using namespace luminis::log;

PYBIND11_MODULE(_core, m)
{
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

  // Math
  py::class_<Vec3>(m, "Vec3")
      .def(py::init<>())
      .def(py::init<double, double, double>(), py::arg("x"), py::arg("y"),
           py::arg("z"))
      .def_readwrite("x", &Vec3::x)
      .def_readwrite("y", &Vec3::y)
      .def_readwrite("z", &Vec3::z)
      .def("__repr__", [](const Vec3 &v)
           { return "Vec3(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ")"; });

  py::class_<Vec2>(m, "Vec2")
      .def(py::init<>())
      .def(py::init<double, double>(), py::arg("x"), py::arg("y"))
      .def_readwrite("x", &Vec2::x)
      .def_readwrite("y", &Vec2::y);

  py::class_<CVec2>(m, "CVec2")
      .def(py::init<>())
      .def(py::init<std::complex<double>, std::complex<double>>(), py::arg("m"), py::arg("n"))
      .def_readwrite("m", &CVec2::m)
      .def_readwrite("n", &CVec2::n);

  // Matrix bindings
  py::class_<Matrix>(m, "Matrix", py::buffer_protocol())
      .def(py::init<uint, uint>(), py::arg("rows"), py::arg("cols"),
           "Initialize a Matrix with given number of rows and columns")
      .def("get", [](const Matrix &mat, uint i, uint j)
           { return mat(i, j); }, py::arg("i"), py::arg("j"), "Get the element at row i and column j")
      .def("set", [](Matrix &mat, uint i, uint j, double value)
           { mat(i, j) = value; }, py::arg("i"), py::arg("j"), py::arg("value"), "Set the element at row i and column j to value")
      .def("get_numpy", [](const Matrix &mat)
           { return py::array_t<double>(
                 {mat.rows, mat.cols},
                 {sizeof(double) * mat.cols, sizeof(double)},
                 mat.data.data()); }, "Get the matrix data as a NumPy array")
      .def_buffer([](Matrix &mat) -> py::buffer_info
                  { return py::buffer_info(
                        mat.data.data(),
                        sizeof(double),
                        py::format_descriptor<double>::format(),
                        2,
                        {mat.rows, mat.cols},
                        {sizeof(double) * mat.cols, sizeof(double)}); })
      .def_readonly("rows", &Matrix::rows, "Get the number of rows in the matrix")
      .def_readonly("cols", &Matrix::cols, "Get the number of columns in the matrix");

  py::class_<CMatrix>(m, "CMatrix", py::buffer_protocol())
      .def(py::init<uint, uint>(), py::arg("rows"), py::arg("cols"),
           "Initialize a CMatrix with given number of rows and columns")
      .def_static("identity", &CMatrix::identity, py::arg("size"),
                  "Create an identity matrix of given size")
      .def("get", [](const CMatrix &mat, uint i, uint j)
           { return mat(i, j); }, py::arg("i"), py::arg("j"), "Get the element at row i and column j")
      .def("set", [](CMatrix &mat, uint i, uint j, std::complex<double> value)
           { mat(i, j) = value; }, py::arg("i"), py::arg("j"), py::arg("value"), "Set the element at row i and column j to value")
      .def("get_numpy", [](const CMatrix &mat)
           { return py::array_t<std::complex<double>>(
                 {mat.rows, mat.cols},
                 {sizeof(std::complex<double>) * mat.cols, sizeof(std::complex<double>)},
                 mat.data.data()); }, "Get the complex matrix data as a NumPy array")
      .def_buffer([](CMatrix &mat) -> py::buffer_info
                  { return py::buffer_info(
                        mat.data.data(),
                        sizeof(std::complex<double>),
                        py::format_descriptor<std::complex<double>>::format(),
                        2,
                        {mat.rows, mat.cols},
                        {sizeof(std::complex<double>) * mat.cols, sizeof(std::complex<double>)}); })
      .def_readonly("rows", &CMatrix::rows, "Get the number of rows in the matrix")
      .def_readonly("cols", &CMatrix::cols, "Get the number of columns in the matrix");

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
      .def(py::init<double, double, double, double, int, double, double>(),
           py::arg("wavelength"), py::arg("radius"), py::arg("n_particle"), py::arg("n_medium"), py::arg("nDiv"),
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
      .def_readwrite("penetration_depth", &Photon::penetration_depth)
      .def_readwrite("alive", &Photon::alive)
      .def_readwrite("wavelength_nm", &Photon::wavelength_nm)
      .def_readwrite("k", &Photon::k)
      .def_readwrite("opticalpath", &Photon::opticalpath)
      .def_readwrite("launch_time", &Photon::launch_time)
      .def_readwrite("velocity", &Photon::velocity)
      .def_readwrite("weight", &Photon::weight)
      .def_readwrite("polarized", &Photon::polarized)
      .def_readwrite("polarization", &Photon::polarization)
      .def_readonly("n_0", &Photon::n_0)
      .def_readonly("s_0", &Photon::s_0)
      .def_readonly("s_1", &Photon::s_1)
      .def_readonly("s_n2", &Photon::s_n2)
      .def_readonly("s_n1", &Photon::s_n1)
      .def_readonly("s_n", &Photon::s_n)
      .def_readonly("r_0", &Photon::r_0)
      .def_readonly("r_n", &Photon::r_n)
      .def_readonly("matrix_T", &Photon::matrix_T)
      .def_readonly("matrix_T_buffer", &Photon::matrix_T_buffer)
      .def("set_polarization", &Photon::set_polarization, py::arg("polarization"),
           "Set the polarization state of the photon")
      .def("get_stokes_parameters", &Photon::get_stokes_parameters,
           "Get the Stokes parameters of the photon");

  py::class_<PhotonRecord>(m, "PhotonRecord")
      .def_readonly("events", &PhotonRecord::events)
      .def_readonly("penetration_depth", &PhotonRecord::penetration_depth)
      .def_readonly("launch_time", &PhotonRecord::launch_time)
      .def_readonly("arrival_time", &PhotonRecord::arrival_time)
      .def_readonly("opticalpath", &PhotonRecord::opticalpath)
      .def_readonly("weight", &PhotonRecord::weight)
      .def_readonly("k", &PhotonRecord::k)
      .def_readonly("position_first_scattering", &PhotonRecord::position_first_scattering)
      .def_readonly("position_last_scattering", &PhotonRecord::position_last_scattering)
      .def_readonly("position_detector", &PhotonRecord::position_detector)
      .def_readonly("direction", &PhotonRecord::direction)
      .def_readonly("m", &PhotonRecord::m)
      .def_readonly("n", &PhotonRecord::n)
      .def_readonly("polarization_forward", &PhotonRecord::polarization_forward)
      .def_readonly("polarization_reverse", &PhotonRecord::polarization_reverse);

  // Laser Bindings
  py::enum_<LaserSource>(m, "LaserSource")
      .value("Point", LaserSource::Point)
      .value("Uniform", LaserSource::Uniform)
      .value("Gaussian", LaserSource::Gaussian)
      .export_values();

  py::enum_<TemporalProfile>(m, "TemporalProfile")
      .value("Delta", TemporalProfile::Delta)
      .value("Gaussian", TemporalProfile::Gaussian)
      .value("TopHat", TemporalProfile::TopHat)
      .value("ExponentialTime", TemporalProfile::Exponential)
      .value("PulseTrain", TemporalProfile::PulseTrain)
      .value("CW", TemporalProfile::CW)
      .export_values();

  py::class_<Laser>(m, "Laser")
      .def(py::init<Vec3, CVec2, double, double, LaserSource>(),
           py::arg("position"), py::arg("polarization"), py::arg("wavelength"),
           py::arg("sigma"), py::arg("source_type"),
           "Initialize a Laser source with given parameters")
      .def_readwrite("position", &Laser::position)
      .def_readwrite("direction", &Laser::direction)
      .def_readwrite("local_m", &Laser::local_m)
      .def_readwrite("local_n", &Laser::local_n)
      .def_readwrite("polarization", &Laser::polarization)
      .def_readwrite("wavelength", &Laser::wavelength)
      .def_readwrite("sigma", &Laser::sigma)
      .def_readwrite("source_type", &Laser::source_type)
      .def_readwrite("temporal_profile", &Laser::temporal_profile)
      .def_readwrite("pulse_duration", &Laser::pulse_duration)
      .def_readwrite("repetition_rate", &Laser::repetition_rate)
      .def_readwrite("time_offset", &Laser::time_offset)
      .def("set_temporal_profile", &Laser::set_temporal_profile,
           py::arg("profile"), py::arg("pulse_duration") = 0.0,
           py::arg("repetition_rate") = 0.0, py::arg("time_offset") = 0.0,
           "Set the temporal profile of the laser")
      .def("sample_emission_time", &Laser::sample_emission_time, py::arg("rng"),
           "Sample an emission time from the temporal profile")
      .def("emit_photon", &Laser::emit_photon, py::arg("rng"),
           "Emit a photon from the laser source");

  // Detection condition bindings
  py::class_<DetectionCondition>(m, "DetectionCondition")
      .def(py::init<std::function<bool(const Photon &)>>(),
           py::arg("condition"),
           "Initialize a DetectionCondition with a callable condition");

  m.def("make_theta_condition",
        &make_theta_condition,
        py::arg("min_theta"), py::arg("max_theta"),
        "Create a detection condition based on polar angle range");

  m.def("make_phi_condition",
        &make_phi_condition,
        py::arg("min_phi"), py::arg("max_phi"),
        "Create a detection condition based on azimuthal angle range");

  m.def("make_position_condition",
        &make_position_condition,
        py::arg("x_min"), py::arg("x_max"),
        py::arg("y_min"), py::arg("y_max"),
        "Create a detection condition based on position range");

  m.def("make_events_condition",
        &make_events_condition,
        py::arg("min_events"), py::arg("max_events"),
        "Create a detection condition based on number of scattering events");

  // MultiDetector bindings
  py::class_<MultiDetector>(m, "MultiDetector")
      .def(py::init<>())
      .def_readonly("detectors", &MultiDetector::detectors)
      .def("add_detector", [](MultiDetector &self, const Detector &det) -> Detector *
           {
          auto cloned_det = det.clone();
          self.add_detector(std::move(cloned_det));
          return self.detectors.back().get(); }, py::arg("detector"), py::return_value_policy::reference_internal, "Agrega un detector y devuelve una referencia a la copia interna");

  // Detector bindings
  py::class_<Detector>(m, "Detector")
      .def(py::init<double>(), py::arg("z"),
           "Initialize a Detector at a given z position")
      .def_readonly("id", &Detector::id)
      .def_readonly("hits", &Detector::hits)
      .def_readonly("origin", &Detector::origin)
      .def_readonly("normal", &Detector::normal)
      .def_readonly("backward_normal", &Detector::backward_normal)
      .def_readonly("n_polarization", &Detector::n_polarization)
      .def_readonly("m_polarization", &Detector::m_polarization)
      .def_readonly("recorded_photons", &Detector::recorded_photons)
      .def("set_theta_limit", &Detector::set_theta_limit,
           py::arg("min_theta"), py::arg("max_theta"),
           "Set the polar angle detection limits (in radians)")
      .def("set_phi_limit", &Detector::set_phi_limit,
           py::arg("min_phi"), py::arg("max_phi"),
           "Set the azimuthal angle detection limits (in radians)");

  // AngleDetector bindings
  py::class_<AngleDetector, Detector>(m, "AngleDetector")
      .def(py::init<double, int, int>(), py::arg("z"), py::arg("n_theta"),
           py::arg("n_phi"),
           "Initialize a AngleDetector at a given z position with angular "
           "resolution")
      .def_readonly("N_theta", &AngleDetector::N_theta)
      .def_readonly("N_phi", &AngleDetector::N_phi)
      .def_readonly("dtheta", &AngleDetector::dtheta)
      .def_readonly("dphi", &AngleDetector::dphi)
      .def_readonly("E_x", &AngleDetector::E_x)
      .def_readonly("E_y", &AngleDetector::E_y)
      .def_readonly("E_z", &AngleDetector::E_z);

  // HistogramDetector bindings
  py::class_<HistogramDetector, Detector>(m, "HistogramDetector")
      .def(py::init<double, u_int>(), py::arg("z"), py::arg("max_events"),
           "Initialize a HistogramDetector at a given z position")
      .def_readonly("histogram", &HistogramDetector::histogram);

  // ThetaHistogramDetector bindings
  py::class_<ThetaHistogramDetector, Detector>(m, "ThetaHistogramDetector")
      .def(py::init<double, u_int>(), py::arg("z"), py::arg("n_bins"),
           "Initialize a ThetaHistogramDetector at a given z position")
      .def_readonly("histogram", &ThetaHistogramDetector::histogram);

  // SpatialDetector bindings
  py::class_<SpatialDetector, Detector>(m, "SpatialDetector")
      .def(py::init<double, double, double, int, int>(), py::arg("z"),
           py::arg("x_len"), py::arg("y_len"), py::arg("N_x"), py::arg("N_y"),
           "Initialize a SpatialDetector at a given z position with spatial "
           "resolution")
      .def_readonly("N_x", &SpatialDetector::N_x)
      .def_readonly("N_y", &SpatialDetector::N_y)
      .def_readonly("dx", &SpatialDetector::dx)
      .def_readonly("dy", &SpatialDetector::dy)
      .def_readonly("E_x", &SpatialDetector::E_x)
      .def_readonly("E_y", &SpatialDetector::E_y)
      .def_readonly("E_z", &SpatialDetector::E_z)
      .def_readonly("I_x", &SpatialDetector::I_x)
      .def_readonly("I_y", &SpatialDetector::I_y)
      .def_readonly("I_z", &SpatialDetector::I_z)
      .def_readonly("I_plus", &SpatialDetector::I_plus)
      .def_readonly("I_minus", &SpatialDetector::I_minus);

  // SpatialCoherentDetector bindings
  py::class_<SpatialCoherentDetector, Detector>(m, "SpatialCoherentDetector")
      .def(py::init<double, double, double, int, int>(), py::arg("z"),
           py::arg("x_len"), py::arg("y_len"), py::arg("N_x"), py::arg("N_y"),
           "Initialize a SpatialCoherentDetector at a given z position with spatial "
           "resolution")
      .def_readonly("N_x", &SpatialCoherentDetector::N_x)
      .def_readonly("N_y", &SpatialCoherentDetector::N_y)
      .def_readonly("dx", &SpatialCoherentDetector::dx)
      .def_readonly("dy", &SpatialCoherentDetector::dy)
      .def_readonly("I_x", &SpatialCoherentDetector::I_x)
      .def_readonly("I_y", &SpatialCoherentDetector::I_y)
      .def_readonly("I_z", &SpatialCoherentDetector::I_z)
      .def_readonly("I_inco_x", &SpatialCoherentDetector::I_inco_x)
      .def_readonly("I_inco_y", &SpatialCoherentDetector::I_inco_y)
      .def_readonly("I_inco_z", &SpatialCoherentDetector::I_inco_z)
      .def_readonly("I_x_theta", &SpatialCoherentDetector::I_x_theta)
      .def_readonly("I_y_theta", &SpatialCoherentDetector::I_y_theta)
      .def_readonly("I_z_theta", &SpatialCoherentDetector::I_z_theta)
      .def_readonly("I_inco_x_theta", &SpatialCoherentDetector::I_inco_x_theta)
      .def_readonly("I_inco_y_theta", &SpatialCoherentDetector::I_inco_y_theta)
      .def_readonly("I_inco_z_theta", &SpatialCoherentDetector::I_inco_z_theta);

  py::class_<AngularCoherentDetector, Detector>(m, "AngularCoherentDetector")
      .def(py::init<double, int, double>(), py::arg("z"), py::arg("n_theta"),
           py::arg("theta_max"),
           "Initialize a AngularCoherentDetector at a given z position with angular "
           "resolution")
      .def_readonly("N_theta", &AngularCoherentDetector::N_theta)
      .def_readonly("dtheta", &AngularCoherentDetector::dtheta)
      .def_readonly("I_x", &AngularCoherentDetector::I_x)
      .def_readonly("I_y", &AngularCoherentDetector::I_y)
      .def_readonly("I_z", &AngularCoherentDetector::I_z)
      .def_readonly("I_plus", &AngularCoherentDetector::I_plus)
      .def_readonly("I_minus", &AngularCoherentDetector::I_minus)
      .def_readonly("I_total", &AngularCoherentDetector::I_total)
      .def_readonly("I_inco_x", &AngularCoherentDetector::I_inco_x)
      .def_readonly("I_inco_y", &AngularCoherentDetector::I_inco_y)
      .def_readonly("I_inco_z", &AngularCoherentDetector::I_inco_z)
      .def_readonly("I_inco_plus", &AngularCoherentDetector::I_inco_plus)
      .def_readonly("I_inco_minus", &AngularCoherentDetector::I_inco_minus)
      .def_readonly("I_inco_total", &AngularCoherentDetector::I_inco_total);

  py::class_<AngularIntensity>(m, "AngularSpeckle")
      .def_readonly("Ix", &AngularIntensity::Ix)
      .def_readonly("Iy", &AngularIntensity::Iy)
      .def_readonly("Iz", &AngularIntensity::Iz)
      .def_readonly("I_total", &AngularIntensity::I_total)
      .def_readonly("Ico", &AngularIntensity::Ico)
      .def_readonly("Icros", &AngularIntensity::Icros)
      .def_readonly("N_theta", &AngularIntensity::N_theta)
      .def_readonly("N_phi", &AngularIntensity::N_phi)
      .def_readonly("theta_max", &AngularIntensity::theta_max)
      .def_readonly("phi_max", &AngularIntensity::phi_max)
      .def_readonly("dtheta", &AngularIntensity::dtheta)
      .def_readonly("dphi", &AngularIntensity::dphi);

  py::class_<SpatialIntensity>(m, "SpatialIntensity")
      .def_readonly("Ix", &SpatialIntensity::Ix)
      .def_readonly("Iy", &SpatialIntensity::Iy)
      .def_readonly("Iz", &SpatialIntensity::Iz)
      .def_readonly("I_total", &SpatialIntensity::I_total)
      .def_readonly("Ico", &SpatialIntensity::Ico)
      .def_readonly("Icros", &SpatialIntensity::Icros)
      .def_readonly("N_x", &SpatialIntensity::N_x)
      .def_readonly("N_y", &SpatialIntensity::N_y)
      .def_readonly("x_len", &SpatialIntensity::x_len)
      .def_readonly("y_len", &SpatialIntensity::y_len)
      .def_readonly("dx", &SpatialIntensity::dx)
      .def_readonly("dy", &SpatialIntensity::dy);

  m.def("compute_events_histogram", &compute_events_histogram,
        py::arg("detector"), py::arg("min_theta"), py::arg("max_theta"),
        py::return_value_policy::take_ownership,
        "Compute the events histogram from the detector data");

  m.def("compute_theta_histogram", &compute_theta_histogram,
        py::arg("detector"), py::arg("min_theta"), py::arg("max_theta"), py::arg("n_bins"),
        py::return_value_policy::take_ownership,
        "Compute the theta histogram from the detector data");

  m.def("compute_phi_histogram", &compute_phi_histogram,
        py::arg("detector"), py::arg("min_phi"), py::arg("max_phi"), py::arg("n_bins"),
        py::return_value_policy::take_ownership,
        "Compute the phi histogram from the detector data");

  m.def("compute_speckle", &compute_speckle,
        py::arg("detector"), py::arg("n_theta"), py::arg("n_phi"),
        py::return_value_policy::take_ownership,
        "Compute the angular speckle pattern from the detector data");

  m.def("compute_speckle_angledetector", &compute_speckle_angledetector,
        py::arg("angledetector"),
        py::return_value_policy::take_ownership,
        "Compute the angular speckle pattern from the detector data");

  // Medium bindings
  py::class_<Medium>(m, "Medium")
      .def_readonly("mu_a", &Medium::mu_absorption)
      .def_readonly("mu_s", &Medium::mu_scattering)
      .def_readonly("mu_t", &Medium::mu_attenuation)
      .def_readonly("light_speed", &Medium::light_speed)
      .def_readonly("refractive_index", &Medium::refractive_index)
      .def_readonly("phase_function", &Medium::phase_function)
      .def("sample_free_path", &Medium::sample_free_path, py::arg("rng"),
           "Sample the free path length in the medium")
      .def("sample_scattering_angle", &Medium::sample_scattering_angle,
           py::arg("rng"), "Sample the scattering angle in the medium")
      .def("sample_azimuthal_angle", &Medium::sample_azimuthal_angle,
           py::arg("rng"), "Sample the azimuthal angle in the medium")
      .def("sample_conditional_azimuthal_angle", &Medium::sample_conditional_azimuthal_angle,
           py::arg("rng"), py::arg("S"), py::arg("E"), py::arg("k"), py::arg("theta"),
           "Sample the azimuthal angle conditioned on scattering angle theta")
      .def("scattering_matrix", &Medium::scattering_matrix, py::arg("theta"),
           py::arg("phi"), py::arg("k"),
           "Get the scattering matrix for given angles and wavenumber")
      .def("light_speed_in_medium", &Medium::light_speed_in_medium,
           "Get the speed of light in the medium");

  py::class_<SimpleMedium, Medium>(m, "SimpleMedium")
      .def(py::init<double, double, PhaseFunction *, double, double, double, double>(),
           py::arg("absorption"), py::arg("scattering"), py::arg("phase_func"),
           py::arg("mfp"), py::arg("radius"), py::arg("n_particle"), py::arg("n_medium"))
      .def_readonly("mean_free_path", &SimpleMedium::mean_free_path)
      .def_readonly("radius", &SimpleMedium::radius)
      .def_readonly("n_particle", &SimpleMedium::n_particle)
      .def_readonly("n_medium", &SimpleMedium::n_medium);

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

  py::class_<AbsorptionTimeDependent>(m, "AbsorptionTimeDependent")
      .def(py::init<double, double, double, double, double, double>(), py::arg("radius"), py::arg("depth"), py::arg("d_r"), py::arg("d_z"), py::arg("d_t"), py::arg("t_max"))
      .def_readonly("radius", &AbsorptionTimeDependent::radius)
      .def_readonly("depth", &AbsorptionTimeDependent::depth)
      .def_readonly("d_r", &AbsorptionTimeDependent::d_r)
      .def_readonly("d_z", &AbsorptionTimeDependent::d_z)
      .def_readonly("d_t", &AbsorptionTimeDependent::d_t)
      .def_readonly("n_t_slices", &AbsorptionTimeDependent::n_t_slices)
      .def_readonly("time_slices", &AbsorptionTimeDependent::time_slices)
      .def("record_absorption", &AbsorptionTimeDependent::record_absorption,
           py::arg("photon"), py::arg("d_weight"),
           "Record absorption from a photon at its current position and time")
      .def("get_absorption_image", &AbsorptionTimeDependent::get_absorption_image, py::arg("n_photons"), py::arg("time_index"),
           "Get the 2D absorption image for a specific time slice");

  m.def("combine_absorptions", &combine_absorptions, py::arg("absorptions"),
        "Combine multiple AbsorptionTimeDependent instances into one");

  // Simulation bindings
  py::class_<SimConfig>(m, "SimConfig")
      .def(py::init<std::size_t, Medium *, Laser *, MultiDetector *, AbsorptionTimeDependent *, bool>(),
           py::arg("n_photons"), py::arg("medium"), py::arg("laser"), py::arg("detector"), py::arg("absorption") = nullptr, py::arg("track_reverse_paths") = false,
           "Initialize a simulation configuration with given parameters")
      .def(py::init<std::uint64_t, std::size_t, Medium *, Laser *, MultiDetector *, AbsorptionTimeDependent *, bool>(),
           py::arg("rng_seed"), py::arg("n_photons"), py::arg("medium"), py::arg("laser"), py::arg("detector"), py::arg("absorption") = nullptr, py::arg("track_reverse_paths") = false,
           "Initialize a simulation configuration with given parameters including RNG seed")
      .def_readonly("seed", &SimConfig::seed)
      .def_readonly("n_photons", &SimConfig::n_photons)
      .def_readonly("medium", &SimConfig::medium, pybind11::return_value_policy::reference)
      .def_readonly("laser", &SimConfig::laser, pybind11::return_value_policy::reference)
      .def_readonly("detector", &SimConfig::detector, pybind11::return_value_policy::reference)
      .def_readonly("absorption", &SimConfig::absorption, pybind11::return_value_policy::reference)
      .def_readonly("track_reverse_paths", &SimConfig::track_reverse_paths)
      .def_readwrite("n_threads", &SimConfig::n_threads);

  m.def(
      "run_simulation",
      [](SimConfig &config)
      {
        py::gil_scoped_release release;
        run_simulation(config);
      },
      py::arg("config"),
      "Run the Monte Carlo simulation with the given configuration, medium, "
      "detector, and laser");

  m.def(
      "run_simulation_parallel",
      [](SimConfig &config)
      {
        py::gil_scoped_release release;
        run_simulation_parallel(config);
      },
      py::arg("config"),
      "Run the Monte Carlo simulation in parallel with the given configuration, medium, "
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
      "set_log_level", [](Level level)
      { Logger::instance().set_level(level); },
      py::arg("level"), "Set the logging level for the luminis-mc module");

  // Math functions
  m.def("dot", &dot, py::arg("a"), py::arg("b"),
        "Compute the dot product of two Vec3 vectors");
  m.def("cross", &cross, py::arg("a"), py::arg("b"),
        "Compute the cross product of two Vec3 vectors");
  m.def("norm", &norm, py::arg("v"),
        "Compute the norm of a Vec3 vector");
  m.def("matmul", &matmul, py::arg("A"), py::arg("B"), py::arg("C"),
        "Multiply two complex matrices A and B, storing result in C");
  m.def("matmulscalar", &matmulscalar, py::arg("scalar"), py::arg("A"),
        "Multiply a complex matrix by a scalar");
  m.def("calculate_rotation_angle", &calculate_rotation_angle,
        py::arg("n_from"), py::arg("n_to"),
        "Calculate the rotation angle between two normal vectors");

  // Laser spatial distributions
  m.def("uniform_distribution", &uniform_distribution,
        py::arg("rng"), py::arg("center"), py::arg("sigma"),
        "Generate a random point from uniform distribution");
  m.def("gaussian_distribution", &gaussian_distribution,
        py::arg("rng"), py::arg("center"), py::arg("sigma"),
        "Generate a random point from Gaussian distribution");

  // Form factor for phase functions
  m.def("form_factor", &form_factor,
        py::arg("theta"), py::arg("k"), py::arg("radius"),
        "Compute the form factor for given scattering angle, wavenumber, and particle radius");

  // meanfreepath bindings

  // Bind TargetDistribution base class
  py::class_<TargetDistribution>(m, "TargetDistribution")
      .def("evaluate", &TargetDistribution::evaluate, py::arg("x"),
           "Evaluate the target distribution at x");

  py::class_<metropolis_hastings>(m, "MetropolisHastings")
      .def(py::init<TargetDistribution *>(), py::arg("target_distribution"),
           "Initialize with a target distribution function pointer")
      .def("accept_reject", &metropolis_hastings::accept_reject,
           py::arg("current_state"), py::arg("target_distribution_current_state"),
           py::arg("proposal_stddev"), py::arg("positive_support"),
           "Perform the accept-reject step of the Metropolis-Hastings algorithm")
      .def("sample", &metropolis_hastings::sample, py::arg("num_samples"),
           py::arg("initial_value"), py::arg("proposal_stddev"), py::arg("positive_support"),
           "Generate samples using the Metropolis-Hastings algorithm")
      .def_readonly("MCMC_samples", &metropolis_hastings::MCMC_samples,
                    "Get the generated MCMC samples");

  py::class_<Exponential, TargetDistribution>(m, "Exponential")
      .def(py::init<double>(), py::arg("lambda"),
           "Initialize the exponential distribution with rate parameter lambda")
      .def("evaluate", &Exponential::evaluate, py::arg("x"),
           "Evaluate the exponential distribution at x");

  py::class_<HardSpheres, TargetDistribution>(m, "HardSpheres")
      .def(py::init<double, double>(), py::arg("radius"), py::arg("density"),
           "Initialize the hard sphere distribution with given radius and density")
      .def("evaluate", &HardSpheres::evaluate, py::arg("x"),
           "Evaluate the hard sphere distribution at x");

  // Save and read data
  m.def("save_recorded_photons", &save_recorded_photons,
        py::arg("filename"), py::arg("detector"),
        "Save recorded photons from the detector to a file");

  m.def("load_recorded_photons", &load_recorded_photons,
        py::arg("filename"), py::arg("detector"),
        "Load recorded photons into the detector from a file");

  m.def("save_angle_detector_fields", &save_angle_detector_fields,
        py::arg("filename"), py::arg("angledetector"),
        "Save the fields of the angle detector to a file");

  m.def("load_angle_detector_fields", &load_angle_detector_fields,
        py::arg("filename"), py::arg("angledetector"),
        "Load the fields of the angle detector from a file");
}
