"""
Python bindings for the luminis-mc Monte Carlo core
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
__all__: list[str] = ['Absorption', 'Backward', 'Both', 'CMatrix', 'CVec2', 'CW', 'CrossingDirection', 'Delta', 'DrainePhaseFunction', 'Exponential', 'ExponentialTime', 'FarFieldCBSProcessed', 'FarFieldCBSRadialProcessed', 'FarFieldCBSSensor', 'Forward', 'Gaussian', 'HardSpheres', 'HenyeyGreensteinPhaseFunction', 'Laser', 'LaserSource', 'Layer', 'LogLevel', 'Matrix', 'MetropolisHastings', 'MieMedium', 'MiePhaseFunction', 'MixtureLayer', 'PhaseFunction', 'Photon', 'PhotonRecord', 'PhotonRecordSensor', 'PlanarFieldProcessed', 'PlanarFieldSensor', 'PlanarFluenceProcessed', 'PlanarFluenceSensor', 'Point', 'PulseTrain', 'RGDMedium', 'RayleighDebyeEMCPhaseFunction', 'RayleighDebyePhaseFunction', 'RayleighPhaseFunction', 'Rng', 'Sample', 'SampleLayer', 'ScatteringMedium', 'Sensor', 'SensorsGroup', 'SimConfig', 'StatisticsSensor', 'StokesMatrixProcessed', 'StokesRadialProcessed', 'TargetDistribution', 'TemporalProfile', 'TopHat', 'Uniform', 'UniformPhaseFunction', 'Vec2', 'Vec3', 'calculate_rotation_angle', 'cross', 'debug', 'dot', 'error', 'form_factor', 'gaussian_distribution', 'info', 'matcmul', 'matcmulscalar', 'norm', 'off', 'postprocess_farfield_cbs', 'postprocess_planar_field', 'postprocess_planar_fluence', 'run_simulation_parallel', 'set_log_level', 'uniform_distribution', 'warn']
class Absorption:
    def __init__(self, radius: typing.SupportsFloat | typing.SupportsIndex, depth: typing.SupportsFloat | typing.SupportsIndex, d_r: typing.SupportsFloat | typing.SupportsIndex, d_z: typing.SupportsFloat | typing.SupportsIndex, d_t: typing.SupportsFloat | typing.SupportsIndex = 0.0, t_max: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> None:
        """
        Construct an absorption recorder.
          d_t == 0  → time-integrated (single bin)
          d_t > 0   → n_t = ceil(t_max / d_t) + 1 bins
                       bin 0 is always integrated, bins >=1 are time windows
        """
    def clone(self) -> Absorption:
        """
        Create an empty clone with identical configuration but zeroed grids
        """
    def get_absorption_image(self, n_photons: typing.SupportsInt | typing.SupportsIndex, time_index: typing.SupportsInt | typing.SupportsIndex = 0) -> Matrix:
        """
        Get the 2D absorption image for a given time bin (default: 0)
        """
    def get_total_image(self, n_photons: typing.SupportsInt | typing.SupportsIndex) -> Matrix:
        """
        Get the integrated absorption image (alias of time_slices[0])
        """
    def merge_from(self, other: Absorption) -> None:
        """
        Accumulate another recorder's data into this one
        """
    def record_absorption(self, photon: Photon, d_weight: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Record absorption from a photon at its current position (and time)
        """
    @property
    def d_r(self) -> float:
        ...
    @property
    def d_t(self) -> float:
        ...
    @property
    def d_z(self) -> float:
        ...
    @property
    def depth(self) -> float:
        ...
    @property
    def n_t(self) -> int:
        ...
    @property
    def radius(self) -> float:
        ...
    @property
    def t_max(self) -> float:
        ...
    @property
    def time_slices(self) -> list[Matrix]:
        ...
class CMatrix:
    @staticmethod
    def identity(size: typing.SupportsInt | typing.SupportsIndex) -> CMatrix:
        """
        Create an identity matrix of given size
        """
    def __buffer__(self, flags):
        """
        Return a buffer object that exposes the underlying memory of the object.
        """
    def __init__(self, rows: typing.SupportsInt | typing.SupportsIndex, cols: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Initialize a CMatrix with given number of rows and columns
        """
    def __release_buffer__(self, buffer):
        """
        Release the buffer object that exposes the underlying memory of the object.
        """
    def get(self, i: typing.SupportsInt | typing.SupportsIndex, j: typing.SupportsInt | typing.SupportsIndex) -> complex:
        """
        Get the element at row i and column j
        """
    def get_numpy(self) -> numpy.typing.NDArray[numpy.complex128]:
        """
        Get the complex matrix data as a NumPy array
        """
    def set(self, i: typing.SupportsInt | typing.SupportsIndex, j: typing.SupportsInt | typing.SupportsIndex, value: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set the element at row i and column j to value
        """
    @property
    def cols(self) -> int:
        """
        Get the number of columns in the matrix
        """
    @property
    def rows(self) -> int:
        """
        Get the number of rows in the matrix
        """
class CVec2:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, m: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, n: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def m(self) -> complex:
        ...
    @m.setter
    def m(self, arg0: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def n(self) -> complex:
        ...
    @n.setter
    def n(self, arg0: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class CrossingDirection:
    """
    Members:
    
      Forward : Photon traveling toward z+ (increasing z)
    
      Backward : Photon traveling toward z- (decreasing z)
    
      Both : Accept either direction
    """
    Backward: typing.ClassVar[CrossingDirection]  # value = <CrossingDirection.Backward: 1>
    Both: typing.ClassVar[CrossingDirection]  # value = <CrossingDirection.Both: 2>
    Forward: typing.ClassVar[CrossingDirection]  # value = <CrossingDirection.Forward: 0>
    __members__: typing.ClassVar[dict[str, CrossingDirection]]  # value = {'Forward': <CrossingDirection.Forward: 0>, 'Backward': <CrossingDirection.Backward: 1>, 'Both': <CrossingDirection.Both: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class DrainePhaseFunction(PhaseFunction):
    def __init__(self, g: typing.SupportsFloat | typing.SupportsIndex, a: typing.SupportsFloat | typing.SupportsIndex, nDiv: typing.SupportsInt | typing.SupportsIndex, minVal: typing.SupportsFloat | typing.SupportsIndex, maxVal: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def pdf(self, x: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
class Exponential(TargetDistribution):
    def __init__(self, lambda_: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Initialize the exponential free-path distribution with mean free path lambda
        """
    def evaluate(self, x: typing.SupportsFloat | typing.SupportsIndex) -> float:
        """
        Evaluate the exponential distribution at x
        """
class FarFieldCBSProcessed:
    @property
    def coherent(self) -> list[StokesMatrixProcessed]:
        ...
    @property
    def dOmega(self) -> Matrix:
        ...
    @property
    def incoherent(self) -> list[StokesMatrixProcessed]:
        ...
class FarFieldCBSRadialProcessed:
    @property
    def coherent(self) -> StokesRadialProcessed:
        ...
    @property
    def incoherent(self) -> StokesRadialProcessed:
        ...
    @property
    def theta_center(self) -> list[float]:
        ...
class FarFieldCBSSensor(Sensor):
    def __init__(self, theta_max: typing.SupportsFloat | typing.SupportsIndex, phi_max: typing.SupportsFloat | typing.SupportsIndex, len_t: typing.SupportsFloat | typing.SupportsIndex, d_theta: typing.SupportsFloat | typing.SupportsIndex, d_phi: typing.SupportsFloat | typing.SupportsIndex, d_t: typing.SupportsFloat | typing.SupportsIndex, estimator: bool = False) -> None:
        """
        Far-field CBS sensor accumulating coherent and incoherent Stokes on a (theta, phi) grid
        """
    @property
    def N_phi(self) -> int:
        ...
    @property
    def N_t(self) -> int:
        ...
    @property
    def N_theta(self) -> int:
        ...
    @property
    def S0_coh(self) -> list[Matrix]:
        ...
    @property
    def S0_incoh(self) -> list[Matrix]:
        ...
    @property
    def S1_coh(self) -> list[Matrix]:
        ...
    @property
    def S1_incoh(self) -> list[Matrix]:
        ...
    @property
    def S2_coh(self) -> list[Matrix]:
        ...
    @property
    def S2_incoh(self) -> list[Matrix]:
        ...
    @property
    def S3_coh(self) -> list[Matrix]:
        ...
    @property
    def S3_incoh(self) -> list[Matrix]:
        ...
    @property
    def dphi(self) -> float:
        ...
    @property
    def dt(self) -> float:
        ...
    @property
    def dtheta(self) -> float:
        ...
    @property
    def phi_max(self) -> float:
        ...
    @property
    def t_max(self) -> float:
        ...
    @property
    def theta_max(self) -> float:
        ...
    @property
    def theta_pp_max(self) -> float:
        ...
    @theta_pp_max.setter
    def theta_pp_max(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class HardSpheres(TargetDistribution):
    def __init__(self, radius: typing.SupportsFloat | typing.SupportsIndex, density: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Initialize the hard sphere distribution with given radius and density
        """
    def evaluate(self, x: typing.SupportsFloat | typing.SupportsIndex) -> float:
        """
        Evaluate the hard sphere distribution at x
        """
class HenyeyGreensteinPhaseFunction(PhaseFunction):
    def __init__(self, g: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class Laser:
    direction: Vec3
    local_m: Vec3
    local_n: Vec3
    polarization: CVec2
    position: Vec3
    source_type: LaserSource
    temporal_profile: TemporalProfile
    def __init__(self, m_state: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, n_state: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, wavelength: typing.SupportsFloat | typing.SupportsIndex, sigma: typing.SupportsFloat | typing.SupportsIndex, source_type: LaserSource) -> None:
        """
        Initialize a Laser source with given parameters
        """
    def emit_photon(self, rng: Rng) -> Photon:
        """
        Emit a photon from the laser source
        """
    def sample_emission_time(self, rng: Rng) -> float:
        """
        Sample an emission time from the temporal profile
        """
    def set_temporal_profile(self, profile: TemporalProfile, pulse_duration: typing.SupportsFloat | typing.SupportsIndex = 0.0, repetition_rate: typing.SupportsFloat | typing.SupportsIndex = 0.0, time_offset: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> None:
        """
        Set the temporal profile of the laser
        """
    @property
    def pulse_duration(self) -> float:
        ...
    @pulse_duration.setter
    def pulse_duration(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def repetition_rate(self) -> float:
        ...
    @repetition_rate.setter
    def repetition_rate(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def sigma(self) -> float:
        ...
    @sigma.setter
    def sigma(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def time_offset(self) -> float:
        ...
    @time_offset.setter
    def time_offset(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def wavelength(self) -> float:
        ...
    @wavelength.setter
    def wavelength(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class LaserSource:
    """
    Members:
    
      Point
    
      Uniform
    
      Gaussian
    """
    Gaussian: typing.ClassVar[LaserSource]  # value = <LaserSource.Gaussian: 2>
    Point: typing.ClassVar[LaserSource]  # value = <LaserSource.Point: 0>
    Uniform: typing.ClassVar[LaserSource]  # value = <LaserSource.Uniform: 1>
    __members__: typing.ClassVar[dict[str, LaserSource]]  # value = {'Point': <LaserSource.Point: 0>, 'Uniform': <LaserSource.Uniform: 1>, 'Gaussian': <LaserSource.Gaussian: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Layer:
    def contains(self, z: typing.SupportsFloat | typing.SupportsIndex) -> bool:
        """
        Check if a z-coordinate lies within this layer
        """
    def mu_absorption(self) -> float:
        """
        Aggregate absorption coefficient mu_a of the layer
        """
    def mu_attenuation(self) -> float:
        """
        Aggregate extinction coefficient mu_t of the layer
        """
    def mu_scattering(self) -> float:
        """
        Aggregate scattering coefficient mu_s of the layer
        """
    def thickness(self) -> float:
        """
        Return the thickness of the layer
        """
    @property
    def z_max(self) -> float:
        ...
    @property
    def z_min(self) -> float:
        ...
class LogLevel:
    """
    Members:
    
      debug
    
      info
    
      warn
    
      error
    
      off
    """
    __members__: typing.ClassVar[dict[str, LogLevel]]  # value = {'debug': <LogLevel.debug: 1>, 'info': <LogLevel.info: 2>, 'warn': <LogLevel.warn: 3>, 'error': <LogLevel.error: 4>, 'off': <LogLevel.off: 6>}
    debug: typing.ClassVar[LogLevel]  # value = <LogLevel.debug: 1>
    error: typing.ClassVar[LogLevel]  # value = <LogLevel.error: 4>
    info: typing.ClassVar[LogLevel]  # value = <LogLevel.info: 2>
    off: typing.ClassVar[LogLevel]  # value = <LogLevel.off: 6>
    warn: typing.ClassVar[LogLevel]  # value = <LogLevel.warn: 3>
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Matrix:
    def __buffer__(self, flags):
        """
        Return a buffer object that exposes the underlying memory of the object.
        """
    def __init__(self, rows: typing.SupportsInt | typing.SupportsIndex, cols: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Initialize a Matrix with given number of rows and columns
        """
    def __release_buffer__(self, buffer):
        """
        Release the buffer object that exposes the underlying memory of the object.
        """
    def get(self, i: typing.SupportsInt | typing.SupportsIndex, j: typing.SupportsInt | typing.SupportsIndex) -> float:
        """
        Get the element at row i and column j
        """
    def get_numpy(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        Get the matrix data as a NumPy array
        """
    def set(self, i: typing.SupportsInt | typing.SupportsIndex, j: typing.SupportsInt | typing.SupportsIndex, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set the element at row i and column j to value
        """
    @property
    def cols(self) -> int:
        """
        Get the number of columns in the matrix
        """
    @property
    def rows(self) -> int:
        """
        Get the number of rows in the matrix
        """
class MetropolisHastings:
    def __init__(self, target_distribution: TargetDistribution) -> None:
        """
        Initialize with a target distribution function pointer
        """
    def accept_reject(self, current_state: typing.SupportsFloat | typing.SupportsIndex, target_distribution_current_state: typing.SupportsFloat | typing.SupportsIndex, proposal_stddev: typing.SupportsFloat | typing.SupportsIndex, positive_support: bool) -> None:
        """
        Perform the accept-reject step of the Metropolis-Hastings algorithm
        """
    def sample(self, num_samples: typing.SupportsInt | typing.SupportsIndex, initial_value: typing.SupportsFloat | typing.SupportsIndex, proposal_stddev: typing.SupportsFloat | typing.SupportsIndex, positive_support: bool) -> None:
        """
        Generate samples using the Metropolis-Hastings algorithm
        """
    @property
    def MCMC_samples(self) -> list[float]:
        """
        Get the generated MCMC samples
        """
class MieMedium(ScatteringMedium):
    def __init__(self, phase_func: PhaseFunction, radius: typing.SupportsFloat | typing.SupportsIndex, n_particle: typing.SupportsFloat | typing.SupportsIndex, n_medium: typing.SupportsFloat | typing.SupportsIndex, wavelength: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_mean_free_path(self, mfp: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set the mean free path used for free-path sampling (does NOT update mu_s/mu_t)
        """
    @property
    def m(self) -> complex:
        ...
    @property
    def mean_free_path(self) -> float:
        ...
    @property
    def radius(self) -> float:
        ...
class MiePhaseFunction(PhaseFunction):
    def __init__(self, wavelength: typing.SupportsFloat | typing.SupportsIndex, radius: typing.SupportsFloat | typing.SupportsIndex, n_particle: typing.SupportsFloat | typing.SupportsIndex, n_medium: typing.SupportsFloat | typing.SupportsIndex, nDiv: typing.SupportsInt | typing.SupportsIndex, minVal: typing.SupportsFloat | typing.SupportsIndex, maxVal: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def pdf(self, x: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def rho_phase_function(self, x: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
class MixtureLayer(Layer):
    def __init__(self, species: collections.abc.Sequence[ScatteringMedium], number_densities: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], z_min: typing.SupportsFloat | typing.SupportsIndex, z_max: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def mfp_total(self) -> float:
        ...
    @property
    def mu_a_total(self) -> float:
        ...
    @property
    def mu_s_i(self) -> list[float]:
        ...
    @property
    def mu_s_total(self) -> float:
        ...
    @property
    def mu_t_total(self) -> float:
        ...
    @property
    def number_densities(self) -> list[float]:
        ...
    @property
    def selection_cdf(self) -> list[float]:
        ...
    @property
    def species(self) -> list[ScatteringMedium]:
        """
        Co-located scattering species.
        """
class PhaseFunction:
    def get_anisotropy_factor(self, n_samples: typing.SupportsInt | typing.SupportsIndex = 200000) -> typing.Annotated[list[float], "FixedSize(2)"]:
        """
        Estimate the anisotropy factor g using Monte Carlo sampling
        """
    def sample(self, x: typing.SupportsFloat | typing.SupportsIndex) -> float:
        """
        Sample the cosine of the scattering angle using a uniform random number x in [0, 1]
        """
    def sample_phi(self, x: typing.SupportsFloat | typing.SupportsIndex) -> float:
        """
        Sample the azimuthal angle phi using a uniform random number x in [0, 1]
        """
    def sample_phi_conditional(self, theta: typing.SupportsFloat | typing.SupportsIndex, S: CMatrix, E: CVec2, k: typing.SupportsFloat | typing.SupportsIndex, rng: Rng) -> float:
        """
        Sample the azimuthal angle phi conditioned on theta
        """
    def sample_theta(self, x: typing.SupportsFloat | typing.SupportsIndex) -> float:
        """
        Sample the scattering angle theta using a uniform random number x in [0, 1]
        """
    def scattering_cross_section(self) -> float:
        """
        Calculate the scattering cross-section sigma_sca for the phase function
        """
    def scattering_efficiency(self) -> float:
        """
        Calculate the scattering efficiency Q_sca for the phase function
        """
class Photon:
    P_local: Matrix
    alive: bool
    polarization: CVec2
    polarized: bool
    pos: Vec3
    prev_pos: Vec3
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, position: Vec3, direction: Vec3, m: Vec3, n: Vec3, wavelength: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def get_stokes_parameters(self) -> typing.Annotated[list[float], "FixedSize(4)"]:
        """
        Get the Stokes parameters of the photon
        """
    def set_polarization(self, polarization: CVec2) -> None:
        """
        Set the polarization state of the photon
        """
    @property
    def P0(self) -> Matrix:
        ...
    @property
    def P1(self) -> Matrix:
        ...
    @property
    def Pn(self) -> Matrix:
        ...
    @property
    def Pn1(self) -> Matrix:
        ...
    @property
    def Pn2(self) -> Matrix:
        ...
    @property
    def current_layer(self) -> int:
        ...
    @current_layer.setter
    def current_layer(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def events(self) -> int:
        ...
    @events.setter
    def events(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def k(self) -> float:
        ...
    @k.setter
    def k(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def launch_time(self) -> float:
        ...
    @launch_time.setter
    def launch_time(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def matrix_T(self) -> CMatrix:
        ...
    @property
    def matrix_T_buffer(self) -> CMatrix:
        ...
    @property
    def opticalpath(self) -> float:
        ...
    @opticalpath.setter
    def opticalpath(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def penetration_depth(self) -> float:
        ...
    @penetration_depth.setter
    def penetration_depth(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def r_1(self) -> Vec3:
        ...
    @property
    def r_n(self) -> Vec3:
        ...
    @property
    def velocity(self) -> float:
        ...
    @velocity.setter
    def velocity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def wavelength_nm(self) -> float:
        ...
    @wavelength_nm.setter
    def wavelength_nm(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def weight(self) -> float:
        ...
    @weight.setter
    def weight(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class PhotonRecord:
    @property
    def arrival_time(self) -> float:
        ...
    @property
    def direction(self) -> Vec3:
        ...
    @property
    def events(self) -> int:
        ...
    @property
    def k(self) -> float:
        ...
    @property
    def launch_time(self) -> float:
        ...
    @property
    def m(self) -> Vec3:
        ...
    @property
    def n(self) -> Vec3:
        ...
    @property
    def opticalpath(self) -> float:
        ...
    @property
    def penetration_depth(self) -> float:
        ...
    @property
    def polarization_forward(self) -> CVec2:
        ...
    @property
    def polarization_reverse(self) -> CVec2:
        ...
    @property
    def position_detector(self) -> Vec3:
        ...
    @property
    def position_first_scattering(self) -> Vec3:
        ...
    @property
    def position_last_scattering(self) -> Vec3:
        ...
    @property
    def weight(self) -> float:
        ...
class PhotonRecordSensor(Sensor):
    def __init__(self, z: typing.SupportsFloat | typing.SupportsIndex, absorb: bool = True) -> None:
        """
        Sensor that stores a full PhotonRecord for each detected photon at plane z
        """
    @property
    def recorded_photons(self) -> list[PhotonRecord]:
        ...
class PlanarFieldProcessed:
    @property
    def Ex(self) -> CMatrix:
        ...
    @property
    def Ey(self) -> CMatrix:
        ...
class PlanarFieldSensor(Sensor):
    def __init__(self, z: typing.SupportsFloat | typing.SupportsIndex, len_x: typing.SupportsFloat | typing.SupportsIndex, len_y: typing.SupportsFloat | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, dy: typing.SupportsFloat | typing.SupportsIndex, absorb: bool = True, estimator: bool = False) -> None:
        """
        Sensor accumulating the complex electric field on an N_x x N_y grid at plane z
        """
    @property
    def Ex(self) -> CMatrix:
        ...
    @property
    def Ey(self) -> CMatrix:
        ...
    @property
    def N_x(self) -> int:
        ...
    @property
    def N_y(self) -> int:
        ...
    @property
    def dx(self) -> float:
        ...
    @property
    def dy(self) -> float:
        ...
    @property
    def len_x(self) -> float:
        ...
    @property
    def len_y(self) -> float:
        ...
class PlanarFluenceProcessed:
    @property
    def S0(self) -> list[Matrix]:
        ...
    @property
    def S1(self) -> list[Matrix]:
        ...
    @property
    def S2(self) -> list[Matrix]:
        ...
    @property
    def S3(self) -> list[Matrix]:
        ...
class PlanarFluenceSensor(Sensor):
    def __init__(self, z: typing.SupportsFloat | typing.SupportsIndex, len_x: typing.SupportsFloat | typing.SupportsIndex, len_y: typing.SupportsFloat | typing.SupportsIndex, len_t: typing.SupportsFloat | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, dy: typing.SupportsFloat | typing.SupportsIndex, dt: typing.SupportsFloat | typing.SupportsIndex, absorb: bool = True, estimator: bool = False) -> None:
        """
        Sensor accumulating Stokes parameters on a spatial (optionally time-resolved) grid at plane z
        """
    @property
    def N_t(self) -> int:
        ...
    @property
    def N_x(self) -> int:
        ...
    @property
    def N_y(self) -> int:
        ...
    @property
    def S0(self) -> list[Matrix]:
        ...
    @property
    def S1(self) -> list[Matrix]:
        ...
    @property
    def S2(self) -> list[Matrix]:
        ...
    @property
    def S3(self) -> list[Matrix]:
        ...
    @property
    def dt(self) -> float:
        ...
    @property
    def dx(self) -> float:
        ...
    @property
    def dy(self) -> float:
        ...
    @property
    def len_t(self) -> float:
        ...
    @property
    def len_x(self) -> float:
        ...
    @property
    def len_y(self) -> float:
        ...
class RGDMedium(ScatteringMedium):
    def __init__(self, phase_func: PhaseFunction, radius: typing.SupportsFloat | typing.SupportsIndex, n_particle: typing.SupportsFloat | typing.SupportsIndex, n_medium: typing.SupportsFloat | typing.SupportsIndex, wavelength: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_mean_free_path(self, mfp: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set the mean free path used for free-path sampling (does NOT update mu_s/mu_t)
        """
    @property
    def mean_free_path(self) -> float:
        ...
    @property
    def radius(self) -> float:
        ...
class RayleighDebyeEMCPhaseFunction(PhaseFunction):
    def __init__(self, wavelength: typing.SupportsFloat | typing.SupportsIndex, radius: typing.SupportsFloat | typing.SupportsIndex, n_particle: typing.SupportsFloat | typing.SupportsIndex, n_medium: typing.SupportsFloat | typing.SupportsIndex, nDiv: typing.SupportsInt | typing.SupportsIndex, minVal: typing.SupportsFloat | typing.SupportsIndex, maxVal: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def pdf(self, x: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def rho_phase_function(self, x: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
class RayleighDebyePhaseFunction(PhaseFunction):
    def __init__(self, wavelength: typing.SupportsFloat | typing.SupportsIndex, radius: typing.SupportsFloat | typing.SupportsIndex, nDiv: typing.SupportsInt | typing.SupportsIndex, minVal: typing.SupportsFloat | typing.SupportsIndex, maxVal: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def pdf(self, x: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
class RayleighPhaseFunction(PhaseFunction):
    def __init__(self, nDiv: typing.SupportsInt | typing.SupportsIndex, minVal: typing.SupportsFloat | typing.SupportsIndex, maxVal: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def pdf(self, x: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
class Rng:
    def __init__(self, seed: typing.SupportsInt | typing.SupportsIndex = 3820158083) -> None:
        """
        Initialize the RNG with an optional seed
        """
    def normal(self, mean: typing.SupportsFloat | typing.SupportsIndex, stddev: typing.SupportsFloat | typing.SupportsIndex) -> float:
        """
        Generate a normally distributed random number with given mean and stddev
        """
    def uniform(self) -> float:
        """
        Generate a uniform random number in [0, 1)
        """
class Sample:
    def __init__(self, n_medium: typing.SupportsFloat | typing.SupportsIndex = 1.0) -> None:
        """
        Construct a Sample with the given host medium refractive index
        """
    def add_layer(self, medium: ScatteringMedium, z_min: typing.SupportsFloat | typing.SupportsIndex, z_max: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Add a new layer to the top of the sample
        """
    def add_mixture_layer(self, species: collections.abc.Sequence[ScatteringMedium], number_densities: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], z_min: typing.SupportsFloat | typing.SupportsIndex, z_max: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Add a mixture layer (several co-located species) to the top of the sample
        """
    def get_layer(self, index: typing.SupportsInt | typing.SupportsIndex) -> Layer:
        """
        Access a layer by index
        """
    def is_inside(self, position: Vec3) -> bool:
        """
        Test whether a position is inside the sample
        """
    def light_speed_in_medium(self) -> float:
        """
        Return the phase speed of light in the host medium
        """
    def size(self) -> int:
        """
        Return the number of layers
        """
    def z_top(self) -> float:
        """
        Return the z_max of the topmost layer
        """
    @property
    def interfaces(self) -> list[float]:
        ...
    @property
    def layers(self) -> list[Layer]:
        """
        List of the sample's layers (polymorphic).
        """
    @property
    def light_speed(self) -> float:
        ...
    @property
    def refractive_index(self) -> float:
        ...
class SampleLayer(Layer):
    def __init__(self, medium: ScatteringMedium, z_min: typing.SupportsFloat | typing.SupportsIndex, z_max: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def medium(self) -> ScatteringMedium:
        ...
class ScatteringMedium:
    def sample_azimuthal_angle(self, rng: Rng) -> float:
        """
        Sample the azimuthal angle in the medium
        """
    def sample_conditional_azimuthal_angle(self, rng: Rng, S: CMatrix, E: CVec2, theta: typing.SupportsFloat | typing.SupportsIndex) -> float:
        """
        Sample the azimuthal angle conditioned on scattering angle theta
        """
    def sample_free_path(self, rng: Rng) -> float:
        """
        Sample the free path length in the medium
        """
    def sample_scattering_angle(self, rng: Rng) -> float:
        """
        Sample the scattering angle in the medium
        """
    def scattering_cross_section(self) -> float:
        """
        Single-particle scattering cross-section sigma_s [mm^2] (from the phase function)
        """
    def scattering_matrix(self, theta: typing.SupportsFloat | typing.SupportsIndex, phi: typing.SupportsFloat | typing.SupportsIndex) -> CMatrix:
        """
        Get the scattering matrix for given angles and wavenumber
        """
    def set_absorption_coefficient(self, mu_a: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set the absorption coefficient mu_a for the medium
        """
    def set_scattering_coefficient(self, mu_s: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set the scattering coefficient mu_s for the medium
        """
    @property
    def k(self) -> float:
        ...
    @property
    def mu_a(self) -> float:
        ...
    @property
    def mu_s(self) -> float:
        ...
    @property
    def mu_t(self) -> float:
        ...
    @property
    def n_medium(self) -> float:
        ...
    @property
    def n_particle(self) -> float:
        ...
    @property
    def phase_function(self) -> PhaseFunction:
        ...
    @property
    def wavelength(self) -> float:
        ...
class Sensor:
    def set_direction_limit(self, direction: CrossingDirection) -> None:
        """
        Set the crossing direction filter (Forward, Backward, or Both)
        """
    def set_events_limit(self, min_events: typing.SupportsInt | typing.SupportsIndex, max_events: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the limits for the number of scattering events (inclusive)
        """
    def set_phi_limit(self, min_phi: typing.SupportsFloat | typing.SupportsIndex, max_phi: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set the azimuthal angle detection limits (in radians)
        """
    def set_position_limit(self, x_min: typing.SupportsFloat | typing.SupportsIndex, x_max: typing.SupportsFloat | typing.SupportsIndex, y_min: typing.SupportsFloat | typing.SupportsIndex, y_max: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set the detection limits for the position on the sensor plane
        """
    def set_theta_limit(self, min_theta: typing.SupportsFloat | typing.SupportsIndex, max_theta: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set the polar angle detection limits (in radians)
        """
    @property
    def absorb_photons(self) -> bool:
        ...
    @property
    def backward_normal(self) -> Vec3:
        ...
    @property
    def estimator_enabled(self) -> bool:
        ...
    @property
    def filter_direction(self) -> CrossingDirection:
        ...
    @property
    def filter_direction_enabled(self) -> bool:
        ...
    @property
    def filter_phi_enabled(self) -> bool:
        ...
    @property
    def filter_phi_max(self) -> float:
        ...
    @property
    def filter_phi_min(self) -> float:
        ...
    @property
    def filter_position_enabled(self) -> bool:
        ...
    @property
    def filter_theta_enabled(self) -> bool:
        ...
    @property
    def filter_theta_max(self) -> float:
        ...
    @property
    def filter_theta_min(self) -> float:
        ...
    @property
    def filter_x_max(self) -> float:
        ...
    @property
    def filter_x_min(self) -> float:
        ...
    @property
    def filter_y_max(self) -> float:
        ...
    @property
    def filter_y_min(self) -> float:
        ...
    @property
    def hits(self) -> int:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def m_polarization(self) -> Vec3:
        ...
    @property
    def n_polarization(self) -> Vec3:
        ...
    @property
    def normal(self) -> Vec3:
        ...
    @property
    def origin(self) -> Vec3:
        ...
class SensorsGroup:
    def __init__(self) -> None:
        ...
    def add_detector(self, detector: Sensor) -> Sensor:
        """
        Add a (cloned) detector to the group and return a reference to the internal copy
        """
    @property
    def detectors(self) -> list[Sensor]:
        ...
class SimConfig:
    absorption: Absorption
    detector: SensorsGroup
    laser: Laser
    pin_threads_to_cores: bool
    sample: Sample
    show_progress: bool
    track_reverse_paths: bool
    def __init__(self) -> None:
        """
        Create a default simulation configuration
        """
    @property
    def MAX_EVENTS(self) -> int:
        ...
    @MAX_EVENTS.setter
    def MAX_EVENTS(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def batch_size(self) -> int:
        ...
    @batch_size.setter
    def batch_size(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def n_photons(self) -> int:
        ...
    @n_photons.setter
    def n_photons(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def n_threads(self) -> int:
        ...
    @n_threads.setter
    def n_threads(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def progress_interval_pct(self) -> int:
        ...
    @progress_interval_pct.setter
    def progress_interval_pct(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def seed(self) -> int:
        ...
    @seed.setter
    def seed(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class StatisticsSensor(Sensor):
    def __init__(self, z: typing.SupportsFloat | typing.SupportsIndex, absorb: bool = False) -> None:
        """
        Sensor accumulating configurable histograms of detected photon properties at plane z
        """
    def set_depth_histogram_bins(self, max_depth: typing.SupportsFloat | typing.SupportsIndex, n_bins: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the bins for the depth histogram
        """
    def set_events_histogram_bins(self, max_events: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the bins for the events histogram
        """
    def set_phi_histogram_bins(self, min_phi: typing.SupportsFloat | typing.SupportsIndex, max_phi: typing.SupportsFloat | typing.SupportsIndex, n_bins: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the bins for the phi histogram
        """
    def set_theta_histogram_bins(self, min_theta: typing.SupportsFloat | typing.SupportsIndex, max_theta: typing.SupportsFloat | typing.SupportsIndex, n_bins: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the bins for the theta histogram
        """
    def set_time_histogram_bins(self, max_time: typing.SupportsFloat | typing.SupportsIndex, n_bins: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the bins for the time histogram
        """
    def set_time_resolution(self, len_t: typing.SupportsFloat | typing.SupportsIndex, dt: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Configure temporal resolution: bin 0 integrated, bins >=1 time windows
        """
    def set_weight_histogram_bins(self, max_weight: typing.SupportsFloat | typing.SupportsIndex, n_bins: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the bins for the weight histogram
        """
    @property
    def N_t(self) -> int:
        ...
    @property
    def ddepth(self) -> float:
        ...
    @property
    def depth_histogram(self) -> list[list[int]]:
        ...
    @property
    def depth_histogram_bins_set(self) -> bool:
        ...
    @property
    def dphi(self) -> float:
        ...
    @property
    def dt(self) -> float:
        ...
    @property
    def dtheta(self) -> float:
        ...
    @property
    def dweight(self) -> float:
        ...
    @property
    def events_histogram(self) -> list[list[int]]:
        ...
    @property
    def events_histogram_bins_set(self) -> bool:
        ...
    @property
    def h_dtime(self) -> float:
        ...
    @property
    def h_max_time(self) -> float:
        ...
    @property
    def max_depth(self) -> float:
        ...
    @property
    def max_events(self) -> int:
        ...
    @property
    def max_phi(self) -> float:
        ...
    @property
    def max_theta(self) -> float:
        ...
    @property
    def max_weight(self) -> float:
        ...
    @property
    def min_phi(self) -> float:
        ...
    @property
    def min_theta(self) -> float:
        ...
    @property
    def n_bins_depth(self) -> int:
        ...
    @property
    def n_bins_phi(self) -> int:
        ...
    @property
    def n_bins_theta(self) -> int:
        ...
    @property
    def n_bins_time(self) -> int:
        ...
    @property
    def n_bins_weight(self) -> int:
        ...
    @property
    def phi_histogram(self) -> list[list[int]]:
        ...
    @property
    def phi_histogram_bins_set(self) -> bool:
        ...
    @property
    def t_max(self) -> float:
        ...
    @property
    def theta_histogram(self) -> list[list[int]]:
        ...
    @property
    def theta_histogram_bins_set(self) -> bool:
        ...
    @property
    def time_histogram(self) -> list[int]:
        ...
    @property
    def time_histogram_bins_set(self) -> bool:
        ...
    @property
    def weight_histogram(self) -> list[int]:
        ...
    @property
    def weight_histogram_bins_set(self) -> bool:
        ...
class StokesMatrixProcessed:
    @property
    def S0(self) -> Matrix:
        ...
    @property
    def S1(self) -> Matrix:
        ...
    @property
    def S2(self) -> Matrix:
        ...
    @property
    def S3(self) -> Matrix:
        ...
class StokesRadialProcessed:
    @property
    def S0(self) -> list[float]:
        ...
    @property
    def S1(self) -> list[float]:
        ...
    @property
    def S2(self) -> list[float]:
        ...
    @property
    def S3(self) -> list[float]:
        ...
class TargetDistribution:
    def evaluate(self, x: typing.SupportsFloat | typing.SupportsIndex) -> float:
        """
        Evaluate the target distribution at x
        """
class TemporalProfile:
    """
    Members:
    
      Delta
    
      Gaussian
    
      TopHat
    
      ExponentialTime
    
      PulseTrain
    
      CW
    """
    CW: typing.ClassVar[TemporalProfile]  # value = <TemporalProfile.CW: 5>
    Delta: typing.ClassVar[TemporalProfile]  # value = <TemporalProfile.Delta: 0>
    ExponentialTime: typing.ClassVar[TemporalProfile]  # value = <TemporalProfile.ExponentialTime: 3>
    Gaussian: typing.ClassVar[TemporalProfile]  # value = <TemporalProfile.Gaussian: 1>
    PulseTrain: typing.ClassVar[TemporalProfile]  # value = <TemporalProfile.PulseTrain: 4>
    TopHat: typing.ClassVar[TemporalProfile]  # value = <TemporalProfile.TopHat: 2>
    __members__: typing.ClassVar[dict[str, TemporalProfile]]  # value = {'Delta': <TemporalProfile.Delta: 0>, 'Gaussian': <TemporalProfile.Gaussian: 1>, 'TopHat': <TemporalProfile.TopHat: 2>, 'ExponentialTime': <TemporalProfile.ExponentialTime: 3>, 'PulseTrain': <TemporalProfile.PulseTrain: 4>, 'CW': <TemporalProfile.CW: 5>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class UniformPhaseFunction(PhaseFunction):
    def __init__(self) -> None:
        ...
class Vec2:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def x(self) -> float:
        ...
    @x.setter
    def x(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def y(self) -> float:
        ...
    @y.setter
    def y(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class Vec3:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, z: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def x(self) -> float:
        ...
    @x.setter
    def x(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def y(self) -> float:
        ...
    @y.setter
    def y(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def z(self) -> float:
        ...
    @z.setter
    def z(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
def calculate_rotation_angle(n_from: Vec3, n_to: Vec3) -> float:
    """
    Calculate the rotation angle between two normal vectors
    """
def cross(a: Vec3, b: Vec3) -> Vec3:
    """
    Compute the cross product of two Vec3 vectors
    """
def dot(a: Vec3, b: Vec3) -> float:
    """
    Compute the dot product of two Vec3 vectors
    """
def form_factor(theta: typing.SupportsFloat | typing.SupportsIndex, k: typing.SupportsFloat | typing.SupportsIndex, radius: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute the form factor for given scattering angle, wavenumber, and particle radius
    """
def gaussian_distribution(rng: Rng, center: Vec3, sigma: typing.SupportsFloat | typing.SupportsIndex) -> Vec3:
    """
    Generate a random point from Gaussian distribution
    """
def matcmul(A: CMatrix, B: CMatrix, C: CMatrix) -> None:
    """
    Multiply two complex matrices A and B, storing result in C
    """
def matcmulscalar(scalar: typing.SupportsFloat | typing.SupportsIndex, A: CMatrix) -> None:
    """
    Multiply a complex matrix by a scalar
    """
def norm(v: Vec3) -> float:
    """
    Compute the norm of a Vec3 vector
    """
def postprocess_farfield_cbs(det: FarFieldCBSSensor, n_photons: typing.SupportsInt | typing.SupportsIndex, normalize_per_solid_angle: bool = True, normalize_per_photon: bool = True, eps: typing.SupportsFloat | typing.SupportsIndex = 1e-30) -> FarFieldCBSProcessed:
    """
    Normalize a FarFieldCBSSensor's coherent/incoherent Stokes grids (per solid angle and/or per photon)
    """
def postprocess_planar_field(det: PlanarFieldSensor, n_photons: typing.SupportsInt | typing.SupportsIndex, normalize_per_photon: bool = True, normalize_per_area: bool = True, eps: typing.SupportsFloat | typing.SupportsIndex = 1e-30) -> PlanarFieldProcessed:
    """
    Normalize a PlanarFieldSensor's accumulated complex field (per photon and/or per pixel area)
    """
def postprocess_planar_fluence(det: PlanarFluenceSensor, n_photons: typing.SupportsInt | typing.SupportsIndex, normalize_per_photon: bool = True, normalize_per_area: bool = True, eps: typing.SupportsFloat | typing.SupportsIndex = 1e-30) -> PlanarFluenceProcessed:
    """
    Normalize a PlanarFluenceSensor's accumulated Stokes grids (per photon and/or per pixel area)
    """
def run_simulation_parallel(config: SimConfig) -> None:
    """
    Run the Monte Carlo simulation in parallel with the given configuration, medium, detector, and laser
    """
def set_log_level(level: LogLevel) -> None:
    """
    Set the logging level for the luminis-mc module
    """
def uniform_distribution(rng: Rng, center: Vec3, sigma: typing.SupportsFloat | typing.SupportsIndex) -> Vec3:
    """
    Generate a random point from uniform distribution
    """
Backward: CrossingDirection  # value = <CrossingDirection.Backward: 1>
Both: CrossingDirection  # value = <CrossingDirection.Both: 2>
CW: TemporalProfile  # value = <TemporalProfile.CW: 5>
Delta: TemporalProfile  # value = <TemporalProfile.Delta: 0>
ExponentialTime: TemporalProfile  # value = <TemporalProfile.ExponentialTime: 3>
Forward: CrossingDirection  # value = <CrossingDirection.Forward: 0>
Gaussian: TemporalProfile  # value = <TemporalProfile.Gaussian: 1>
Point: LaserSource  # value = <LaserSource.Point: 0>
PulseTrain: TemporalProfile  # value = <TemporalProfile.PulseTrain: 4>
TopHat: TemporalProfile  # value = <TemporalProfile.TopHat: 2>
Uniform: LaserSource  # value = <LaserSource.Uniform: 1>
debug: LogLevel  # value = <LogLevel.debug: 1>
error: LogLevel  # value = <LogLevel.error: 4>
info: LogLevel  # value = <LogLevel.info: 2>
off: LogLevel  # value = <LogLevel.off: 6>
warn: LogLevel  # value = <LogLevel.warn: 3>
