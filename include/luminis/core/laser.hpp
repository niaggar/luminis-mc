/**
 * @file laser.hpp
 * @brief Photon source (laser) with configurable spatial and temporal profiles.
 *
 * Defines the `Laser` emitter used to seed Monte Carlo transport. Each call to
 * `emit_photon()` produces a fresh `Photon` sampled from the configured spatial
 * source (point / uniform disk / Gaussian beam) and temporal profile, with the
 * initial polarization set as a Jones vector in the local (m, n) frame.
 *
 * @see photon.hpp     — Photon packet produced by emit_photon()
 * @see simulation.hpp — transport loop that consumes the emitted photons
 */

#pragma once
#include <luminis/core/photon.hpp>
#include <luminis/math/rng.hpp>
#include <luminis/math/vec.hpp>

using namespace luminis::math;

namespace luminis::core {

/// @brief Spatial distribution of the photon launch position.
enum class LaserSource {
  Point = 0,    ///< All photons launched from a single point.
  Uniform = 1,  ///< Uniformly distributed over a disk of radius `sigma`.
  Gaussian = 2, ///< Gaussian-distributed around the center with standard deviation `sigma`.
};

/// @brief Temporal distribution of the photon emission time.
enum class TemporalProfile {
  Delta,       ///< Instantaneous pulse (all photons at t = time_offset).
  Gaussian,    ///< Gaussian pulse of width `pulse_duration`.
  TopHat,      ///< Uniform pulse over `pulse_duration`.
  Exponential, ///< Exponentially decaying emission.
  PulseTrain,  ///< Periodic pulses at `repetition_rate`.
  CW           ///< Continuous-wave (uniform over the repetition period).
};

/// @brief Photon source emitting packets with a given spatial and temporal profile.
///
/// The beam axis is `direction`; `local_m`/`local_n` form the transverse frame
/// in which the Jones `polarization` is expressed. `emit_photon()` samples the
/// launch position and emission time from the configured profiles.
struct Laser {
  Vec3 position;       ///< Beam center / launch origin [mm].
  Vec3 direction;      ///< Propagation direction unit vector.
  Vec3 local_m;        ///< Transverse basis vector m (defines the polarization frame).
  Vec3 local_n;        ///< Transverse basis vector n (perpendicular to m and direction).
  CVec2 polarization;  ///< Initial Jones vector in the (m, n) frame.

  double wavelength;        ///< Free-space wavelength [nm].
  double sigma;             ///< Spatial spread: disk radius (Uniform) or std-dev (Gaussian) [mm].
  LaserSource source_type;  ///< Spatial source distribution.

  TemporalProfile temporal_profile{TemporalProfile::Delta}; ///< Temporal emission profile.
  double pulse_duration{0.0};  ///< Pulse width [ps] (interpretation depends on the profile).
  double repetition_rate{0.0}; ///< Pulse repetition rate [Hz] (PulseTrain / CW).
  double time_offset{0.0};     ///< Emission time offset [ps].

  /// @brief Construct a laser from its initial Jones state and spatial profile.
  /// @param m_state     Complex amplitude along the m-axis.
  /// @param n_state     Complex amplitude along the n-axis.
  /// @param wavelength  Free-space wavelength [nm].
  /// @param sigma       Spatial spread (disk radius or Gaussian std-dev) [mm].
  /// @param source_type Spatial source distribution.
  Laser(std::complex<double> m_state, std::complex<double> n_state, double wavelength, double sigma, LaserSource source_type);

  /// @brief Configure the temporal emission profile.
  /// @param profile         Temporal profile to use.
  /// @param pulse_duration  Pulse width [ps].
  /// @param repetition_rate Pulse repetition rate [Hz].
  /// @param time_offset     Emission time offset [ps].
  void set_temporal_profile(TemporalProfile profile, double pulse_duration = 0.0, double repetition_rate = 0.0, double time_offset = 0.0) {
    temporal_profile = profile;
    this->pulse_duration = pulse_duration;
    this->repetition_rate = repetition_rate;
    this->time_offset = time_offset;
  }

  /// @brief Sample an emission time from the configured temporal profile.
  /// @param rng Random number generator.
  /// @return Emission time [ns].
  double sample_emission_time(Rng &rng) const;

  /// @brief Emit a new photon sampled from the spatial and temporal profiles.
  /// @param rng Random number generator.
  /// @return A fully initialized Photon ready for transport.
  Photon emit_photon(Rng &rng) const;
};

/// @brief Sample a point uniformly within a disk of radius `sigma` around `center`.
Vec3 uniform_distribution(Rng &rng, const Vec3 &center, const double sigma);

/// @brief Sample a point from a 2D Gaussian of std-dev `sigma` around `center`.
Vec3 gaussian_distribution(Rng &rng, const Vec3 &center, const double sigma);

} // namespace luminis::core
