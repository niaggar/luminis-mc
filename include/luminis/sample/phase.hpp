/**
 * @file phase.hpp
 * @brief Phase-function hierarchy for sampling scattering angles.
 *
 * A `PhaseFunction` samples the polar scattering angle θ (and azimuthal angle φ)
 * for a single scattering event. Each concrete model implements a different
 * angular distribution; most precompute a `SamplingTable` (inverse-CDF) at
 * construction so that per-event sampling is a cheap table look-up.
 *
 * @see table.hpp  — SamplingTable used for inverse-CDF sampling
 * @see medium.hpp — ScatteringMedium that delegates angle sampling here
 */

#pragma once
#include <luminis/sample/table.hpp>
#include <luminis/math/vec.hpp>
#include <luminis/math/rng.hpp>
#include <complex>

using namespace luminis::math;

namespace luminis::sample {

/// @brief Rayleigh-Gans-Debye form factor F(θ) for a homogeneous sphere.
/// @param theta  Polar scattering angle [rad].
/// @param k      Wave number in the medium [1/mm].
/// @param radius Particle radius [mm].
double form_factor(const double theta, const double k, const double radius);

/// @brief Abstract phase function: samples scattering angles for one event.
///
/// Derived classes must implement `sample_cos()` and `sample_theta()`.
/// `sample_phi()` is uniform by default; `sample_phi_conditional()` provides a
/// polarization-aware azimuthal sampler for CBS transport.
class PhaseFunction {
public:
  virtual ~PhaseFunction() = default;

  /// @brief Sample the azimuthal angle φ from a uniform input `x ∈ [0,1)`. @return φ ∈ [0, 2π).
  virtual double sample_phi(double x) const;

  /// @brief Sample φ conditioned on the current polarization state (CBS transport).
  /// @param theta Polar angle of this event [rad].
  /// @param S     2×2 amplitude scattering matrix at θ.
  /// @param E     Current Jones vector.
  /// @param k     Wave number [1/mm].
  /// @param rng   Random number generator.
  /// @return φ ∈ [0, 2π).
  virtual double sample_phi_conditional(double theta, CMatrix& S, CVec2& E, double k, Rng& rng) const;

  /// @brief Sample cos(θ) from a uniform input `x ∈ [0,1)`.
  virtual double sample_cos(double x) const = 0;

  /// @brief Sample the polar angle θ from a uniform input `x ∈ [0,1)`. @return θ ∈ [0, π].
  virtual double sample_theta(double x) const = 0;

  /// @brief Scattering efficiency Q_sca (dimensionless). Defaults to 1 unless overridden.
  virtual double scattering_efficiency() const;

  /// @brief Scattering cross-section σ_sca [mm²]. Defaults to 0 unless overridden.
  virtual double scattering_cross_section() const;

  /// @brief Estimate the anisotropy factor g = <cos θ> by Monte Carlo sampling.
  /// @param n_samples Number of samples to average over.
  /// @return {g, variance estimate}.
  std::array<double, 2> get_anisotropy_factor(std::size_t n_samples = 200000) const;
};

/// @brief Isotropic phase function (uniform over the sphere).
class UniformPhaseFunction : public PhaseFunction {
public:
  double sample_cos(double x) const override;
  double sample_theta(double x) const override;
};

/// @brief Rayleigh phase function ∝ (1 + cos²θ), for particles much smaller than λ.
class RayleighPhaseFunction : public PhaseFunction {
public:
  /// @param nDiv   Number of sampling-table divisions.
  /// @param minVal Lower angle bound [rad].
  /// @param maxVal Upper angle bound [rad].
  RayleighPhaseFunction(int nDiv, double minVal, double maxVal);
  double sample_cos(double x) const override;
  double sample_theta(double x) const override;
  double PDF(double x); ///< Unnormalized angular probability density at θ = x.
private:
  SamplingTable table;
};

/// @brief Henyey-Greenstein phase function parameterized by anisotropy g.
class HenyeyGreensteinPhaseFunction : public PhaseFunction {
public:
  /// @param g Anisotropy factor g = <cos θ> ∈ (-1, 1).
  HenyeyGreensteinPhaseFunction(double g);
  double sample_cos(double x) const override;
  double sample_theta(double x) const override;
private:
  double g; ///< Anisotropy factor.
};

/// @brief Rayleigh-Debye phase function using the RGD form factor.
class RayleighDebyePhaseFunction : public PhaseFunction {
public:
  /// @param wavelenght Vacuum wavelength [mm].
  /// @param radius     Particle radius [mm].
  /// @param nDiv       Number of sampling-table divisions.
  /// @param minVal     Lower angle bound [rad].
  /// @param maxVal     Upper angle bound [rad].
  RayleighDebyePhaseFunction(double wavelenght, double radius, int nDiv, double minVal, double maxVal);
  double sample_cos(double x) const override;
  double sample_theta(double x) const override;
  double PDF(double x); ///< Unnormalized angular probability density at θ = x.
private:
  SamplingTable table;
  double wavelength;
  double radius;
  double k;
};

/// @brief Rayleigh-Debye phase function with electromagnetic correction factor.
///
/// Extends RayleighDebyePhaseFunction with the polarization/contrast correction,
/// allowing it to report the scattering efficiency and cross-section.
class RayleighDebyeEMCPhaseFunction : public PhaseFunction {
public:
  /// @param wavelenght Vacuum wavelength [mm].
  /// @param radius     Particle radius [mm].
  /// @param n_particle Particle refractive index.
  /// @param n_medium   Host medium refractive index.
  /// @param nDiv       Number of sampling-table divisions.
  /// @param minVal     Lower angle bound [rad].
  /// @param maxVal     Upper angle bound [rad].
  RayleighDebyeEMCPhaseFunction(double wavelenght, double radius, double n_particle, double n_medium, int nDiv, double minVal, double maxVal);
  double sample_cos(double x) const override;
  double sample_theta(double x) const override;
  double scattering_efficiency() const override;
  double scattering_cross_section() const override;
  double rho_phase_function(double x) const; ///< Normalized phase-function value at θ = x.
  double PDF(double x) const;                ///< Unnormalized angular probability density at θ = x.
private:
  SamplingTable table;
  double wavelength;
  double radius;
  double k;
  double n_particle;
  double scattering_cross_section_value;
  double n_medium;
};

/// @brief Draine phase function: a generalized Henyey-Greenstein with parameter `a`.
class DrainePhaseFunction : public PhaseFunction {
public:
  /// @param g      Anisotropy factor.
  /// @param a      Draine shape parameter.
  /// @param nDiv   Number of sampling-table divisions.
  /// @param minVal Lower angle bound [rad].
  /// @param maxVal Upper angle bound [rad].
  DrainePhaseFunction(double g, double a, int nDiv, double minVal, double maxVal);
  double sample_cos(double x) const override;
  double sample_theta(double x) const override;
  double PDF(double x); ///< Unnormalized angular probability density at θ = x.
private:
  SamplingTable table;
  double g; ///< Anisotropy factor.
  double a; ///< Draine parameter.
};

/// @brief Full Mie phase function for spheres of arbitrary size parameter.
///
/// Builds the angular distribution from Mie theory; also reports the Mie
/// scattering efficiency and cross-section.
class MiePhaseFunction : public PhaseFunction {
public:
  /// @param wavelength Vacuum wavelength [mm].
  /// @param radius     Particle radius [mm].
  /// @param n_particle Particle refractive index.
  /// @param n_medium   Host medium refractive index.
  /// @param nDiv       Number of sampling-table divisions.
  /// @param minVal     Lower angle bound [rad].
  /// @param maxVal     Upper angle bound [rad].
  MiePhaseFunction(double wavelength, double radius, double n_particle, double n_medium, int nDiv, double minVal, double maxVal);
  double sample_cos(double x) const override;
  double sample_theta(double x) const override;
  double scattering_efficiency() const override;
  double scattering_cross_section() const override;
  double rho_phase_function(double x) const; ///< Normalized phase-function value at θ = x.
  double PDF(double x) const;                ///< Unnormalized angular probability density at θ = x.
private:
  SamplingTable table;
  double wavelength;
  double radius;
  double k;
  double n_particle;
  double n_medium;
  double scattering_cross_section_value;
  std::complex<double> m; ///< Relative refractive index m = n_particle / n_medium.
};

} // namespace luminis::sample
