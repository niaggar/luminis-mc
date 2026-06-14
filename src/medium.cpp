/**
 * @file medium.cpp
 * @brief Implementations of ScatteringMedium, RGDMedium, and MieMedium.
 *
 * Provides the base-class method bodies (free-path sampling delegation,
 * angle sampling via phase function) and the two concrete
 * derived-class constructors plus their `sample_free_path`,
 * `scattering_matrix`, and (for MieMedium) `precompute_scattering_tables`.
 *
 * @see include/luminis/core/medium.hpp
 */
#include "luminis/math/vec.hpp"
#include <cmath>
#include <complex>
#include <luminis/core/medium.hpp>
#include <luminis/log/logger.hpp>
#include <luminis/sample/meanfreepath.hpp>
#include "luminis/mie/dmiev.h"

namespace luminis::core
{

// ══════════════════════════════════════════════════════════════════════════════
//  ScatteringMedium — base class
// ══════════════════════════════════════════════════════════════════════════════

    /// Initialises the phase-function pointer; optical coefficients default to zero.
    ScatteringMedium::ScatteringMedium(PhaseFunction *phase_func)
      : phase_function(phase_func) {}

  double ScatteringMedium::sample_azimuthal_angle(Rng &rng) const
  {
    if (phase_function)
    {
      return phase_function->sample_phi(rng.uniform());
    }
    LLOG_ERROR("ScatteringMedium::sample_azimuthal_angle: Phase function is not defined!");
    std::exit(EXIT_FAILURE);
  }

  double ScatteringMedium::sample_conditional_azimuthal_angle(Rng &rng, CMatrix &S, CVec2 &E, double theta) const
  {
    if (phase_function)
    {
      return phase_function->sample_phi_conditional(theta, S, E, k, rng);
    }
    LLOG_ERROR("ScatteringMedium::sample_conditional_azimuthal_angle: Phase function is not defined!");
    std::exit(EXIT_FAILURE);
  }

  double ScatteringMedium::sample_scattering_angle(Rng &rng) const
  {
    if (phase_function)
    {
      return phase_function->sample_theta(rng.uniform());
    }
    LLOG_ERROR("ScatteringMedium::sample_scattering_angle: Phase function is not defined!");
    std::exit(EXIT_FAILURE);
  }

  void ScatteringMedium::set_scattering_coefficient(double mu_s)
  {
    mu_scattering = mu_s;
    mu_attenuation = mu_absorption + mu_scattering;
  }

  void ScatteringMedium::set_absorption_coefficient(double mu_a)
  {
    mu_absorption = mu_a;
    mu_attenuation = mu_absorption + mu_scattering;
  }

// ══════════════════════════════════════════════════════════════════════════════
//  RGDMedium — Rayleigh-Gans-Debye (RGD) approximation
// ══════════════════════════════════════════════════════════════════════════════

  RGDMedium::RGDMedium(PhaseFunction *phase_func, double r, double n_particle, double n_medium, double wavelength)
      : ScatteringMedium(phase_func)
  {
    radius = r;
    this->n_particle = n_particle;
    this->n_medium = n_medium;
    this->wavelength = wavelength;
    this->k = 2 * M_PI * n_medium / wavelength; // Compute wave number from wavelength

    // Precompute the RGD amplitude prefactor -k³ (m-1) V / (2π).  This is a
    // medium-level constant, so the per-event scattering matrix reduces to a
    // couple of multiplies instead of recomputing pow(k,3)/pow(r,3) every event.
    const double kkk = k * k * k;
    const double volume = 4.0 * M_PI * (radius * radius * radius) / 3.0;
    const double mm = (n_particle / n_medium) - 1.0; // contrast index (m - 1)
    this->rgd_prefactor = -1.0 * kkk * mm * volume / (2.0 * M_PI);

    // Tabulate s1(θ)/s2(θ) on a uniform θ grid so the hot-path scattering_matrix
    // is a couple of O(1) look-ups instead of recomputing the form factor.
    precompute_scattering_tables(2000);
  }

  void RGDMedium::precompute_scattering_tables(std::size_t n_samples)
  {
    if (n_samples < 2)
      n_samples = 2;

    const double dtheta = M_PI / static_cast<double>(n_samples - 1);

    std::vector<std::complex<double>> s1_values(n_samples);
    std::vector<std::complex<double>> s2_values(n_samples);

    for (std::size_t i = 0; i < n_samples; ++i)
    {
      const double theta = static_cast<double>(i) * dtheta;
      const double F = form_factor(theta, k, radius);
      // s2 = -i k³ (m-1) V F cosθ / (2π),  s1 = -i k³ (m-1) V F / (2π).
      s1_values[i] = std::complex<double>(0.0, rgd_prefactor * F);
      s2_values[i] = std::complex<double>(0.0, rgd_prefactor * F * std::cos(theta));
    }

    S1_table.initialize_uniform(0.0, dtheta, s1_values);
    S2_table.initialize_uniform(0.0, dtheta, s2_values);
  }

  /// Exponential distribution: l = -l_mean · ln(U), U ~ Uniform(0, 1).
  double RGDMedium::sample_free_path(Rng &rng) const
  {
    return -1 * mean_free_path * std::log(rng.uniform());
  }

  /**
   * @brief RGD amplitude scattering matrix.
   *
   * Computes the 2×2 amplitude scattering matrix using the Rayleigh-Gans-Debye
   * approximation:
   *   - s2 = -i k³ (m-1) V F(θ,k,r) cos(θ) / (2π)   — parallel (p) component
   *   - s1 = -i k³ (m-1) V F(θ,k,r)         / (2π)   — perpendicular (s) component
   *
   * where V = (4π/3) r³ is the sphere volume and F is the form factor.
   */
  void RGDMedium::scattering_matrix(const double theta, const double phi, CMatrix &out) const
  {
    // O(1) interpolated look-up from the uniform θ tables built in the constructor.
    const std::complex<double> s1 = S1_table.Sample(theta);
    const std::complex<double> s2 = S2_table.Sample(theta);

    out(0, 0) = s2;
    out(0, 1) = std::complex<double>(0, 0);
    out(1, 0) = std::complex<double>(0, 0);
    out(1, 1) = s1;
  }

  void RGDMedium::set_mean_free_path(double mfp)
  {
    mean_free_path = mfp;
  }


// ══════════════════════════════════════════════════════════════════════════════
//  MieMedium — Mie theory via MIEV0 with precomputed S1/S2 tables
// ══════════════════════════════════════════════════════════════════════════════

  MieMedium::MieMedium(PhaseFunction *phase_func, double r, double n_particle, double n_medium, double wavelength)
      : ScatteringMedium(phase_func)
  {
    this->radius = r;
    this->n_particle = n_particle;
    this->n_medium = n_medium;
    this->m = std::complex<double>(n_particle/n_medium, 0); // purely real relative index
    this->wavelength = wavelength;
    this->k = 2 * M_PI * n_medium / wavelength; // Compute wave number from wavelength

    // Precompute S1/S2 tables at x = 2π r / λ with 1000 angle samples.
    precompute_scattering_tables(wavelength, 2 * M_PI * radius / wavelength, 1000);
  }

  /// Exponential distribution: l = -l_mean · ln(U), U ~ Uniform(0, 1).
  double MieMedium::sample_free_path(Rng &rng) const
  {
    return -1 * mean_free_path * std::log(rng.uniform());
  }

  /// Interpolates S1(θ) and S2(θ) from the precomputed DataTables.
  void MieMedium::scattering_matrix(const double theta, const double phi, CMatrix &out) const
  {
    const std::complex<double> s1 = S1_table.Sample(theta);
    const std::complex<double> s2 = S2_table.Sample(theta);

    out(0, 0) = s2;
    out(0, 1) = std::complex<double>(0, 0);
    out(1, 0) = std::complex<double>(0, 0);
    out(1, 1) = s1;
  }

  /**
   * @brief Populate S1_table and S2_table using the MIEV0 Mie solver.
   *
   * Allocates temporary arrays for the cosine of angle (`mulist`) and the
   * complex amplitude values (`s1tab`, `s2tab`), calls the MIEV0 Fortran
   * solver via the `miev()` C wrapper, then converts the cos-θ grid back to
   * a θ grid before initialising the DataTable objects.
   *
   * Angles are distributed uniformly in θ ∈ [0, π], so μ = cos(θ) is sampled
   * non-uniformly.  `S1_table` and `S2_table` store values as functions of θ.
   */
  void MieMedium::precompute_scattering_tables(double wavelength, double size_parameter, std::size_t n_samples)
  {
    std::vector<std::complex<double>> s1_values(n_samples);
    std::vector<std::complex<double>> s2_values(n_samples);

    std::complex<double>* s1tab = new std::complex<double>[n_samples];
    std::complex<double>* s2tab = new std::complex<double>[n_samples];
    double *mulist = new double[n_samples];

    // Build a uniform θ grid in [0, π]; convert to μ = cos(θ) for MIEV0.
    const double dtheta = M_PI / static_cast<double>(n_samples - 1);
    for (std::size_t i = 0; i < n_samples; ++i)
    {
      const double theta = static_cast<double>(i) * dtheta;
      mulist[i] = std::cos(theta);
    }

    miev(size_parameter, m, n_samples, mulist, s1tab, s2tab);

    // The θ grid is exactly uniform by construction, so store the tables as a
    // uniform grid to enable O(1) direct-index look-ups (no acos round-trip).
    for (std::size_t i = 0; i < n_samples; ++i)
    {
      s1_values[i] = s1tab[i];
      s2_values[i] = s2tab[i];
    }

    S1_table.initialize_uniform(0.0, dtheta, s1_values);
    S2_table.initialize_uniform(0.0, dtheta, s2_values);

    delete[] s1tab;
    delete[] s2tab;
    delete[] mulist;
  }

  void MieMedium::set_mean_free_path(double mfp)
  {
    mean_free_path = mfp;
  }

} // namespace luminis::core