/**
 * @file medium.cpp
 * @brief Implementations of Medium, SimpleMedium, and MieMedium.
 *
 * Provides the base-class method bodies (free-path sampling delegation,
 * boundary test, angle sampling via phase function) and the two concrete
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
//  Medium — base class
// ══════════════════════════════════════════════════════════════════════════════

  /// Initialises μ_a, μ_s, μ_t = μ_a + μ_s, and the phase-function pointer.
  Medium::Medium(double absorption, double scattering, PhaseFunction *phase_func)
      : mu_absorption(absorption), mu_scattering(scattering), mu_attenuation(absorption + scattering), phase_function(phase_func) {}

  double Medium::sample_azimuthal_angle(Rng &rng) const
  {
    if (phase_function)
    {
      return phase_function->sample_phi(rng.uniform());
    }
    LLOG_ERROR("Medium::sample_azimuthal_angle: Phase function is not defined!");
    std::exit(EXIT_FAILURE);
  }

  double Medium::sample_conditional_azimuthal_angle(Rng &rng, CMatrix &S, CVec2 &E, double k, double theta) const
  {
    if (phase_function)
    {
      return phase_function->sample_phi_conditional(theta, S, E, k, rng);
    }
    LLOG_ERROR("Medium::sample_conditional_azimuthal_angle: Phase function is not defined!");
    std::exit(EXIT_FAILURE);
  }

  double Medium::light_speed_in_medium() const
  {
    return light_speed;
  }

  /// Returns false if z < 0, or if any coordinate is NaN or infinite.
  bool Medium::is_inside(const Vec3 &position) const
  {
    bool inside = true;
    if (position.z < 0)
    {
      inside = false;
    }
    else if (std::isnan(position.x) || std::isnan(position.y) || std::isnan(position.z))
    {
      inside = false;
    }
    else if (std::isinf(position.x) || std::isinf(position.y) || std::isinf(position.z))
    {
      inside = false;
    }
    return inside;
  }

  double Medium::sample_scattering_angle(Rng &rng) const
  {
    if (phase_function)
    {
      return phase_function->sample_theta(rng.uniform());
    }
    LLOG_ERROR("Medium::sample_scattering_angle: Phase function is not defined!");
    std::exit(EXIT_FAILURE);
  }

// ══════════════════════════════════════════════════════════════════════════════
//  SimpleMedium — Rayleigh-Gans-Debye (RGD) approximation
// ══════════════════════════════════════════════════════════════════════════════

  SimpleMedium::SimpleMedium(double absorption, double scattering, PhaseFunction *phase_func, double mfp, double r, double n_particle, double n_medium)
      : Medium(absorption, scattering, phase_func)
  {
    mean_free_path = mfp;
    radius = r;
    this->n_particle = n_particle;
    this->n_medium = n_medium;
  }

  /// Exponential distribution: l = -l_mean · ln(U), U ~ Uniform(0, 1).
  double SimpleMedium::sample_free_path(Rng &rng) const
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
  CMatrix SimpleMedium::scattering_matrix(const double theta, const double phi, const double k) const
  {
    const double F = form_factor(theta, k, radius);
    const double kkk = std::pow(k, 3);
    const double volume = 4 * M_PI * std::pow(radius, 3) / 3.0;
    const double relative_refractive_index = n_particle / n_medium;
    const double mm = relative_refractive_index - 1.0; // contrast index (m - 1)

    const std::complex<double> s2 = std::complex<double>(0, -1 * kkk * mm * volume * F * std::cos(theta) / (2 * M_PI));
    const std::complex<double> s1 = std::complex<double>(0, -1 * kkk * mm * volume * F / (2 * M_PI));

    CMatrix res(2, 2);
    res(0, 0) = s2;
    res(0, 1) = std::complex<double>(0, 0);
    res(1, 0) = std::complex<double>(0, 0);
    res(1, 1) = s1;
    return res;
  }

// ══════════════════════════════════════════════════════════════════════════════
//  MieMedium — Mie theory via MIEV0 with precomputed S1/S2 tables
// ══════════════════════════════════════════════════════════════════════════════

  MieMedium::MieMedium(double absorption, double scattering, PhaseFunction *phase_func, double mfp, double r, double n_particle, double n_medium, double wavelength)
      : Medium(absorption, scattering, phase_func)
  {
    mean_free_path = mfp;
    radius = r;
    this->n_particle = n_particle;
    this->n_medium = n_medium;
    this->m = std::complex<double>(n_particle/n_medium, 0); // purely real relative index
    this->wavelength = wavelength;

    // Precompute S1/S2 tables at x = 2π r / λ with 1000 angle samples.
    precompute_scattering_tables(wavelength, 2 * M_PI * radius / wavelength, 1000);
  }

  /// Exponential distribution: l = -l_mean · ln(U), U ~ Uniform(0, 1).
  double MieMedium::sample_free_path(Rng &rng) const
  {
    return -1 * mean_free_path * std::log(rng.uniform());
  }

  /// Interpolates S1(θ) and S2(θ) from the precomputed DataTables.
  CMatrix MieMedium::scattering_matrix(const double theta, const double phi, const double k) const
  {
    std::complex<double> s1;
    std::complex<double> s2;

    s1 = S1_table.Sample(theta);
    s2 = S2_table.Sample(theta);

    CMatrix res(2, 2);
    res(0, 0) = s2;
    res(0, 1) = std::complex<double>(0, 0);
    res(1, 0) = std::complex<double>(0, 0);
    res(1, 1) = s1;
    return res;
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
    std::vector<double> theta_values(n_samples);
    std::vector<std::complex<double>> s1_values(n_samples);
    std::vector<std::complex<double>> s2_values(n_samples);

    std::complex<double>* s1tab = new std::complex<double>[n_samples];
    std::complex<double>* s2tab = new std::complex<double>[n_samples];
    double *mulist = new double[n_samples];

    // Build a uniform θ grid; convert to μ = cos(θ) for MIEV0.
    for (std::size_t i = 0; i < n_samples; ++i)
    {
      double theta = (static_cast<double>(i) / (n_samples - 1)) * M_PI;
      mulist[i] = std::cos(theta);
    }

    miev(size_parameter, m, n_samples, mulist, s1tab, s2tab);

    // Convert μ back to θ and copy into STL vectors for DataTable initialisation.
    for (std::size_t i = 0; i < n_samples; ++i)
    {
      theta_values[i] = std::acos(mulist[i]);
      s1_values[i] = s1tab[i];
      s2_values[i] = s2tab[i];
    }

    S1_table.initialize(theta_values, s1_values);
    S2_table.initialize(theta_values, s2_values);
  }

} // namespace luminis::core