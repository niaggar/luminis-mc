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
  CMatrix RGDMedium::scattering_matrix(const double theta, const double phi) const
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

  double RGDMedium::scattering_efficiency() const
  {
    const double x = radius * k;
    const double V = 4 * M_PI * std::pow(radius, 3) / 3.0;
    const double a = (std::pow(k, 6) * std::pow((n_particle / n_medium - 1.0), 2) * V * V) / (4 * std::pow(M_PI, 2));
    const double c = a / std::pow(x, 2);

    // Numerical integration of F^2(theta) * sin(theta) * (1 + cos^2(theta)) over [0, pi]
    // using Simpson's rule with N subintervals (N must be even).
    const int N = 100000;
    const double h = M_PI / N;

    auto integrand = [&](double theta) -> double {
      const double F = form_factor(theta, k, radius);
      const double cos_t = std::cos(theta);
      return F * F * std::sin(theta) * (1.0 + cos_t * cos_t);
    };

    double sum = integrand(0.0) + integrand(M_PI);
    for (int i = 1; i < N; i += 2)
      sum += 4.0 * integrand(i * h);
    for (int i = 2; i < N; i += 2)
      sum += 2.0 * integrand(i * h);

    const double integral = sum * h / 3.0;

    return c * integral;
  }

  double RGDMedium::scattering_cross_section() const
  {
    const double Q_sca = scattering_efficiency();
    const double geometric_cross_section = M_PI * std::pow(radius, 2);
    return Q_sca * geometric_cross_section;
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
  CMatrix MieMedium::scattering_matrix(const double theta, const double phi) const
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

  double MieMedium::scattering_efficiency() const
  {
    const double x = radius * k;
    double qext, qsca, g;
		mievinfo(x, m, &qext, &qsca, &g);

    // Log all the values
    LLOG_INFO("MieMedium::scattering_efficiency: x = {}, Q_ext = {}, Q_sca = {}, g = {}", x, qext, qsca, g);

    return qsca;
  }

  double MieMedium::scattering_cross_section() const
  {
    const double Q_sca = scattering_efficiency();
    const double geometric_cross_section = M_PI * std::pow(radius, 2);
    return Q_sca * geometric_cross_section;
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

  void MieMedium::set_mean_free_path(double mfp)
  {
    mean_free_path = mfp;
  }

} // namespace luminis::core