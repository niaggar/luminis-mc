#include "luminis/math/vec.hpp"
#include <cmath>
#include <complex>
#include <luminis/core/medium.hpp>
#include <luminis/log/logger.hpp>
#include <luminis/sample/meanfreepath.hpp>
#include "luminis/mie/dmiev.h"

namespace luminis::core
{

  Medium::Medium(double absorption, double scattering, PhaseFunction *phase_func)
      : mu_absorption(absorption), mu_scattering(scattering), mu_attenuation(absorption + scattering), phase_function(phase_func) {}
  double Medium::sample_azimuthal_angle(Rng &rng) const
  {
    if (phase_function)
    {
      return phase_function->sample_phi(rng.uniform());
    }
    LLOG_ERROR("SimpleMedium::sample_azimuthal_angle: Phase function is not defined!");
    std::exit(EXIT_FAILURE);
  }
  double Medium::sample_conditional_azimuthal_angle(Rng &rng, CMatrix &S, CVec2 &E, double k, double theta) const
  {
    if (phase_function)
    {
      return phase_function->sample_phi_conditional(theta, S, E, k, rng);
    }
    LLOG_ERROR("SimpleMedium::sample_conditional_azimuthal_angle: Phase function is not defined!");
    std::exit(EXIT_FAILURE);
  }
  double Medium::light_speed_in_medium() const
  {
    return light_speed;
  }
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
    LLOG_ERROR("SimpleMedium::sample_scattering_angle: Phase function is not defined!");
    std::exit(EXIT_FAILURE);
  }

  SimpleMedium::SimpleMedium(double absorption, double scattering, PhaseFunction *phase_func, double mfp, double r, double n_particle, double n_medium)
      : Medium(absorption, scattering, phase_func)
  {
    mean_free_path = mfp;
    radius = r;
    this->n_particle = n_particle;
    this->n_medium = n_medium;
  }
  double SimpleMedium::sample_free_path(Rng &rng) const
  {
    return -1 * mean_free_path * std::log(rng.uniform());
  }
  CMatrix SimpleMedium::scattering_matrix(const double theta, const double phi, const double k) const
  {
    const double F = form_factor(theta, k, radius);
    const double kkk = std::pow(k, 3);
    const double volume = 4 * M_PI * std::pow(radius, 3) / 3.0;
    const double relative_refractive_index = n_particle / n_medium;
    const double mm = relative_refractive_index - 1.0;

    const std::complex<double> s2 = std::complex<double>(0, -1 * kkk * mm * volume * F * std::cos(theta) / (2 * M_PI));
    const std::complex<double> s1 = std::complex<double>(0, -1 * kkk * mm * volume * F / (2 * M_PI));

    CMatrix res(2, 2);
    res(0, 0) = s2;
    res(0, 1) = std::complex<double>(0, 0);
    res(1, 0) = std::complex<double>(0, 0);
    res(1, 1) = s1;
    return res;
  }


  MieMedium::MieMedium(double absorption, double scattering, PhaseFunction *phase_func, double mfp, double r, double n_particle, double n_medium)
      : Medium(absorption, scattering, phase_func)
  {
    mean_free_path = mfp;
    radius = r;
    this->n_particle = n_particle;
    this->n_medium = n_medium;
    this->m = std::complex<double>(n_particle/n_medium, 0);
  }
  double MieMedium::sample_free_path(Rng &rng) const
  {
    return -1 * mean_free_path * std::log(rng.uniform());
  }
  CMatrix MieMedium::scattering_matrix(const double theta, const double phi, const double k) const
  {
    std::complex<double> s1;
    std::complex<double> s2;
    double mu = std::cos(theta);
    if (mu > 1.0) mu = 1.0;
    if (mu < -1.0) mu = -1.0;
    std::complex<double> crefin = m;
    double sizep = k * this->radius;

    amiev(&sizep, &crefin, &mu, &s1, &s2);

    CMatrix res(2, 2);
    res(0, 0) = s2;
    res(0, 1) = std::complex<double>(0, 0);
    res(1, 0) = std::complex<double>(0, 0);
    res(1, 1) = s1;
    return res;
  }

} // namespace luminis::core
