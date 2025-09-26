#include <cmath>
#include <luminis/core/medium.hpp>
#include <luminis/log/logger.hpp>
#include <complex>

namespace luminis::core
{

  double form_factor(const double theta, const double k, const double radius)
  {
    const double ks = 2.0 * k * std::sin(theta / 2.0) * radius;
    const double numerator = 3 * (std::sin(ks) - ks * std::cos(ks));
    const double denominator = std::pow(ks, 3);
    
    return numerator / denominator;
  }

  Medium::Medium(double absorption, double scattering, PhaseFunction *phase_func)
      : mu_a(absorption), mu_s(scattering), phase_function(phase_func) {}







  SimpleMedium::SimpleMedium(double absorption, double scattering, PhaseFunction *phase_func, double mfp, double r)
      : Medium(absorption, scattering, phase_func), mean_free_path(mfp), radius(r) {}

  double SimpleMedium::sample_free_path(Rng &rng) const
  {
    return -1 * mean_free_path * std::log(rng.uniform());
  }

  double SimpleMedium::sample_scattering_angle(Rng &rng) const
  {
    if (phase_function)
    {
      return phase_function->Sample(rng.uniform());
    }

    LLOG_ERROR("SimpleMedium::sample_scattering_angle: Phase function is not defined!");
    std::exit(EXIT_FAILURE);
  }

  double SimpleMedium::sample_azimuthal_angle(Rng &rng) const
  {
    return rng.uniform() * 2.0 * M_PI;
  }

  CVec2 SimpleMedium::scattering_matrix(const double theta, const double phi, const double k) const
  {
    const double F = form_factor(theta, k, radius);
    const double kkk = std::pow(k, 3);
    
    const std::complex<double> s2 = std::complex<double>(0, -1*kkk*F*std::cos(theta));
    const std::complex<double> s1 = std::complex<double>(0, -1*kkk*F);

    return {s1, s2};
  }

}
