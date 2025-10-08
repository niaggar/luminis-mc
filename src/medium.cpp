#include <cmath>
#include <complex>
#include <luminis/core/medium.hpp>
#include <luminis/log/logger.hpp>

namespace luminis::core {

Medium::Medium(double absorption, double scattering, PhaseFunction *phase_func) {
  mu_absorption = absorption;
  mu_scattering = scattering;
  mu_attenuation = mu_absorption + mu_scattering;
  phase_function = phase_func;
}
double Medium::sample_azimuthal_angle(Rng &rng) const {
  if (phase_function) {
    return phase_function->sample_phi(rng.uniform());
  }
  LLOG_ERROR("SimpleMedium::sample_azimuthal_angle: Phase function is not defined!");
  std::exit(EXIT_FAILURE);
}
double Medium::sample_conditional_azimuthal_angle(Rng &rng, CVec2& S, CVec2& E, double k, double theta) const {
  if (phase_function) {
    return phase_function->sample_phi_conditional(theta, S, E, k, rng);
  }
  LLOG_ERROR("SimpleMedium::sample_conditional_azimuthal_angle: Phase function is not defined!");
  std::exit(EXIT_FAILURE);
}
double Medium::light_speed_in_medium() const {
  return light_speed / refractive_index;
}



SimpleMedium::SimpleMedium(double absorption, double scattering, PhaseFunction *phase_func, double mfp, double r) : Medium(absorption, scattering, phase_func) {
  if (mfp <= 0 || std::abs(mfp - 1.0 / mu_attenuation) > 1e-6) {
    LLOG_ERROR("SimpleMedium::SimpleMedium: Invalid mean free path value!");
    std::exit(EXIT_FAILURE);
  }
  mean_free_path = mfp;
  radius = r;
}
double SimpleMedium::sample_free_path(Rng &rng) const {
  return -1 * mean_free_path * std::log(rng.uniform());
}
double SimpleMedium::sample_scattering_angle(Rng &rng) const {
  if (phase_function) {
    return phase_function->sample_theta(rng.uniform());
  }
  LLOG_ERROR("SimpleMedium::sample_scattering_angle: Phase function is not defined!");
  std::exit(EXIT_FAILURE);
}
CVec2 SimpleMedium::scattering_matrix(const double theta, const double phi, const double k) const {
  const double F = form_factor(theta, k, radius);
  const double kkk = std::pow(k, 3);

  const std::complex<double> s2 = std::complex<double>(0, -1 * kkk * F * std::cos(theta));
  const std::complex<double> s1 = std::complex<double>(0, -1 * kkk * F);

  return {s1, s2};
}

} // namespace luminis::core
