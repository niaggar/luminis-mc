#include <cmath>
#include <luminis/core/medium.hpp>
#include <luminis/log/logger.hpp>

namespace luminis::core
{

  Medium::Medium(double absorption, double scattering, PhaseFunction *phase_func)
      : mu_a(absorption), mu_s(scattering), phase_function(phase_func) {}







  SimpleMedium::SimpleMedium(double absorption, double scattering, PhaseFunction *phase_func, double mfp)
      : Medium(absorption, scattering, phase_func), mean_free_path(mfp) {}

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

}
