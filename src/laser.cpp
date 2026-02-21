#include <luminis/log/logger.hpp>
#include <cmath>
#include <luminis/core/laser.hpp>
#include <luminis/math/rng.hpp>
#include <luminis/math/vec.hpp>

namespace luminis::core {

const double GEOMETRY_EPSILON = 1e-9;

math::Vec3 uniform_distribution(Rng &rng, const math::Vec3 &center, const double sigma) {
  const double theta = rng.uniform() * 2.0 * M_PI;
  const double r = std::sqrt(rng.uniform()) * sigma;
  return {
    center.x + r * std::cos(theta),
    center.y + r * std::sin(theta),
    center.z
  };
}

math::Vec3 gaussian_distribution(Rng &rng, const math::Vec3 &center, const double sigma) {
  return {
    rng.normal(center.x, sigma),
    rng.normal(center.y, sigma),
    center.z
  };
}

Laser::Laser(std::complex<double> m_state, std::complex<double> n_state, double wavelength, double sigma, LaserSource source_type)
{
  this->position = {0, 0, GEOMETRY_EPSILON};
  this->polarization = CVec2(m_state, n_state);
  this->wavelength = wavelength;
  this->sigma = sigma;
  this->source_type = source_type;

  direction = {0, 0, 1};
  local_m = {1, 0, 0};
  local_n = {0, 1, 0};
}

double Laser::sample_emission_time(Rng &rng) const {
  constexpr double PS_TO_NS      = 1e-3;
  constexpr double FWHM_TO_SIGMA = 1.0 / (2.0 * std::sqrt(2.0 * std::log(2.0)));

  const double t0_ns = time_offset * PS_TO_NS;

  switch (temporal_profile) {
  case TemporalProfile::Delta:
    return t0_ns;

  case TemporalProfile::Gaussian: {
    const double sigma_ns = pulse_duration * FWHM_TO_SIGMA * PS_TO_NS;
    return rng.normal(t0_ns, sigma_ns);
  }

  case TemporalProfile::TopHat:
    return t0_ns + (rng.uniform() - 0.5) * pulse_duration * PS_TO_NS;

  case TemporalProfile::Exponential:
    return t0_ns - std::log(rng.uniform()) * pulse_duration * PS_TO_NS;

  case TemporalProfile::PulseTrain:
    if (repetition_rate <= 0.0)
      return t0_ns;
    return t0_ns + rng.uniform() * (1e9 / repetition_rate);

  case TemporalProfile::CW:
    if (pulse_duration <= 0.0)
      return t0_ns;
    return t0_ns + rng.uniform() * pulse_duration * PS_TO_NS;
  }

  return t0_ns;
}

Photon Laser::emit_photon(Rng &rng) const {
  Vec3 pos;

  switch (source_type) {
  case LaserSource::Point:
    pos = {
      position.x,
      position.y,
      position.z
    };
    break;
  case LaserSource::Uniform:
    pos = uniform_distribution(rng, position, sigma);
    break;
  case LaserSource::Gaussian:
    pos = gaussian_distribution(rng, position, sigma);
    break;
  }
  Photon photon = Photon(pos, direction, local_m, local_n, wavelength);
  photon.set_polarization(polarization);
  photon.launch_time = sample_emission_time(rng);
  return photon;
}

} // namespace luminis::core
