#include "luminis/log/logger.hpp"
#include <cmath>
#include <luminis/core/laser.hpp>
#include <luminis/math/rng.hpp>
#include <luminis/math/vec.hpp>

namespace luminis::core {

math::Vec3 uniform_distribution(Rng &rng, const math::Vec3 &center, const double sigma) {
  const double theta = rng.uniform() * 2.0 * M_PI;
  const double r = std::sqrt(rng.uniform()) * sigma;
  return {
    center[0] + r * std::cos(theta),
    center[1] + r * std::sin(theta),
    center[2]
  };
}

math::Vec3 gaussian_distribution(Rng &rng, const math::Vec3 &center, const double sigma) {
  return {rng.normal(center[0], sigma), rng.normal(center[1], sigma), center[2]};
}

Laser::Laser(Vec3 position, Vec3 direction, Vec3 local_m, Vec3 local_n,
             CVec2 polarization, double wavelength, double sigma,
             LaserSource source_type)
    : position(normalize(position)), direction(normalize(direction)),
      local_m(normalize(local_m)), local_n(normalize(local_n)),
      polarization(polarization), wavelength(wavelength), sigma(sigma),
      source_type(source_type) {
        LLOG_DEBUG("Polarization1 real: {}, imag: {}", polarization[0].real(), polarization[0].imag());
        LLOG_DEBUG("Polarization2 real: {}, imag: {}", polarization[1].real(), polarization[1].imag());
      }

// TODO: Implement time sampling based on pulse duration and repetition rate
double Laser::sample_emission_time(Rng &rng) const {
  return 0.0;
}

Photon Laser::emit_photon(Rng &rng) const {
  Vec3 pos;

  switch (source_type) {
  case LaserSource::Point:
    pos = luminis::math::copy(position);
    break;
  case LaserSource::Uniform:
    pos = uniform_distribution(rng, position, sigma);
    break;
  case LaserSource::Gaussian:
    pos = gaussian_distribution(rng, position, sigma);
    break;
  }
  Photon photon = Photon(pos, direction, local_m, local_n, wavelength);
  photon.set_polarization(polarization[0], polarization[1]);
  return photon;
}

} // namespace luminis::core
