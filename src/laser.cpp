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

Laser::Laser(Vec3 position, Vec3 direction, Vec3 local_m, Vec3 local_n,
             CVec2 polarization, double wavelength, double sigma,
             LaserSource source_type)
    : position(position), direction(direction),
      local_m(local_m), local_n(local_n),
      polarization(polarization), wavelength(wavelength), sigma(sigma),
      source_type(source_type) {}

// TODO: Implement time sampling based on pulse duration and repetition rate
double Laser::sample_emission_time(Rng &rng) const {
  return 0.0;
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
  return photon;
}

} // namespace luminis::core
