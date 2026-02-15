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

Laser::Laser(Vec3 position, CVec2 polarization, double wavelength, double sigma, LaserSource source_type)
{
  this->position = {position.x, position.y, position.z + GEOMETRY_EPSILON};
  this->polarization = polarization;
  this->wavelength = wavelength;
  this->sigma = sigma;
  this->source_type = source_type;

  direction = {0, 0, 1};
  local_m = {1, 0, 0};
  local_n = {0, 1, 0};
}

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
