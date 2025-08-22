#pragma once
#include "luminis/math/rng.hpp"
#include <luminis/math/vec.hpp>
#include <luminis/core/photon.hpp>

namespace luminis::core {

enum class LaserSource {
  Point = 0,
  Uniform = 1,
  Gaussian = 2,
};

struct Laser {
public:
  Laser(const luminis::math::Vec3 &position, const luminis::math::Vec3 &direction,
        const luminis::math::Vec2 &polarization, double wavelength, double sigma,
        LaserSource source_type, Rng rng);

  Photon emitPhoton() const;

private:
  luminis::math::Vec3 position;
  luminis::math::Vec3 direction;
  luminis::math::Vec2 polarization;
  double wavelength;
  double sigma;
  LaserSource source_type;
  Rng rng;
};

Vec3 uniform_distribution(Rng rng, const Vec3 center, const double sigma);
Vec3 gaussian_distribution(Rng rng, const Vec3 center, const double sigma);

}
