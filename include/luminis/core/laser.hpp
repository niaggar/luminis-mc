#pragma once
#include <luminis/core/photon.hpp>
#include <luminis/math/rng.hpp>
#include <luminis/math/vec.hpp>

using namespace luminis::math;

namespace luminis::core {

enum class LaserSource {
  Point = 0,
  Uniform = 1,
  Gaussian = 2,
};

struct Laser {
  Vec3 position;
  Vec3 direction;
  Vec3 local_m;
  Vec3 local_n;
  Vec2 polarization;

  double wavelength;
  double sigma;
  LaserSource source_type;

  Laser(Vec3 position, Vec3 direction, Vec3 local_m, Vec3 local_n,
        Vec2 polarization, double wavelength, double sigma,
        LaserSource source_type);

  Photon emit_photon(Rng &rng) const;
};

Vec3 uniform_distribution(Rng &rng, const Vec3 &center, const double sigma);
Vec3 gaussian_distribution(Rng &rng, const Vec3 &center, const double sigma);

} // namespace luminis::core
