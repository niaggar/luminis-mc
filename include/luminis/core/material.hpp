#pragma once
#include <luminis/math/rng.hpp>
#include <luminis/math/vec.hpp>

using namespace luminis::math;

namespace luminis::core {

// Minimal material: single mean free path, isotropic scattering
struct Material {
  double mean_free_path; // same units as your geometry

  explicit Material(double mfp) : mean_free_path(mfp) {}

  // Sample a free-path length
  double sample_free_path(Rng &rng) const {
    return rng.exp_path(mean_free_path);
  }

  // Isotropic scatter: returns new direction
  Vec3 scatter(Rng &rng) const {
    auto [theta, phi] = rng.isotropic_angles();
    return from_spherical(theta, phi);
  }
};

} // namespace luminis::core
