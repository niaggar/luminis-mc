#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>

namespace luminis::math {

struct Rng {
  std::mt19937_64 gen;
  std::uniform_real_distribution<double> uni{0.0, 1.0};

  explicit Rng(uint64_t seed = std::random_device{}()) : gen(seed) {}
  double uniform() { return uni(gen); }

  // Sample exponential free path with mean mfp
  double exp_path(double mfp) {
    double u = std::max(1e-16, uniform());
    return -mfp * std::log(u);
  }

  // Isotropic scattering: cos(theta) uniform in [-1,1], phi uniform in [0,2π)
  std::pair<double, double> isotropic_angles() {
    double u = 2.0 * uniform() - 1.0;                   // cos(theta)
    double phi = 2.0 * M_PI * uniform();                // [0, 2π)
    double theta = std::acos(std::clamp(u, -1.0, 1.0)); // theta in [0, π]
    return {theta, phi};
  }
};

} // namespace luminis::math
