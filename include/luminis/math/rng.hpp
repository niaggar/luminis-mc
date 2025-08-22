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

  double uniform();
  double normal(const double mean, const double stddev);
  double exp_path(const double mfp);
  std::pair<double, double> isotropic_angles();
};

} // namespace luminis::math
