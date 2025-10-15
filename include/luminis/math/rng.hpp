#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>

namespace luminis::math {

std::uint64_t mix_seed(std::uint64_t base, std::uint64_t tid);

struct Rng {
  std::mt19937_64 gen;
  std::uniform_real_distribution<double> uni{0.0, 1.0};

  explicit Rng(uint64_t seed = std::random_device{}()) : gen(seed) {}

  double uniform();
  double normal(const double mean, const double stddev);
};

} // namespace luminis::math
