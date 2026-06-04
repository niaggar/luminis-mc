#include <luminis/math/rng.hpp>

namespace luminis::math {

std::uint64_t mix_seed(std::uint64_t base, std::uint64_t tid) {
  std::uint64_t z = base + 0x9E3779B97F4A7C15ULL * (tid + 1);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

double Rng::uniform() {
  // Top 53 bits → double in [0, 1).  Matches the standard xoshiro recipe and is
  // far cheaper than std::uniform_real_distribution.
  return (next_u64() >> 11) * 0x1.0p-53;
}

double Rng::normal(const double mean, const double stddev) {
  // uniform() draws from [0, 1); guard against u1 == 0, which would make
  // std::log(u1) == -inf and produce a NaN/Inf sample.
  double u1 = uniform();
  while (u1 <= 0.0)
    u1 = uniform();
  const double u2 = uniform();
  const double z0 = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
  return z0 * stddev + mean;
}

}
