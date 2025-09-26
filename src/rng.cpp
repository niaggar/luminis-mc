#include <luminis/math/rng.hpp>

namespace luminis::math {

double Rng::uniform() {
  return uni(gen);
}

double Rng::normal(const double mean, const double stddev) {
  const double u1 = uniform();
  const double u2 = uniform();
  const double z0 = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
  return z0 * stddev + mean;
}

}
