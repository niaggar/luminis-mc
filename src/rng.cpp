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

double Rng::exp_path(const double mfp) {
  const double u = std::max(1e-16, uniform());
  return -mfp * std::log(u);
}

std::pair<double, double> Rng::isotropic_angles() {
  const double u = 2.0 * uniform() - 1.0;                   // cos(theta)
  const double phi = 2.0 * M_PI * uniform();                // [0, 2π)
  const double theta = std::acos(std::clamp(u, -1.0, 1.0)); // theta in [0, π]
  return {theta, phi};
}

}
