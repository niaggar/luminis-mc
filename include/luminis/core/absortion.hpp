#pragma once
#include <cstddef>
#include <luminis/core/photon.hpp>
#include <luminis/math/vec.hpp>
#include <vector>

using namespace luminis::math;

namespace luminis::core {

struct Absorption {
  double radius{0.0};
  double depth{0.0};
  double d_r{0.0};
  double d_z{0.0};
  std::vector<std::vector<double>> absorption_values;

  Absorption(double r, double z, double dr, double dz);
  void record_absorption(const Photon &photon, double d_weight);
  std::vector<std::vector<double>> get_absorption_image(const int n_photons) const;
};

} // namespace luminis::core
