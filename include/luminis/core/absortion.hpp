#pragma once
#include <cstddef>
#include <luminis/core/photon.hpp>
#include <luminis/math/vec.hpp>
#include <vector>

using namespace luminis::math;

namespace luminis::core {


struct Absorption {
  double radius;
  double depth;
  double d_r;
  double d_z;
  Matrix absorption_values;

  Absorption(double r, double z, double dr, double dz);
  void record_absorption(const Photon &photon, double d_weight);
  Matrix get_absorption_image(const int n_photons) const;
};

struct AbsorptionTimeDependent {
  double radius;
  double depth;
  double d_r;
  double d_z;
  double d_t;
  int n_t_slices;
  double t_max;
  std::vector<Absorption> time_slices;

  AbsorptionTimeDependent(double r, double z, double dr, double dz, double dt, double t_max);

  AbsorptionTimeDependent clone() const;
  void merge_from(const AbsorptionTimeDependent &other);

  void record_absorption(const Photon &photon, double d_weight);
  Matrix get_absorption_image(const int n_photons, const int time_index) const;
};

AbsorptionTimeDependent* combine_absorptions(const std::vector<AbsorptionTimeDependent> &absorptions);

} // namespace luminis::core
