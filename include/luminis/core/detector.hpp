#pragma once
#include <cstddef>
#include <luminis/core/photon.hpp>
#include <luminis/math/vec.hpp>
#include <vector>

using namespace luminis::math;

namespace luminis::core {

struct AngularSpeckle {
  std::vector<std::vector<double>> Ix, Iy, I;
  int N_theta{1125}, N_phi{360};
  double theta_max{0.5*M_PI}, phi_max{2.0*M_PI};
};

struct Detector {
  std::size_t hits{0};
  Vec3 origin{0, 0, 0};
  Vec3 normal{0, 0, 1};
  Vec3 n_polarization{1, 0, 0};
  Vec3 m_polarization{0, 1, 0};
  std::vector<Photon> recorded_photons;

  Detector() = default;
  Detector(const Vec3 o, const Vec3 normal, const Vec3 n, const Vec3 m);

  void record_hit(Photon &photon);

  std::vector<double> compute_events_histogram(const double min_theta, const double max_theta);
  AngularSpeckle compute_speckle_maps(const int n_theta=1125, const int n_phi=360);
};

} // namespace luminis::core
