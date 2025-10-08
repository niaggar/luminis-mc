#pragma once
#include <cstddef>
#include <luminis/core/photon.hpp>
#include <luminis/math/vec.hpp>
#include <vector>

using namespace luminis::math;

namespace luminis::core {

struct AngularIntensity {
  std::vector<std::vector<double>> Ix, Iy, I;
  int N_theta{360}, N_phi{360};
  double theta_max{0.5*M_PI}, phi_max{2.0*M_PI};
};

struct SpatialIntensity {
  std::vector<std::vector<double>> Ix, Iy, I;
  int N_x{1125}, N_y{1125};
  double x_max{10.0}, y_max{10.0};
  double dx{2.0 * x_max / N_x}, dy{2.0 * y_max / N_y};

  SpatialIntensity(int nx, int ny, double xmax, double ymax)
    : N_x(nx), N_y(ny), x_max(xmax), y_max(ymax) {
      Ix.resize(N_x, std::vector<double>(N_y, 0.0));
      Iy.resize(N_x, std::vector<double>(N_y, 0.0));
      I.resize(N_x, std::vector<double>(N_y, 0.0));
      dx = 2.0 * x_max / N_x;
      dy = 2.0 * y_max / N_y;
    }
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
  AngularIntensity compute_speckle(const int n_theta=1125, const int n_phi=360);
  SpatialIntensity compute_spatial_intensity(const double max_theta, const int n_x=1125, const int n_y=1125, const double x_max=10.0, const double y_max=10.0);
  AngularIntensity compute_angular_intensity(const double max_theta, const double max_phi, const int n_theta=1125, const int n_phi=360);
};

} // namespace luminis::core
