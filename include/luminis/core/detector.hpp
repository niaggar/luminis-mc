#pragma once
#include <cstddef>
#include <luminis/core/photon.hpp>
#include <luminis/math/vec.hpp>
#include <vector>

using namespace luminis::math;

namespace luminis::core {

/// @brief Angular intensity distribution in spherical coordinates
struct AngularIntensity {
  Matrix Ix, Iy, Iz, I_total, Ico, Icros; ///< Intensity components and total
  int N_theta, N_phi;          ///< Number of bins
  double theta_max, phi_max;   ///< Maximum angles (radians)
  double dtheta, dphi;         ///< Angular resolution

  AngularIntensity(int n_theta, int n_phi, double theta_max, double phi_max)
      : N_theta(n_theta), N_phi(n_phi), theta_max(theta_max), phi_max(phi_max),
        dtheta(theta_max / n_theta), dphi(phi_max / n_phi),
        Ix(n_theta, n_phi), Iy(n_theta, n_phi), Iz(n_theta, n_phi), I_total(n_theta, n_phi), Ico(n_theta, n_phi), Icros(n_theta, n_phi) {}
};

/// @brief Spatial intensity distribution in cartesian coordinates
struct SpatialIntensity {
  Matrix Ix, Iy, Iz, I_total, Ico, Icros; ///< Intensity components and total
  int N_x, N_y;                ///< Number of bins
  double x_len, y_len;         ///< Physical dimensions
  double dx, dy;               ///< Spatial resolution

  SpatialIntensity(int n_x, int n_y, double x_len, double y_len)
      : N_x(n_x), N_y(n_y), x_len(x_len), y_len(y_len),
        dx(x_len / n_x), dy(y_len / n_y),
        Ix(n_x, n_y), Iy(n_x, n_y), Iz(n_x, n_y), I_total(n_x, n_y), Ico(n_x, n_y), Icros(n_x, n_y) {}
};

/// @brief Photon detector plane for recording scattered light
struct Detector {
  Vec3 origin;                                  ///< Detector position
  Vec3 normal;                                  ///< Surface normal (forward)
  Vec3 backward_normal;                         ///< Surface normal (backward)
  Vec3 n_polarization;                          ///< Polarization basis vector n
  Vec3 m_polarization;                          ///< Polarization basis vector m
  std::vector<PhotonRecord> recorded_photons{}; ///< Recorded photon data
  std::size_t hits{0};                          ///< Total photon hits

  /// @brief Construct detector at z-position
  /// @param z Detector z-coordinate
  Detector(double z);

  /// @brief Record photon intersection with detector plane
  /// @param photon Photon to validate and record
  void record_hit(Photon &photon);

  /// @brief Create empty detector copy for parallel processing
  Detector copy_start() const;

  /// @brief Merge results from another detector
  /// @param other Detector to merge from
  void merge_from(const Detector &other);

  /// @brief Compute histogram of scattering events
  /// @param min_theta Minimum angle (rads)
  /// @param max_theta Maximum angle (rads)
  /// @return Histogram bins
  std::vector<double> compute_events_histogram(const double min_theta, const double max_theta) const;

  /// @brief Compute histogram of theta angles of detection
  /// @param min_theta Minimum angle (rads)
  /// @param max_theta Maximum angle (rads)
  /// @return Histogram bins
  std::vector<double> compute_theta_histogram(const double min_theta, const double max_theta, const int n_bins) const;

  /// @brief Compute histogram of phi angles of detection
  /// @param min_phi Minimum angle (rads)
  /// @param max_phi Maximum angle (rads)
  /// @return Histogram bins
  std::vector<double> compute_phi_histogram(const double min_phi, const double max_phi, const int n_bins) const;

  /// @brief Compute speckle pattern from interference
  /// @param n_theta Angular bins (theta)
  /// @param n_phi Angular bins (phi)
  /// @return Angular intensity pattern
  AngularIntensity compute_speckle(const int n_theta=1125, const int n_phi=360) const;

  /// @brief Compute spatial intensity distribution centered at origin=(0,0,z)
  /// @param x_len Length in x direction
  /// @param y_len Length in y direction
  /// @param max_theta Maximum collection angle (rads)
  /// @param n_x Number of bins in x direction
  /// @param n_y Number of bins in y direction
  /// @return Spatial intensity pattern
  SpatialIntensity compute_spatial_intensity(const double x_len, const double y_len, const double max_theta, const int n_x=1125, const int n_y=1125) const;

  /// @brief Compute angular intensity distribution
  AngularIntensity compute_angular_intensity(const double max_theta, const double max_phi, const int n_theta=1125, const int n_phi=360) const;

  /// @brief Compute time-resolved spatial intensity
  std::vector<SpatialIntensity> compute_time_resolved_spatial_intensity(const double x_len, const double y_len, const double max_theta, const double t_max, const double dt, const int n_x=1125, const int n_y=1125) const;

  /// @brief Save photon records to binary file
  /// @param filename Output file path
  void save_recorded_photons(const std::string &filename) const;

  /// @brief Load photon records from binary file
  /// @param filename Input file path
  void load_recorded_photons(const std::string &filename);
};

} // namespace luminis::core
