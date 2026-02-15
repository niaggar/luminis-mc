#pragma once
#include <cstddef>
#include <luminis/core/photon.hpp>
#include <luminis/math/vec.hpp>
#include <luminis/core/detector.hpp>
#include <vector>

using namespace luminis::math;

namespace luminis::core
{

  // /// @brief Compute histogram of scattering events
  // /// @param min_theta Minimum angle (rads)
  // /// @param max_theta Maximum angle (rads)
  // /// @return Histogram bins
  // std::vector<double> compute_events_histogram(
  //     Sensor &detector,
  //     const double min_theta,
  //     const double max_theta);

  // /// @brief Compute histogram of theta angles of detection
  // /// @param min_theta Minimum angle (rads)
  // /// @param max_theta Maximum angle (rads)
  // /// @return Histogram bins
  // std::vector<double> compute_theta_histogram(
  //     Sensor &detector,
  //     const double min_theta,
  //     const double max_theta,
  //     const int n_bins);

  // /// @brief Compute histogram of phi angles of detection
  // /// @param min_phi Minimum angle (rads)
  // /// @param max_phi Maximum angle (rads)
  // /// @return Histogram bins
  // std::vector<double> compute_phi_histogram(
  //     Sensor &detector,
  //     const double min_phi,
  //     const double max_phi,
  //     const int n_bins);

  // /// @brief Compute speckle pattern from interference
  // /// @param n_theta Angular bins (theta)
  // /// @param n_phi Angular bins (phi)
  // /// @return Angular intensity pattern
  // AngularIntensity compute_speckle(
  //     Sensor &detector,
  //     const int n_theta = 1125,
  //     const int n_phi = 360);

  // /// @brief Compute speckle pattern from interference
  // /// @return Angular intensity pattern
  // AngularIntensity compute_speckle_angledetector(
  //     AngleDetector &detector);

  // /// @brief Compute spatial intensity as a grid distribution centered at origin=(0,0,z)
  // /// @param x_len Length in x direction
  // /// @param y_len Length in y direction
  // /// @param max_theta Maximum collection angle (rads)
  // /// @param n_x Number of bins in x direction
  // /// @param n_y Number of bins in y direction
  // /// @return Spatial intensity pattern
  // SpatialIntensity compute_spatial_intensity(
  //     Sensor &detector,
  //     const double x_len,
  //     const double y_len,
  //     const double max_theta,
  //     const int n_x = 1125,
  //     const int n_y = 1125);

  // /// @brief Compute angular intensity distribution
  // AngularIntensity compute_angular_intensity(
  //     Sensor &detector,
  //     const double max_theta,
  //     const double max_phi,
  //     const int n_theta = 1125,
  //     const int n_phi = 360);

  // /// @brief Compute time-resolved spatial intensity
  // std::vector<SpatialIntensity> compute_time_resolved_spatial_intensity(
  //     Sensor &detector,
  //     const double x_len,
  //     const double y_len,
  //     const double max_theta,
  //     const double t_max,
  //     const double dt,
  //     const int n_x = 1125,
  //     const int n_y = 1125);

  // /// @brief Save photon records to binary file
  // /// @param filename Output file path
  // void save_recorded_photons(const std::string &filename, const Sensor &detector);

  // /// @brief Load photon records from binary file
  // /// @param filename Input file path
  // void load_recorded_photons(const std::string &filename, Sensor &detector);


  // void save_angle_detector_fields(const std::string &filename, const AngleDetector &detector);

  // void load_angle_detector_fields(const std::string &filename, AngleDetector &detector);

} // namespace luminis::core
