#pragma once
#include <cstddef>
#include <luminis/core/photon.hpp>
#include <luminis/math/vec.hpp>
#include <vector>
#include <functional>

using namespace luminis::math;

namespace luminis::core {

using DetectionCondition = std::function<bool(const Photon&)>;

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
  std::vector<DetectionCondition> conditions{}; ///< Detection conditions
  std::size_t hits{0};                          ///< Total photon hits
  
  /// @brief Construct detector at z-position
  /// @param z Detector z-coordinate
  Detector(double z);

  bool is_hit_by(const Photon &photon) const;

  /// @brief Record photon intersection with detector plane
  /// @param photon Photon to validate and record
  virtual void record_hit(Photon &photon);

  /// @brief Create empty detector copy for parallel processing
  virtual std::unique_ptr<Detector> clone() const;

  /// @brief Merge results from another detector
  /// @param other Detector to merge from
  virtual void merge_from(const Detector &other);

  void add_detection_condition(const DetectionCondition &condition) {
    conditions.push_back(condition);
  }

  bool validate_detection_conditions(const Photon &photon) const {
    for (const auto &condition : conditions) {
      if (!condition(photon)) {
        return false;
      }
    }
    return true;
  }
};

struct AngleDetector : public Detector {
  int N_theta;                           ///< Number of theta bins
  int N_phi;                             ///< Number of phi bins
  double dtheta;                         ///< Theta resolution
  double dphi;                           ///< Phi resolution
  CMatrix E_x;                      ///< Accumulated E-field x-component
  CMatrix E_y;                      ///< Accumulated E-field y-component
  CMatrix E_z;                      ///< Accumulated E-field z-component

  /// @brief Construct speckle detector at z-position with speckle size
  /// @param z Detector z-coordinate
  /// @param n_theta Number of theta bins
  /// @param n_phi Number of phi bins
  AngleDetector(double z, int n_theta, int n_phi);

  /// @brief Record photon intersection with detector plane (overrides base)
  /// @param photon Photon to validate and record
  void record_hit(Photon &photon) override;

  /// @brief Create empty speckle detector copy for parallel processing
  /// @return Cloned speckle detector
  std::unique_ptr<Detector> clone() const override;

  /// @brief Merge results from another speckle detector
  /// @param other Speckle detector to merge from
  void merge_from(const Detector &other) override;
};


DetectionCondition make_theta_condition(double min_theta, double max_theta);
DetectionCondition make_phi_condition(double min_phi, double max_phi);
DetectionCondition make_position_condition(double min_x, double max_x, double min_y, double max_y);
DetectionCondition make_events_condition(uint min_events, uint max_events);

} // namespace luminis::core
