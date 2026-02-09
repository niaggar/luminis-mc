#pragma once
#include <cstddef>
#include <luminis/core/photon.hpp>
#include <luminis/math/vec.hpp>
#include <vector>
#include <functional>

using namespace luminis::math;

namespace luminis::core
{

  using DetectionCondition = std::function<bool(const Photon &)>;

  /// @brief Angular intensity distribution in spherical coordinates
  struct AngularIntensity
  {
    Matrix Ix, Iy, Iz, I_total, Ico, Icros; ///< Intensity components and total
    int N_theta, N_phi;                     ///< Number of bins
    double theta_max, phi_max;              ///< Maximum angles (radians)
    double dtheta, dphi;                    ///< Angular resolution

    AngularIntensity(int n_theta, int n_phi, double theta_max, double phi_max)
        : N_theta(n_theta), N_phi(n_phi), theta_max(theta_max), phi_max(phi_max),
          dtheta(theta_max / n_theta), dphi(phi_max / n_phi),
          Ix(n_theta, n_phi), Iy(n_theta, n_phi), Iz(n_theta, n_phi), I_total(n_theta, n_phi), Ico(n_theta, n_phi), Icros(n_theta, n_phi) {}
  };

  /// @brief Spatial intensity distribution in cartesian coordinates
  struct SpatialIntensity
  {
    Matrix Ix, Iy, Iz, I_total, Ico, Icros; ///< Intensity components and total
    int N_x, N_y;                           ///< Number of bins
    double x_len, y_len;                    ///< Physical dimensions
    double dx, dy;                          ///< Spatial resolution

    SpatialIntensity(int n_x, int n_y, double x_len, double y_len)
        : N_x(n_x), N_y(n_y), x_len(x_len), y_len(y_len),
          dx(x_len / n_x), dy(y_len / n_y),
          Ix(n_x, n_y), Iy(n_x, n_y), Iz(n_x, n_y), I_total(n_x, n_y), Ico(n_x, n_y), Icros(n_x, n_y) {}
  };

  /// @brief Photon detector plane for recording photon hits
  struct Detector
  {
    u_int id;                                     ///< Unique detector ID
    Vec3 origin;                                  ///< Detector position
    Vec3 normal;                                  ///< Surface normal (forward)
    Vec3 backward_normal;                         ///< Surface normal (backward)
    Vec3 n_polarization;                          ///< Polarization basis vector n
    Vec3 m_polarization;                          ///< Polarization basis vector m
    std::vector<PhotonRecord> recorded_photons{}; ///< Recorded photon data
    std::size_t hits{0};                          ///< Total photon hits

    // Detection filters
    bool filter_theta_enabled = false;
    double filter_theta_min = 0.0;
    double filter_theta_max = 0.0;
    double _cache_cos_theta_min = 0.0;
    double _cache_cos_theta_max = 0.0;

    bool filter_phi_enabled = false;
    double filter_phi_min = 0.0;
    double filter_phi_max = 0.0;

    /// @brief Construct detector at z-position
    /// @param z Detector z-coordinate
    Detector(double z);
    Detector(const Detector &) = delete;
    Detector &operator=(const Detector &) = delete;
    Detector(Detector &&) = default;
    Detector &operator=(Detector &&) = default;

    /// @brief Record photon intersection with detector plane
    /// @param photon Photon to validate and record
    virtual bool record_hit(Photon &photon, std::function<void()> coherent_calculation);

    /// @brief Create empty detector copy for parallel processing
    virtual std::unique_ptr<Detector> clone() const;

    /// @brief Merge results from another detector
    /// @param other Detector to merge from
    virtual void merge_from(const Detector &other);

    /// @brief Set detection condition for theta angle
    /// @param min Minimum angle (rads)
    /// @param max Maximum angle (rads)
    void set_theta_limit(double min, double max);

    /// @brief Set detection condition for phi angle
    /// @param min Minimum angle (rads)
    /// @param max Maximum angle (rads)
    void set_phi_limit(double min, double max);

    /// @brief Validate all detection conditions
    /// @param photon Photon to validate
    /// @return True if all conditions are met
    bool check_conditions(const Photon &photon) const;
  };

  /// @brief Angular detector for recording scattered light field
  struct AngleDetector : public Detector
  {
    int N_theta;   ///< Number of theta bins
    int N_phi;     ///< Number of phi bins
    double dtheta; ///< Theta resolution
    double dphi;   ///< Phi resolution
    CMatrix E_x;   ///< Accumulated E-field x-component
    CMatrix E_y;   ///< Accumulated E-field y-component
    CMatrix E_z;   ///< Accumulated E-field z-component

    /// @brief Construct speckle detector at z-position with speckle size
    /// @param z Detector z-coordinate
    /// @param n_theta Number of theta bins
    /// @param n_phi Number of phi bins
    AngleDetector(double z, int n_theta, int n_phi);

    /// @brief Record photon intersection with detector plane (overrides base)
    /// @param photon Photon to validate and record
    bool record_hit(Photon &photon, std::function<void()> coherent_calculation) override;

    /// @brief Create empty speckle detector copy for parallel processing
    /// @return Cloned speckle detector
    std::unique_ptr<Detector> clone() const override;

    /// @brief Merge results from another speckle detector
    /// @param other Speckle detector to merge from
    void merge_from(const Detector &other) override;
  };

  /// @brief Histogram detector for recording photon event counts
  struct HistogramDetector : public Detector
  {
    std::vector<int> histogram;
    u_int max_events{0};

    HistogramDetector(double z, u_int n_bins)
        : Detector(z), histogram(n_bins, 0) {}

    bool record_hit(Photon &photon, std::function<void()> coherent_calculation) override;

    std::unique_ptr<Detector> clone() const override;

    void merge_from(const Detector &other) override;
  };

  /// @brief Theta Histogram detector for recording photon theta angle counts
  struct ThetaHistogramDetector : public Detector
  {
    std::vector<int> histogram;

    ThetaHistogramDetector(double z, u_int n_bins)
        : Detector(z), histogram(n_bins, 0) {}

    bool record_hit(Photon &photon, std::function<void()> coherent_calculation) override;

    std::unique_ptr<Detector> clone() const override;

    void merge_from(const Detector &other) override;
  };

  /// @brief Spatial detector for recording spatial intensity distribution
  struct SpatialDetector : public Detector
  {
    int N_x;       ///< Number of x bins
    int N_y;       ///< Number of y bins
    double dx;     ///< x resolution
    double dy;     ///< y resolution
    CMatrix E_x;    ///< Accumulated E-field x-component
    CMatrix E_y;    ///< Accumulated E-field y-component
    CMatrix E_z;    ///< Accumulated E-field z-component

    Matrix I_x;    ///< Accumulated Intensity x-component
    Matrix I_y;    ///< Accumulated Intensity y-component
    Matrix I_z;    ///< Accumulated Intensity z-component
    Matrix I_plus;   ///< Accumulated Copolarized Intensity
    Matrix I_minus; ///< Accumulated Crosspolarized Intensity

    /// @brief Construct spatial detector at z-position with spatial size
    /// @param z Detector z-coordinate
    /// @param x_len Physical x length
    /// @param y_len Physical y length
    /// @param n_x Number of x bins
    /// @param n_y Number of y bins
    SpatialDetector(double z, double x_len, double y_len, int n_x, int n_y);

    /// @brief Record photon intersection with detector plane (overrides base)
    /// @param photon Photon to validate and record
    bool record_hit(Photon &photon, std::function<void()> coherent_calculation) override;

    /// @brief Create empty spatial detector copy for parallel processing
    /// @return Cloned spatial detector
    std::unique_ptr<Detector> clone() const override;

    /// @brief Merge results from another spatial detector
    /// @param other Spatial detector to merge from
    void merge_from(const Detector &other) override;
  };

  struct SpatialCoherentDetector : public Detector
  {
    int N_x;   ///< Number of x bins
    int N_y;     ///< Number of y bins
    double dx; ///< x resolution
    double dy;   ///< y resolution
    Matrix I_x;   ///< Accumulated Intensity x-component
    Matrix I_y;   ///< Accumulated Intensity y-component
    Matrix I_z;   ///< Accumulated Intensity z-component

    Matrix I_inco_x;   ///< Accumulated Incoherent Intensity x-component
    Matrix I_inco_y;   ///< Accumulated Incoherent Intensity y-component
    Matrix I_inco_z;   ///< Accumulated Incoherent Intensity z-component

    std::vector<double> I_x_theta;
    std::vector<double> I_y_theta;
    std::vector<double> I_z_theta;

    std::vector<double> I_inco_x_theta;
    std::vector<double> I_inco_y_theta;
    std::vector<double> I_inco_z_theta;

    /// @brief Construct spatial coherent detector at z-position with spatial size
    /// @param z Detector z-coordinate
    /// @param x_len Physical x length
    /// @param y_len Physical y length
    /// @param n_x Number of x bins
    /// @param n_y Number of y bins
    SpatialCoherentDetector(double z, double x_len, double y_len, int n_x, int n_y);

    /// @brief Record photon intersection with detector plane (overrides base)
    /// @param photon Photon to validate and record
    bool record_hit(Photon &photon, std::function<void()> coherent_calculation) override;

    /// @brief Create empty spatial detector copy for parallel processing
    /// @return Cloned spatial detector
    std::unique_ptr<Detector> clone() const override;

    /// @brief Merge results from another spatial detector
    /// @param other Spatial detector to merge from
    void merge_from(const Detector &other) override;
  };

  struct AngularCoherentDetector : public Detector
  {
    int N_theta;   ///< Number of theta bins
    double dtheta; ///< Theta resolution
    std::vector<double> I_x;   ///< Accumulated Intensity x-component
    std::vector<double> I_y;   ///< Accumulated Intensity y-component
    std::vector<double> I_z;   ///< Accumulated Intensity z-component
    std::vector<double> I_plus;   ///< Accumulated Coherent Intensity
    std::vector<double> I_minus; ///< Accumulated Crossed Intensity
    std::vector<double> I_total; ///< Accumulated Total Intensity

    std::vector<double> I_inco_x;   ///< Accumulated Incoherent Intensity x-component
    std::vector<double> I_inco_y;   ///< Accumulated Incoherent Intensity y-component
    std::vector<double> I_inco_z;   ///< Accumulated Incoherent Intensity z-component
    std::vector<double> I_inco_plus;   ///< Accumulated Incoherent Coherent Intensity
    std::vector<double> I_inco_minus; ///< Accumulated Incoherent Crossed Intensity
    std::vector<double> I_inco_total; ///< Accumulated Incoherent Total Intensity

    /// @brief Construct angular coherent detector at z-position with angular size
    /// @param z Detector z-coordinate
    /// @param n_theta Number of theta bins
    AngularCoherentDetector(double z, int n_theta, double max_theta);

    /// @brief Record photon intersection with detector plane (overrides base)
    /// @param photon Photon to validate and record
    bool record_hit(Photon &photon, std::function<void()> coherent_calculation) override;

    /// @brief Create empty angular detector copy for parallel processing
    /// @return Cloned angular detector
    std::unique_ptr<Detector> clone() const override;

    /// @brief Merge results from another angular detector
    /// @param other Angular detector to merge from
    void merge_from(const Detector &other) override;
  };

  /// @brief Collection of multiple detectors
  struct MultiDetector
  {
    std::vector<std::unique_ptr<Detector>> detectors; ///< Collection of detectors

    MultiDetector() = default;
    MultiDetector(const MultiDetector &) = delete;
    MultiDetector &operator=(const MultiDetector &) = delete;
    MultiDetector(MultiDetector &&) = default;
    MultiDetector &operator=(MultiDetector &&) = default;

    /// @brief Add a detector to the multi-detector
    /// @param detector Detector to add
    void add_detector(std::unique_ptr<Detector> detector);

    /// @brief Record hit by all detectors
    /// @param photon Photon to record
    bool record_hit(Photon &photon, std::function<void()> coherent_calculation);

    /// @brief Merge results from another multi-detector
    /// @param other Multi-detector to merge from
    void merge_from(const MultiDetector &other);

    /// @brief Clone multi-detector
    /// @return Cloned multi-detector
    std::unique_ptr<MultiDetector> clone() const;
  };

  DetectionCondition make_theta_condition(double min_theta, double max_theta);
  DetectionCondition make_phi_condition(double min_phi, double max_phi);
  DetectionCondition make_position_condition(double min_x, double max_x, double min_y, double max_y);
  DetectionCondition make_events_condition(uint min_events, uint max_events);

} // namespace luminis::core
