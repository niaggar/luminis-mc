#pragma once
#include <cstddef>
#include <luminis/core/photon.hpp>
#include <luminis/core/medium.hpp>
#include <luminis/math/vec.hpp>
#include <vector>
#include <functional>
#include <memory>
#include <map>

using namespace luminis::math;

namespace luminis::core
{
  struct DoubleComparator {
      bool operator()(double a, double b) const { return a < b - 1e-9; }
  };

  struct InteractionInfo
  {
    Vec3 intersection_point;
    std::complex<double> phase;
  };

  /// @brief Photon detector plane for recording photon hits
  struct Sensor
  {
    u_int id;                                     ///< Unique detector ID
    Vec3 origin;                                  ///< Sensor position
    Vec3 normal;                                  ///< Surface normal (forward)
    Vec3 backward_normal;                         ///< Surface normal (backward)
    Vec3 n_polarization;                          ///< Polarization basis vector n
    Vec3 m_polarization;                          ///< Polarization basis vector m
    std::size_t hits{0};                          ///< Total photon hits
    // TODO: Implement a form to control how the contribution will be handled
    bool absorb_photons{true};                  ///< Whether photons are absorbed (terminate) upon detection
    bool estimator_enabled{true};                 ///< Whether photons are partially absorbed (weight reduced) upon detection

    // Detection filters
    bool filter_theta_enabled = false;
    double filter_theta_min = 0.0;
    double filter_theta_max = 0.0;
    double _cache_cos_theta_min = 0.0;
    double _cache_cos_theta_max = 0.0;

    bool filter_phi_enabled = false;
    double filter_phi_min = 0.0;
    double filter_phi_max = 0.0;

    bool filter_position_enabled = false;
    double filter_x_min = 0.0;
    double filter_x_max = 0.0;
    double filter_y_min = 0.0;
    double filter_y_max = 0.0;

    /// @brief Construct detector at z-position
    /// @param z Sensor z-coordinate
    Sensor(double z);
    virtual ~Sensor() = default;
    Sensor(const Sensor &) = delete;
    Sensor &operator=(const Sensor &) = delete;
    Sensor(Sensor &&) = default;
    Sensor &operator=(Sensor &&) = default;

    /// @brief Record photon intersection with detector plane
    /// @param photon Photon to validate and record
    virtual void process_hit(Photon &photon, InteractionInfo &info, const Medium &medium) = 0;

    /// @brief Process photon contribution for estimator mode (partial absorption)
    /// @param photon Photon to validate and process
    /// @param medium Medium through which the photon is propagating
    virtual void process_estimation(const Photon &photon, const Medium &medium);

    /// @brief Create empty detector copy for parallel processing
    virtual std::unique_ptr<Sensor> clone() const = 0;

    /// @brief Merge results from another detector
    /// @param other Sensor to merge from
    virtual void merge_from(const Sensor &other) = 0;

    /// @brief Set detection condition for theta angle
    /// @param min Minimum angle (rads)
    /// @param max Maximum angle (rads)
    void set_theta_limit(double min, double max);

    /// @brief Set detection condition for phi angle
    /// @param min Minimum angle (rads)
    /// @param max Maximum angle (rads)
    void set_phi_limit(double min, double max);

    /// @brief Set detection condition for position
    /// @param x_min Minimum x-coordinate
    /// @param x_max Maximum x-coordinate
    /// @param y_min Minimum y-coordinate
    /// @param y_max Maximum y-coordinate
    void set_position_limit(double x_min, double x_max, double y_min, double y_max);

    /// @brief Validate all detection conditions
    /// @param hit_point Point of intersection on the detector
    /// @param hit_direction Direction of the photon at the hit point
    /// @return True none of the conditions are violated, false if any condition is violated
    bool check_conditions(const Vec3 &hit_point, const Vec3 &hit_direction) const;

    /// @brief Enable or disable estimator mode (partial absorption)
    /// @param enabled If true, the photon will contribute to the detector although it have not reached the detector plane.
    void set_estimator_mode(bool enabled);
  };

  struct PhotonRecordSensor : public Sensor
  {
    std::vector<PhotonRecord> recorded_photons{};

    PhotonRecordSensor(double z);
    std::unique_ptr<Sensor> clone() const override;
    void merge_from(const Sensor &other) override;
    void process_hit(Photon &photon, InteractionInfo &info, const Medium &medium) override;
  };

  struct PlanarFieldSensor : public Sensor
  {
    CMatrix Ex, Ey;
    int N_x, N_y;
    double len_x, len_y;
    double dx, dy;

    PlanarFieldSensor(double z, double len_x, double len_y, double dx, double dy);
    std::unique_ptr<Sensor> clone() const override;
    void merge_from(const Sensor &other) override;
    void process_hit(Photon &photon, InteractionInfo &info, const Medium &medium) override;
    void process_estimation(const Photon &photon, const Medium &medium) override;
  };

  struct PlanarFluenceSensor : public Sensor
  {
    int N_t;
    int N_x, N_y;
    double len_x, len_y, len_t;
    double dx, dy, dt;
    std::vector<Matrix> S0_t, S1_t, S2_t, S3_t;

    PlanarFluenceSensor(double z, double len_x, double len_y, double len_t, double dx, double dy, double dt);
    std::unique_ptr<Sensor> clone() const override;
    void merge_from(const Sensor &other) override;
    void process_hit(Photon &photon, InteractionInfo &info, const Medium &medium) override;
    void process_estimation(const Photon &photon, const Medium &medium) override;
  };

  struct PlanarCBSSensor : public Sensor
  {
    int N_x, N_y;
    double len_x, len_y;
    double dx, dy;
    Matrix S0, S1, S2, S3;

    PlanarCBSSensor(double len_x, double len_y, double dx, double dy);
    std::unique_ptr<Sensor> clone() const override;
    void merge_from(const Sensor &other) override;
    void process_hit(Photon &photon, InteractionInfo &info, const Medium &medium) override;
  };

  struct FarFieldFluenceSensor : public Sensor
  {
    int N_theta, N_phi;
    double theta_max, phi_max;
    double dtheta, dphi;
    Matrix S0, S1, S2, S3;

    FarFieldFluenceSensor(double z, double theta_max, double phi_max, int n_theta, int n_phi);
    std::unique_ptr<Sensor> clone() const override;
    void merge_from(const Sensor &other) override;
    void process_hit(Photon &photon, InteractionInfo &info, const Medium &medium) override;
    void process_estimation(const Photon &photon, const Medium &medium) override;
  };

  struct FarFieldCBSSensor : public Sensor
  {
    int N_theta, N_phi;
    double theta_max, phi_max;
    double dtheta, dphi;
    Matrix S0_coh, S1_coh, S2_coh, S3_coh;
    Matrix S0_incoh, S1_incoh, S2_incoh, S3_incoh;

    FarFieldCBSSensor(double theta_max, double phi_max, int n_theta, int n_phi);
    std::unique_ptr<Sensor> clone() const override;
    void merge_from(const Sensor &other) override;
    void process_hit(Photon &photon, InteractionInfo &info, const Medium &medium) override;
    void process_estimation(const Photon &photon, const Medium &medium) override;
  };

  CVec2 coherent_estimation(const Photon &photon, const Medium &medium, Matrix last_scattering_P);
  void coherent_calculation(Photon &photon, const Medium &medium);
  

  struct StatisticsSensor : public Sensor
  {
    std::vector<int> events_histogram;
    std::vector<int> theta_histogram;
    std::vector<int> phi_histogram;
    std::vector<int> depth_histogram;
    std::vector<int> time_histogram;
    std::vector<int> weight_histogram;

    // Settings for histograms
    bool events_histogram_bins_set = false;
    int max_events = 0;
    bool theta_histogram_bins_set = false;
    double min_theta = 0.0;
    double max_theta = 0.0;
    int n_bins_theta = 0;
    double dtheta = 0.0;
    bool phi_histogram_bins_set = false;
    double min_phi = 0.0;
    double max_phi = 0.0;
    int n_bins_phi = 0;
    double dphi = 0.0;
    bool depth_histogram_bins_set = false;
    double max_depth = 0.0;
    int n_bins_depth = 0;
    double ddepth = 0.0;
    bool time_histogram_bins_set = false;
    double max_time = 0.0;
    int n_bins_time = 0;
    double dtime = 0.0;
    bool weight_histogram_bins_set = false;
    double max_weight = 0.0;
    int n_bins_weight = 0;
    double dweight = 0.0;

    StatisticsSensor(double z);

    void set_events_histogram_bins(int max_events);
    void set_theta_histogram_bins(double min_theta, double max_theta, int n_bins);
    void set_phi_histogram_bins(double min_phi, double max_phi, int n_bins);
    void set_depth_histogram_bins(double max_depth, int n_bins);
    void set_time_histogram_bins(double max_time, int n_bins);
    void set_weight_histogram_bins(double max_weight, int n_bins);

    std::unique_ptr<Sensor> clone() const override;
    void merge_from(const Sensor &other) override;
    void process_hit(Photon &photon, InteractionInfo &info, const Medium &medium) override;
  };

  /// @brief Collection of multiple detectors
  struct SensorsGroup
  {
    std::vector<std::unique_ptr<Sensor>> detectors;
    std::map<double, std::vector<Sensor*>, DoubleComparator> z_layers;
    std::vector<Sensor*> active_estimators;

    SensorsGroup() = default;
    SensorsGroup(const SensorsGroup &) = delete;
    SensorsGroup &operator=(const SensorsGroup &) = delete;
    SensorsGroup(SensorsGroup &&) = default;
    SensorsGroup &operator=(SensorsGroup &&) = default;

    /// @brief Add a detector to the multi-detector
    /// @param detector Sensor to add
    void add_detector(std::unique_ptr<Sensor> detector);

    /// @brief Record hit by all detectors
    /// @param photon Photon to record
    bool record_hit(Photon &photon, const Medium &medium);

    /// @brief Process estimator contribution by all detectors
    /// @param photon Photon to process
    /// @param medium Medium through which the photon is propagating
    void run_estimators(const Photon &photon, const Medium &medium);

    /// @brief Merge results from another multi-detector
    /// @param other Multi-detector to merge from
    void merge_from(const SensorsGroup &other);

    /// @brief Clone multi-detector
    /// @return Cloned multi-detector
    std::unique_ptr<SensorsGroup> clone() const;
  };

} // namespace luminis::core
