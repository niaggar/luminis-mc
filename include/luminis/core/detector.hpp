/// @file detector.hpp
/// @brief Photon detection sensors for Monte Carlo light transport simulations.
///
/// Defines the sensor hierarchy used to collect photon statistics during simulation.
/// All sensors are placed at a z-plane and process photons that cross that plane.
///
/// The sensor system supports two detection modes:
///   - **Direct hit**: The photon physically crosses the detector z-plane during
///     propagation. SensorsGroup::record_hit() computes the exact intersection point
///     and calls process_hit() on each matching sensor.
///   - **Estimator**: After each scattering event, sensors with estimator_enabled=true
///     receive a call to process_estimation(), which calculates the photon's virtual
///     contribution to the detector without requiring physical intersection.
///
/// Each sensor type stores its results in a different representation:
///   - PhotonRecordSensor:   per-photon records (full state snapshot)
///   - PlanarFieldSensor:    complex electric field on a spatial grid
///   - PlanarFluenceSensor:  Stokes parameters on a spatial grid (optionally time-resolved)
///   - FarFieldFluenceSensor: Stokes parameters on an angular grid (theta, phi)
///   - FarFieldCBSSensor:    coherent and incoherent Stokes for CBS enhancement
///   - StatisticsSensor:     configurable histograms of photon properties
///
/// **Thread safety**: Sensors are NOT thread-safe. For parallel simulations, each thread
/// works on a cloned SensorsGroup (via clone()). After all threads finish, results are
/// combined via merge_from().

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
  /// @brief Comparator for floating-point keys in ordered containers.
  /// @details Uses a tolerance of 1e-9 to treat nearly-equal z-coordinates as identical,
  ///          preventing duplicate z-layers from floating-point imprecision.
  struct DoubleComparator
  {
    bool operator()(double a, double b) const { return a < b - 1e-9; }
  };

  /// @brief Information about a photon-detector intersection event.
  /// @details Computed by SensorsGroup::record_hit() when a photon's trajectory
  ///          crosses a detector z-plane, then passed to each sensor's process_hit().
  struct InteractionInfo
  {
    Vec3 intersection_point;    ///< Exact (x, y, z) coordinates where the photon crossed the detector plane.
    std::complex<double> phase; ///< Accumulated optical phase exp(i * k * optical_path) at the intersection point.
  };

  // ═══════════════════════════════════════════════════════════════════════════
  // Sensor — Abstract base class
  // ═══════════════════════════════════════════════════════════════════════════

  /// @brief Abstract base class for all photon detection sensors.
  ///
  /// A sensor is a virtual infinite plane at a fixed z-coordinate. It is always
  /// oriented with its normal along the z-axis. Photons that cross this plane
  /// during propagation may be recorded, depending on the applied filters.
  ///
  /// **Filters**: Each sensor supports optional acceptance filters that reject
  /// photons based on polar angle (theta), azimuthal angle (phi), or spatial
  /// position (x, y) at the intersection point. Multiple filters stack
  /// conjunctively (all must pass).
  ///
  /// **Lifecycle in parallel simulation**:
  /// 1. User creates sensors and adds them to a SensorsGroup.
  /// 2. Each thread clones the entire SensorsGroup (clone()).
  /// 3. Threads run independently, accumulating into their local sensors.
  /// 4. After joining, thread-local sensors are merged into the original (merge_from()).
  ///
  /// Subclasses must implement: process_hit(), clone(), merge_from().
  /// Subclasses may override: process_estimation() (default is no-op).
  struct Sensor
  {
    u_int id;             ///< Unique identifier assigned by SensorsGroup::add_detector().
    Vec3 origin;          ///< Position of the sensor plane center (always at (0, 0, z)).
    Vec3 normal;          ///< Outward-facing surface normal (always +z).
    Vec3 backward_normal; ///< Inward-facing surface normal (always -z).
    Vec3 n_polarization;  ///< Reference polarization basis vector n (initialized to +y).
    Vec3 m_polarization;  ///< Reference polarization basis vector m (initialized to +x).
    std::size_t hits{0};  ///< Total number of photons that passed all filters and were processed.

    /// @brief If true, photons are terminated (alive = false) after being detected.
    /// @details Set to false to allow the same photon to be recorded by multiple sensors
    ///          or to continue propagating after detection.
    bool absorb_photons{false};

    /// @brief If true, this sensor participates in estimator-based detection.
    /// @details When enabled, SensorsGroup::run_estimators() calls process_estimation()
    ///          on this sensor after every scattering event, allowing collection of
    ///          virtual contributions without requiring physical plane intersection.
    bool estimator_enabled{false};

    /// @name Polar angle (theta) filter
    /// @brief Accepts photons whose exit angle satisfies theta_min <= theta <= theta_max.
    /// @details Theta is measured as acos(-dir_z), i.e., the angle between the photon
    ///          direction and the backward detector normal. The cosine values are
    ///          precomputed and cached for efficient per-photon evaluation.
    /// @{
    bool filter_theta_enabled = false;
    double filter_theta_min = 0.0;     ///< Minimum accepted polar angle [rad].
    double filter_theta_max = 0.0;     ///< Maximum accepted polar angle [rad].
    double _cache_cos_theta_min = 0.0; ///< Precomputed: min of cos(theta_min), cos(theta_max).
    double _cache_cos_theta_max = 0.0; ///< Precomputed: max of cos(theta_min), cos(theta_max).
    /// @}

    /// @name Azimuthal angle (phi) filter
    /// @brief Accepts photons whose azimuthal exit angle satisfies phi_min <= phi <= phi_max.
    /// @details Phi is computed as atan2(dir_y, dir_x), mapped to [0, 2*pi).
    /// @{
    bool filter_phi_enabled = false;
    double filter_phi_min = 0.0; ///< Minimum accepted azimuthal angle [rad].
    double filter_phi_max = 0.0; ///< Maximum accepted azimuthal angle [rad].
    /// @}

    /// @name Spatial position filter
    /// @brief Accepts photons whose intersection point (x, y) lies within a rectangular window.
    /// @{
    bool filter_position_enabled = false;
    double filter_x_min = 0.0; ///< Minimum accepted x-coordinate at the detector plane.
    double filter_x_max = 0.0; ///< Maximum accepted x-coordinate at the detector plane.
    double filter_y_min = 0.0; ///< Minimum accepted y-coordinate at the detector plane.
    double filter_y_max = 0.0; ///< Maximum accepted y-coordinate at the detector plane.
    /// @}

    /// @brief Construct a sensor centered at the given z-coordinate.
    /// @param z The z-position of the detection plane.
    /// @param absorb If true, photons are terminated after being detected.
    /// @param estimator If true, this sensor participates in estimator-based detection.
    /// @details Initializes the sensor with origin=(0,0,z), normal=+z, and default
    ///          polarization basis vectors m=+x, n=+y.
    Sensor(double z, bool absorb = true, bool estimator = false);
    virtual ~Sensor() = default;

    Sensor(const Sensor &) = delete;
    Sensor &operator=(const Sensor &) = delete;
    Sensor(Sensor &&) = default;
    Sensor &operator=(Sensor &&) = default;

    /// @brief Process a photon that has physically crossed this sensor's z-plane.
    /// @param photon   The photon being detected (may be modified, e.g., marked as dead).
    /// @param info     Intersection data (exact hit point and accumulated optical phase).
    /// @param medium   The scattering medium (needed for matrix lookups in some sensors).
    /// @note Called by SensorsGroup::record_hit() after the photon passes all filters.
    ///       Subclasses accumulate their specific quantities (fields, Stokes, records, etc.).
    virtual void process_hit(Photon &photon, InteractionInfo &info, const Medium &medium) = 0;

    /// @brief Compute a virtual (estimator) contribution from a photon after scattering.
    /// @param photon The photon in its current state (not modified).
    /// @param medium The scattering medium (used for phase function evaluation).
    /// @details Called by SensorsGroup::run_estimators() after every scattering event
    ///          for sensors that have estimator_enabled = true. The default implementation
    ///          is a no-op; override in subclasses that support estimator-based detection.
    ///
    ///          The estimator approach calculates what the photon's contribution *would be*
    ///          if it were scattered directly toward the detector, applying the appropriate
    ///          phase function weight and exponential attenuation for the remaining distance.
    virtual void process_estimation(const Photon &photon, const Medium &medium);

    /// @brief Create an empty clone of this sensor with identical configuration but zeroed data.
    /// @return A new sensor of the same type with all filter settings copied.
    /// @details Used by SensorsGroup::clone() to create thread-local sensor copies
    ///          for parallel simulation. The clone must have the same type, geometry,
    ///          grid dimensions, and filter settings, but empty/zero accumulators.
    virtual std::unique_ptr<Sensor> clone() const = 0;

    /// @brief Merge accumulated results from another sensor of the same type.
    /// @param other The source sensor to merge from (must be the same concrete type).
    /// @details Called after parallel simulation to combine thread-local results.
    ///          Adds the other sensor's hit count and accumulated data to this sensor.
    /// @throws std::bad_cast if @p other is not the same concrete type as this sensor.
    virtual void merge_from(const Sensor &other) = 0;

    /// @brief Set the polar angle acceptance window.
    /// @param min Minimum polar angle theta [rad].
    /// @param max Maximum polar angle theta [rad].
    /// @details Enables the theta filter. Precomputes and caches cos(min) and cos(max)
    ///          for efficient per-photon evaluation in check_conditions().
    void set_theta_limit(double min, double max);

    /// @brief Set the azimuthal angle acceptance window.
    /// @param min Minimum azimuthal angle phi [rad].
    /// @param max Maximum azimuthal angle phi [rad].
    /// @details Enables the phi filter. Phi is computed as atan2(dir_y, dir_x) in [0, 2*pi).
    void set_phi_limit(double min, double max);

    /// @brief Set a rectangular spatial acceptance window on the detector plane.
    /// @param x_min Minimum x-coordinate.
    /// @param x_max Maximum x-coordinate.
    /// @param y_min Minimum y-coordinate.
    /// @param y_max Maximum y-coordinate.
    /// @details Enables the position filter. Only photons hitting within this rectangle
    ///          will be accepted.
    void set_position_limit(double x_min, double x_max, double y_min, double y_max);

    /// @brief Test whether a photon passes all enabled acceptance filters.
    /// @param hit_point     The (x, y, z) intersection point on the detector plane.
    /// @param hit_direction The photon's propagation direction at the hit point.
    /// @return true if the photon satisfies ALL enabled filters, false if any filter rejects it.
    /// @details Filters are evaluated in order: theta, phi, position. Evaluation
    ///          short-circuits on the first failing filter for efficiency.
    bool check_conditions(const Vec3 &hit_point, const Vec3 &hit_direction) const;

    /// @brief Enable or disable estimator-based detection for this sensor.
    /// @param enabled If true, process_estimation() will be called after every scattering event.
    void set_estimator_mode(bool enabled);
  };

  // ═══════════════════════════════════════════════════════════════════════════
  // PhotonRecordSensor — Full per-photon state recording
  // ═══════════════════════════════════════════════════════════════════════════

  /// @brief Sensor that stores a complete PhotonRecord for each detected photon.
  ///
  /// This is the most general-purpose sensor: it captures the full state of each
  /// photon at the moment of detection (position, direction, polarization, weight,
  /// optical path, number of events, etc.) as a PhotonRecord struct.
  ///
  /// Useful for:
  ///   - Post-processing analysis in Python (speckle patterns, custom statistics)
  ///   - Debugging individual photon trajectories
  ///   - Any analysis that requires access to individual photon data
  ///
  /// @warning Memory usage grows linearly with the number of detected photons.
  ///          For large simulations, consider using an accumulating sensor instead.
  struct PhotonRecordSensor : public Sensor
  {
    std::vector<PhotonRecord> recorded_photons{}; ///< All recorded photon snapshots.

    /// @brief Construct a photon record sensor at the given z-coordinate.
    /// @param z The z-position of the detection plane.
    /// @param absorb If true, photons are terminated after being detected.
    /// @details Initializes the sensor with origin=(0,0,z), normal=+z, and default
    ///          polarization basis vectors m=+x, n=+y.
    PhotonRecordSensor(double z, bool absorb = true);
    std::unique_ptr<Sensor> clone() const override;
    void merge_from(const Sensor &other) override;

    /// @brief Record the photon's full state as a PhotonRecord.
    /// @details Captures: events, penetration depth, optical path, weight, arrival time,
    ///          positions (first/last scattering, detector), direction, polarization basis,
    ///          and both forward and reverse polarization states.
    void process_hit(Photon &photon, InteractionInfo &info, const Medium &medium) override;
  };

  // ═══════════════════════════════════════════════════════════════════════════
  // PlanarFieldSensor — Complex electric field on a spatial grid
  // ═══════════════════════════════════════════════════════════════════════════

  /// @brief Sensor that accumulates the complex electric field on a 2D spatial grid.
  ///
  /// Collects the coherent sum of electric field amplitudes (Ex, Ey) on a grid of
  /// N_x × N_y pixels centered at the sensor origin. Each photon contributes its
  /// amplitude (not intensity), so the accumulated field preserves phase information
  /// and produces speckle-like interference patterns.
  ///
  /// The contribution of each photon to pixel (i, j) is:
  ///   E_x(i,j) += (Em * P[0,0] + En * P[1,0]) * exp(i*k*L) * sqrt(weight)
  ///   E_y(i,j) += (Em * P[0,1] + En * P[1,1]) * exp(i*k*L) * sqrt(weight)
  ///
  /// where Em, En are the local polarization components, P is the photon's local
  /// basis matrix, and L is the optical path. The sqrt(weight) factor ensures that
  /// |E|^2 gives the correct intensity for the Monte Carlo weight scheme.
  ///
  /// Also supports estimator-based detection (process_estimation) which calculates
  /// a virtual scatter toward the detector, including phase function evaluation,
  /// exponential attenuation, and proper polarization rotation.
  ///
  /// Useful for: speckle pattern simulation, coherent imaging, near-field interference.
  struct PlanarFieldSensor : public Sensor
  {
    CMatrix Ex, Ey;      ///< Accumulated complex electric field components [N_x × N_y].
    int N_x, N_y;        ///< Number of grid pixels in x and y.
    double len_x, len_y; ///< Physical dimensions of the sensor area [same units as simulation].
    double dx, dy;       ///< Pixel size in x and y.

    /// @brief Construct a planar field sensor.
    /// @param z     Z-coordinate of the detection plane.
    /// @param len_x Total sensor width in x.
    /// @param len_y Total sensor height in y.
    /// @param dx    Pixel size in x (N_x = ceil(len_x / dx)).
    /// @param dy    Pixel size in y (N_y = ceil(len_y / dy)).
    /// @param absorb If true, photons are terminated after being detected.
    /// @param estimator If true, this sensor participates in estimator-based detection.
    /// @details Automatically sets a position filter to the sensor area bounds.
    PlanarFieldSensor(double z, double len_x, double len_y, double dx, double dy, bool absorb = true, bool estimator = false);
    std::unique_ptr<Sensor> clone() const override;
    void merge_from(const Sensor &other) override;
    void process_hit(Photon &photon, InteractionInfo &info, const Medium &medium) override;
    void process_estimation(const Photon &photon, const Medium &medium) override;
  };

  // ═══════════════════════════════════════════════════════════════════════════
  // PlanarFluenceSensor — Stokes parameters on a spatial (+ time) grid
  // ═══════════════════════════════════════════════════════════════════════════

  /// @brief Sensor that accumulates Stokes parameters (S0, S1, S2, S3) on a 2D
  ///        spatial grid, optionally resolved in time.
  ///
  /// Unlike PlanarFieldSensor (which stores complex amplitudes), this sensor stores
  /// the incoherent sum of Stokes parameters. Each photon's contribution is computed
  /// from its field components projected onto the detector coordinate system:
  ///   S0 = |Ex|^2 + |Ey|^2    (total intensity)
  ///   S1 = |Ex|^2 - |Ey|^2    (linear horizontal/vertical preference)
  ///   S2 = 2 * Re(Ex * Ey*)   (linear +45/-45 preference)
  ///   S3 = 2 * Im(Ex * Ey*)   (circular polarization preference)
  ///
  /// Time resolution:
  ///   - If dt > 0, data is stored as S0_t[t_idx](x_idx, y_idx) with N_t time bins.
  ///   - If dt == 0, a single time bin is used (time-integrated).
  ///
  /// Also supports estimator-based detection for improved convergence.
  ///
  /// Useful for: spatially-resolved fluence maps, time-resolved imaging, polarimetry.
  struct PlanarFluenceSensor : public Sensor
  {
    int N_t;             ///< Number of time bins (1 if time-integrated).
    int N_x, N_y;        ///< Number of grid pixels in x and y.
    double len_x, len_y; ///< Physical dimensions of the sensor area.
    double len_t;        ///< Total time window length [same time units as simulation].
    double dx, dy;       ///< Pixel size in x and y.
    double dt;           ///< Time bin width (0 for time-integrated).

    /// @brief Time-resolved Stokes parameter grids: S_t[time_index](x_index, y_index).
    /// @details Each element is an N_x × N_y Matrix. The vector has N_t elements.
    std::vector<Matrix> S0_t, S1_t, S2_t, S3_t;

    /// @brief Construct a planar fluence sensor.
    /// @param z     Z-coordinate of the detection plane.
    /// @param len_x Total sensor width in x.
    /// @param len_y Total sensor height in y.
    /// @param len_t Total time window length (0 for time-integrated).
    /// @param dx    Pixel size in x.
    /// @param dy    Pixel size in y.
    /// @param dt    Time bin width (0 for time-integrated, meaning N_t = 1).
    /// @details Automatically sets a position filter to the sensor area bounds.
    PlanarFluenceSensor(double z, double len_x, double len_y, double len_t, double dx, double dy, double dt, bool absorb = true, bool estimator = false);
    std::unique_ptr<Sensor> clone() const override;
    void merge_from(const Sensor &other) override;
    void process_hit(Photon &photon, InteractionInfo &info, const Medium &medium) override;
    void process_estimation(const Photon &photon, const Medium &medium) override;
  };

  // ═══════════════════════════════════════════════════════════════════════════
  // PlanarCBSSensor — Coherent backscattering on a spatial grid (WIP)
  // ═══════════════════════════════════════════════════════════════════════════

  /// @brief Sensor for coherent backscattering (CBS) analysis on a near-field spatial grid.
  ///
  /// @warning **Work in progress.** The process_hit() implementation is currently incomplete
  ///          (most code is commented out). The data structures are in place but the CBS
  ///          coherent/incoherent decomposition is not yet functional.
  ///
  /// Placed at z=0 (backscattering surface). Intended to accumulate Stokes parameters
  /// with CBS enhancement in a spatial (x, y) grid representation.
  struct PlanarCBSSensor : public Sensor
  {
    int N_x, N_y;          ///< Number of grid pixels in x and y.
    double len_x, len_y;   ///< Physical dimensions of the sensor area.
    double dx, dy;         ///< Pixel size in x and y.
    Matrix S0, S1, S2, S3; ///< Accumulated Stokes parameter grids [N_x × N_y].

    /// @brief Construct a planar CBS sensor at z=0.
    /// @param len_x Total sensor width in x.
    /// @param len_y Total sensor height in y.
    /// @param dx    Pixel size in x.
    /// @param dy    Pixel size in y.
    /// @param estimator If true, this sensor participates in estimator-based detection.
    /// @details Automatically sets a position filter to the sensor area bounds.
    PlanarCBSSensor(double len_x, double len_y, double dx, double dy, bool estimator = false);
    std::unique_ptr<Sensor> clone() const override;
    void merge_from(const Sensor &other) override;
    void process_hit(Photon &photon, InteractionInfo &info, const Medium &medium) override;
    void process_estimation(const Photon &photon, const Medium &medium) override;
  };

  // ═══════════════════════════════════════════════════════════════════════════
  // FarFieldFluenceSensor — Stokes parameters on an angular grid
  // ═══════════════════════════════════════════════════════════════════════════

  /// @brief Sensor that accumulates Stokes parameters on a far-field angular grid (theta, phi).
  ///
  /// Bins photons by their exit direction into an angular grid:
  ///   - theta (polar): measured from the backward normal (-z), so theta=0 means
  ///     exact backscattering. Range: [0, theta_max].
  ///   - phi (azimuthal): measured as atan2(dir_y, dir_x) in [0, phi_max].
  ///
  /// Each photon's Stokes parameters are computed from its polarization state
  /// projected onto the laboratory (x, y) frame and deposited into the
  /// corresponding angular bin.
  ///
  /// Supports estimator-based detection (currently commented out in implementation).
  ///
  /// Useful for: angular distribution of diffuse light, far-field radiation patterns.
  struct FarFieldFluenceSensor : public Sensor
  {
    int N_theta, N_phi;        ///< Number of angular bins in theta and phi.
    double theta_max, phi_max; ///< Maximum angular extents [rad].
    double dtheta, dphi;       ///< Angular bin widths: dtheta = theta_max / N_theta, dphi = phi_max / N_phi.
    Matrix S0, S1, S2, S3;     ///< Accumulated Stokes parameter grids [N_theta × N_phi].

    /// @brief Construct a far-field fluence sensor.
    /// @param theta_max Maximum polar angle [rad].
    /// @param phi_max   Maximum azimuthal angle [rad] (typically 2*pi).
    /// @param n_theta   Number of polar angle bins.
    /// @param n_phi     Number of azimuthal angle bins.
    /// @param estimator If true, this sensor participates in estimator-based detection.
    /// @details Automatically sets theta and phi filters to [0, theta_max] and [0, phi_max].
    FarFieldFluenceSensor(double theta_max, double phi_max, int n_theta, int n_phi, bool estimator = false);
    std::unique_ptr<Sensor> clone() const override;
    void merge_from(const Sensor &other) override;
    void process_hit(Photon &photon, InteractionInfo &info, const Medium &medium) override;
    void process_estimation(const Photon &photon, const Medium &medium) override;
  };

  // ═══════════════════════════════════════════════════════════════════════════
  // FarFieldCBSSensor — Coherent backscattering in far-field angular space
  // ═══════════════════════════════════════════════════════════════════════════

  /// @brief Primary sensor for Coherent Backscattering (CBS) simulation in far-field angular space.
  ///
  /// This sensor computes the CBS enhancement cone by accumulating both the *coherent*
  /// and *incoherent* Stokes parameters on an angular grid (theta, phi) for backscattered
  /// light. The CBS enhancement factor is obtained as:
  ///   eta(theta) = S0_coh(theta) / S0_incoh(theta)
  /// which approaches 2.0 at exact backscattering (theta=0) for conservative media.
  ///
  /// **Physics background:**
  /// For each photon trajectory with >= 2 scattering events, there exists a time-reversed
  /// (reciprocal) path visiting the same scatterers in reverse order. These two paths
  /// accumulate the same total optical path but different Jones matrices. Their
  /// interference produces the CBS enhancement.
  ///
  /// **How it works (process_hit):**
  /// 1. Requires photon.events >= 2 (single-scattering has no reciprocal partner).
  /// 2. Calls coherent_calculation() to compute the reverse-path polarization state
  ///    (photon.polarization_reverse) from stored scattering history.
  /// 3. Computes the CBS geometric phase factor:
  ///      path_phase = exp(i * k * (s_out + s_in) · (r_n - r_0))
  ///    where s_in is the incident direction, s_out the exit direction, and
  ///    r_0, r_n are the first and last scattering positions.
  ///    At exact backscattering (s_out = -s_in), this phase vanishes → constructive interference.
  /// 4. Projects both forward (E_f) and reverse (E_r) fields to laboratory (x, y) components.
  /// 5. Accumulates:
  ///      - Coherent:   |E_f + E_r|^2  → captures interference (CBS cone peak)
  ///      - Incoherent: |E_f|^2 + |E_r|^2  → baseline without interference
  ///
  /// **Required simulation setup:**
  ///   - SimConfig::track_reverse_paths must be true so that the photon stores the
  ///     scattering history (P0, P1, Pn1, Pn, r_0, r_n, matrix_T) needed for the
  ///     reverse-path calculation.
  ///
  /// **Post-processing:**
  ///   Use postprocess_farfield_cbs() to normalize the raw accumulated Stokes data
  ///   by solid angle and photon count.
  ///
  /// @see coherent_calculation() for the reverse-path polarization computation.
  /// @see postprocess_farfield_cbs() for result normalization.
  struct FarFieldCBSSensor : public Sensor
  {
    int N_theta, N_phi;        ///< Number of angular bins in theta and phi.
    double theta_max, phi_max; ///< Maximum angular extents [rad].
    double dtheta, dphi;       ///< Angular bin widths [rad].

    // --- NEW: partial photon / next-event estimator control ---
    double theta_pp_max{-1.0};     // si <0 => usa theta_max
    int theta_stride{1};           // subsampling para performance
    int phi_stride{1};

    // --- NEW: cache para normalización angular (depende de k) ---
    mutable double _I_norm{-1.0};
    mutable double _I_norm_k{0.0};

    /// @name Coherent Stokes grids
    /// @brief Accumulated |E_forward + E_reverse|^2 Stokes parameters [N_theta × N_phi].
    /// @details These capture the interference between forward and reverse paths,
    ///          producing the CBS enhancement cone centered at theta=0.
    /// @{
    Matrix S0_coh, S1_coh, S2_coh, S3_coh;
    /// @}

    /// @name Incoherent Stokes grids
    /// @brief Accumulated |E_forward|^2 + |E_reverse|^2 Stokes parameters [N_theta × N_phi].
    /// @details These represent the sum of individual intensities without interference,
    ///          serving as the flat baseline for computing the enhancement factor.
    /// @{
    Matrix S0_incoh, S1_incoh, S2_incoh, S3_incoh;
    /// @}

    /// @brief Construct a far-field CBS sensor at z=0.
    /// @param theta_max Maximum polar angle [rad] (measured from exact backscattering).
    /// @param phi_max   Maximum azimuthal angle [rad] (typically 2*pi).
    /// @param n_theta   Number of polar angle bins.
    /// @param n_phi     Number of azimuthal angle bins.
    /// @param estimator If true, this sensor participates in estimator-based detection.
    /// @details Automatically sets theta and phi filters to [0, theta_max] and [0, phi_max].
    FarFieldCBSSensor(double theta_max, double phi_max, int n_theta, int n_phi, bool estimator = false);
    std::unique_ptr<Sensor> clone() const override;
    void merge_from(const Sensor &other) override;

    /// @brief Process a backscattered photon, computing forward and reverse contributions.
    /// @details See the class-level documentation for the full algorithm description.
    ///          Ignored for photons with fewer than 2 scattering events.
    void process_hit(Photon &photon, InteractionInfo &info, const Medium &medium) override;

    /// @brief Estimator-based CBS contribution (currently unimplemented).
    /// @details The commented-out code would iterate over all (theta, phi) bins,
    ///          computing virtual scatter contributions and their reverse-path
    ///          counterparts for improved statistical convergence.
    void process_estimation(const Photon &photon, const Medium &medium) override;
  };

  // ═══════════════════════════════════════════════════════════════════════════
  // CBS helper functions
  // ═══════════════════════════════════════════════════════════════════════════

  /// @brief Compute the reverse-path polarization state for a given estimated exit direction.
  ///
  /// This is the estimator variant of the CBS reverse-path computation. Unlike
  /// coherent_calculation(), which uses the photon's actual exit direction, this
  /// function accepts an arbitrary exit basis (last_scattering_P) so it can be used
  /// by process_estimation() to evaluate virtual contributions toward any angular bin.
  ///
  /// @param photon             The photon with stored scattering history (P0, P1, Pn1, matrix_T, etc.).
  /// @param medium             The scattering medium (for scattering matrix evaluation).
  /// @param last_scattering_P  The 3×3 local basis matrix at the hypothetical exit point.
  /// @return The complex polarization vector (Em, En) of the reverse path, expressed
  ///         in the basis given by last_scattering_P.
  ///
  /// @see coherent_calculation() for the direct-hit version.
  CVec2 coherent_estimation_partial(
      const Photon &photon,
      const Medium &medium,
      const Matrix &P_last_in,  // frame de s_{n-1} (antes del scatter estimado)
      const Matrix &P_last_out, // frame de s_n     (después del scatter estimado)
      const CMatrix &Tmid);

  /// @brief Compute the reverse-path polarization and store it in photon.polarization_reverse.
  ///
  /// Uses the photon's stored scattering history to reconstruct the time-reversed
  /// (reciprocal) path's Jones chain and compute the resulting polarization state.
  ///
  /// The algorithm has three stages:
  ///
  /// **Stage A — First reverse scatter (at position r_n):**
  ///   The reverse path enters with the original incident direction s_0 and exits
  ///   toward -s_{n-1} (reverse of the penultimate leg). The scattering angle is
  ///   theta_a = acos(s_0 · (-s_{n-1})). Rotation matrices align the initial
  ///   polarization basis (m_0, n_0) to the scattering plane, then the Jones matrix
  ///   S(theta_a) is applied, followed by rotation into the (m_{n-1}, -n_{n-1}) basis.
  ///
  /// **Stage B — Intermediate scatterers (reciprocity shortcut):**
  ///   Instead of recomputing each intermediate scatter individually, the reciprocity
  ///   theorem is applied: the reverse-path Jones chain for scatters 2...(n-1) equals
  ///   Q · T^T · Q, where T = J_2 · J_3 · ... · J_{n-1} is the accumulated forward
  ///   Jones matrix (stored in photon.matrix_T), T^T is its transpose (NOT conjugate
  ///   transpose), and Q = diag(1, -1) accounts for the n-component sign flip in
  ///   the reversed propagation direction.
  ///
  /// **Stage C — Last reverse scatter (at position r_1):**
  ///   The reverse path enters from -s_1 and exits along s_n (the forward exit
  ///   direction). The scattering angle is theta_b = acos((-s_1) · s_n). Similar
  ///   rotation and Jones matrix operations produce the final polarization, which
  ///   is rotated into the photon's detection basis (m_n, n_n).
  ///
  /// The result is written to photon.polarization_reverse.
  ///
  /// @param photon The photon (modified: polarization_reverse is set).
  /// @param medium The scattering medium (for scattering matrix evaluation at new angles).
  ///
  /// @pre photon.events >= 2
  /// @pre SimConfig::track_reverse_paths was true during simulation (so that P0, P1,
  ///      Pn1, Pn, r_0, r_n, initial_polarization, and matrix_T are populated).
  void coherent_calculation(Photon &photon, const Medium &medium);

  // ═══════════════════════════════════════════════════════════════════════════
  // StatisticsSensor — Configurable histograms of photon properties
  // ═══════════════════════════════════════════════════════════════════════════

  /// @brief Sensor that accumulates configurable histograms of detected photon properties.
  ///
  /// Supports independent histograms for:
  ///   - **events**: number of scattering events [0, max_events)
  ///   - **theta**: exit polar angle [min_theta, max_theta)
  ///   - **phi**: exit azimuthal angle [min_phi, max_phi)
  ///   - **depth**: maximum penetration depth [0, max_depth)
  ///   - **time**: arrival time [0, max_time)
  ///   - **weight**: statistical weight [0, max_weight)
  ///
  /// Each histogram must be explicitly enabled and configured using the corresponding
  /// set_*_histogram_bins() method before simulation. Histograms that are not configured
  /// remain empty and incur no runtime cost.
  ///
  /// Useful for: transport statistics, validation against analytical models,
  /// scattering order distributions, time-of-flight analysis.
  struct StatisticsSensor : public Sensor
  {
    /// @name Histogram data
    /// @brief Integer bin counts for each configured histogram.
    /// @{
    std::vector<int> events_histogram; ///< Scattering event count distribution.
    std::vector<int> theta_histogram;  ///< Exit polar angle distribution.
    std::vector<int> phi_histogram;    ///< Exit azimuthal angle distribution.
    std::vector<int> depth_histogram;  ///< Penetration depth distribution.
    std::vector<int> time_histogram;   ///< Arrival time distribution.
    std::vector<int> weight_histogram; ///< Statistical weight distribution.
    /// @}

    /// @name Histogram configuration
    /// @details Each histogram has a flag (*_bins_set), range parameters, and a bin width.
    ///          Histograms are only active if their flag is true.
    /// @{
    bool events_histogram_bins_set = false;
    int max_events = 0; ///< Upper bound for events histogram (exclusive).

    bool theta_histogram_bins_set = false;
    double min_theta = 0.0; ///< Lower bound for theta histogram [rad].
    double max_theta = 0.0; ///< Upper bound for theta histogram [rad].
    int n_bins_theta = 0;   ///< Number of theta bins.
    double dtheta = 0.0;    ///< Theta bin width [rad].

    bool phi_histogram_bins_set = false;
    double min_phi = 0.0; ///< Lower bound for phi histogram [rad].
    double max_phi = 0.0; ///< Upper bound for phi histogram [rad].
    int n_bins_phi = 0;   ///< Number of phi bins.
    double dphi = 0.0;    ///< Phi bin width [rad].

    bool depth_histogram_bins_set = false;
    double max_depth = 0.0; ///< Upper bound for depth histogram.
    int n_bins_depth = 0;   ///< Number of depth bins.
    double ddepth = 0.0;    ///< Depth bin width.

    bool time_histogram_bins_set = false;
    double max_time = 0.0; ///< Upper bound for time histogram.
    int n_bins_time = 0;   ///< Number of time bins.
    double dtime = 0.0;    ///< Time bin width.

    bool weight_histogram_bins_set = false;
    double max_weight = 0.0; ///< Upper bound for weight histogram.
    int n_bins_weight = 0;   ///< Number of weight bins.
    double dweight = 0.0;    ///< Weight bin width.
    /// @}

    /// @brief Construct a statistics sensor at the given z-coordinate.
    /// @param z The z-position of the detection plane.
    /// @param absorb If true, photons hitting this sensor are absorbed.
    /// @note No histograms are active by default; call set_*_histogram_bins() to enable them.
    StatisticsSensor(double z, bool absorb = true);

    /// @brief Configure the scattering events histogram.
    /// @param max_events Number of bins (one bin per event count, from 0 to max_events-1).
    void set_events_histogram_bins(int max_events);

    /// @brief Configure the polar angle histogram.
    /// @param min_theta Lower bound [rad].
    /// @param max_theta Upper bound [rad].
    /// @param n_bins    Number of bins.
    void set_theta_histogram_bins(double min_theta, double max_theta, int n_bins);

    /// @brief Configure the azimuthal angle histogram.
    /// @param min_phi Lower bound [rad].
    /// @param max_phi Upper bound [rad].
    /// @param n_bins  Number of bins.
    void set_phi_histogram_bins(double min_phi, double max_phi, int n_bins);

    /// @brief Configure the penetration depth histogram.
    /// @param max_depth Upper bound.
    /// @param n_bins    Number of bins.
    void set_depth_histogram_bins(double max_depth, int n_bins);

    /// @brief Configure the arrival time histogram.
    /// @param max_time Upper bound.
    /// @param n_bins   Number of bins.
    void set_time_histogram_bins(double max_time, int n_bins);

    /// @brief Configure the statistical weight histogram.
    /// @param max_weight Upper bound.
    /// @param n_bins     Number of bins.
    void set_weight_histogram_bins(double max_weight, int n_bins);

    std::unique_ptr<Sensor> clone() const override;
    void merge_from(const Sensor &other) override;

    /// @brief Deposit the photon into each active histogram based on its properties.
    void process_hit(Photon &photon, InteractionInfo &info, const Medium &medium) override;
  };

  // ═══════════════════════════════════════════════════════════════════════════
  // SensorsGroup — Container and dispatcher for multiple sensors
  // ═══════════════════════════════════════════════════════════════════════════

  /// @brief Owns and manages a collection of Sensor instances at potentially different z-planes.
  ///
  /// SensorsGroup is the entry point for all photon-sensor interactions during simulation.
  /// It organizes sensors by their z-coordinate in an ordered map (z_layers) for efficient
  /// intersection testing: only z-planes between the photon's previous and current position
  /// are checked.
  ///
  /// **Hit detection algorithm (record_hit):**
  /// 1. Compute z_min and z_max from photon.prev_pos.z and photon.pos.z.
  /// 2. Look up all z-layers in [z_min, z_max] via the ordered map.
  /// 3. For each crossed z-plane, compute the exact ray-plane intersection point.
  /// 4. For each sensor at that z-plane, check acceptance filters and call process_hit().
  /// 5. If any sensor has absorb_photons=true, the photon is terminated.
  ///
  /// **Estimator dispatch (run_estimators):**
  /// Called after every scattering event. Iterates over all sensors in active_estimators
  /// and calls process_estimation() on each.
  ///
  /// **Parallelism:**
  /// clone() creates a deep copy with empty accumulators. merge_from() combines results
  /// by matching sensors via their id field.
  struct SensorsGroup
  {
    std::vector<std::unique_ptr<Sensor>> detectors;                     ///< Owned sensor instances.
    std::map<double, std::vector<Sensor *>, DoubleComparator> z_layers; ///< Sensors indexed by z-coordinate for fast lookup.

    SensorsGroup() = default;
    SensorsGroup(const SensorsGroup &) = delete;
    SensorsGroup &operator=(const SensorsGroup &) = delete;
    SensorsGroup(SensorsGroup &&) = default;
    SensorsGroup &operator=(SensorsGroup &&) = default;

    /// @brief Add a sensor to this group. Takes ownership of the sensor.
    /// @param detector The sensor to add (moved into internal storage).
    /// @details Assigns a sequential id to the sensor, registers it in z_layers,
    ///          and adds it to active_estimators if estimator_enabled is true.
    void add_detector(std::unique_ptr<Sensor> detector);

    /// @brief Check all z-planes for photon intersection and dispatch to sensors.
    /// @param photon The propagating photon (may be terminated if a sensor absorbs it).
    /// @param medium The scattering medium (passed through to process_hit).
    /// @return true if the photon was absorbed by at least one sensor, false otherwise.
    bool record_hit(Photon &photon, const Medium &medium);

    /// @brief Dispatch estimator contributions to all enabled sensors.
    /// @param photon The photon in its current state after scattering (not modified).
    /// @param medium The scattering medium (passed through to process_estimation).
    void run_estimators(const Photon &photon, const Medium &medium);

    /// @brief Combine accumulated results from another SensorsGroup (after parallel run).
    /// @param other The source group (sensors are matched by id).
    void merge_from(const SensorsGroup &other);

    /// @brief Create a deep copy with identical structure but zeroed accumulators.
    /// @return A new SensorsGroup ready for independent accumulation in a worker thread.
    std::unique_ptr<SensorsGroup> clone() const;
  };

} // namespace luminis::core
