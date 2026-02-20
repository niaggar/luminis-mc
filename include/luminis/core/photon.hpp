/**
 * @file photon.hpp
 * @brief Photon state and photon record data structures for Monte Carlo transport.
 *
 * This header defines the two central data types used throughout the simulation:
 *
 * - **Photon**: the live simulation packet carrying all physical state needed
 *   during transport — position, polarization (Jones vector), scattering-frame
 *   history, accumulated Jones matrix, and CBS bookkeeping.
 *
 * - **PhotonRecord**: a lightweight, immutable snapshot of a detected photon
 *   written by `PhotonRecordSensor`. It captures just enough information for
 *   post-processing (position, direction, polarization, path length, etc.)
 *   without carrying the full simulation state.
 *
 * ## Polarization convention
 * Polarization state is stored as a Jones vector `CVec2{m, n}` in the local
 * right-handed frame `(m, n, s)`, where `s` is the propagation direction.
 * The frame is represented by a 3×3 rotation matrix `P_local` whose rows are
 * `m`, `n`, and `s` respectively. This frame is updated at every scattering
 * event.
 *
 * ## CBS frame history
 * For coherent backscattering (CBS), the simulation must retain the local
 * frames at the first and last two scattering vertices (P0, P1, Pn1, Pn),
 * the positions of those vertices (r_0, r_n), and the accumulated Jones
 * transfer matrix T. These fields are updated during transport only when
 * `SimConfig::track_reverse_paths` is enabled.
 *
 * @see detector.hpp for the CBS algorithm that consumes the frame history.
 * @see simulation.hpp for the transport loop that populates Photon fields.
 */

#pragma once
#include <complex>
#include <luminis/math/vec.hpp>
#include <sys/types.h>
#include <array>

using namespace luminis::math;

namespace luminis::core
{

  // ═══════════════════════════════════════════════════════════════════════════
  // Photon — live transport packet
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * @brief Represents a single photon packet during Monte Carlo transport.
   *
   * A `Photon` carries all the physical state required to propagate and scatter
   * a photon through a turbid medium, including position, polarization, optical
   * path length, weight, and the scattering-frame history needed for CBS.
   *
   * During a simulation run, a single `Photon` instance is typically reused
   * across many independent photon trajectories. Each trajectory starts with
   * launcher initialization and ends when the photon is absorbed, exits the
   * medium, or is killed by a sensor.
   *
   * @note Thread safety: each simulation thread owns its own `Photon` instance;
   *       concurrent access to a single `Photon` is not supported.
   */
  struct Photon
  {
    // ─── Spatial state ───────────────────────────────────────────────────────

    Vec3 prev_pos{0, 0, 0};     ///< Position at the start of the current step (before move).
    Vec3 pos{0, 0, 0};          ///< Current position.
    Vec3 detected_pos{0, 0, 0}; ///< Position at the last detection plane crossing.

    // ─── Transport scalars ───────────────────────────────────────────────────

    uint events{0};               ///< Number of scattering events accumulated so far.
    double penetration_depth{0.0};///< Maximum depth reached inside the scattering medium [mm].
    bool alive{true};             ///< False when the photon has been terminated (absorbed or exited).
    double wavelength_nm{0.0};    ///< Free-space wavelength [nm].
    double k{0.0};                ///< Wave number in free space: k = 2π / λ [rad/mm].
    double opticalpath{0.0};      ///< Accumulated optical path length [mm].
    double launch_time{0.0};      ///< Emission time [ns]; used for time-gated detection.
    double velocity{1.0};         ///< Phase velocity in the medium [mm/ns] (c/n).
    double weight{1.0};           ///< Statistical weight for variance reduction (Russian roulette / absorption).

    // ─── Polarization state ──────────────────────────────────────────────────

    /**
     * @brief Local scattering-frame basis matrix (3×3).
     *
     * Rows encode the right-handed orthonormal basis at the current propagation
     * direction:
     *   - Row 0: m-axis (tangential, "horizontal")
     *   - Row 1: n-axis (tangential, "vertical")
     *   - Row 2: s-axis (propagation direction)
     *
     * The basis is updated at each scattering event by rotating the frame into
     * the new propagation direction. Sensors use this matrix to project the
     * Jones vector into the laboratory (x,y) frame.
     */
    Matrix P_local = Matrix(3, 3);

    bool polarized{true}; ///< When false, polarization state is ignored and incoherent transport is used.

    /**
     * @brief Forward-path Jones vector in the local frame (m, n).
     *
     * Carries the complex electric field amplitudes along the m and n axes.
     * Updated at each scattering event by the medium's amplitude scattering
     * matrix and the rotation between successive local frames.
     */
    CVec2 polarization{std::complex<double>(1, 0), std::complex<double>(0, 0)};

    /**
     * @brief Reverse-path Jones vector, populated by `coherent_calculation()`.
     *
     * This field is written just before the photon reaches a CBS detector.
     * It holds the Jones vector of the time-reversed path, computed via the
     * three-stage algorithm (stages A, B, C) using the stored frame history
     * and the accumulated Jones transfer matrix T.
     *
     * @see coherent_calculation() in detector.cpp
     */
    CVec2 polarization_reverse{std::complex<double>(1, 0), std::complex<double>(0, 0)};

    // ─── CBS frame history ───────────────────────────────────────────────────

    bool coherent_path_calculated{false}; ///< Flag: set to true once the reverse path has been computed for this step.

    /**
     * @brief Jones vector of the photon as launched (before any scattering).
     *
     * Stored at launch and used as the initial input for the reverse-path
     * computation. Along with `P0`, it defines the state at the entry point
     * of the scattering sequence.
     */
    CVec2 initial_polarization{std::complex<double>(1, 0), std::complex<double>(0, 0)};

    /**
     * @name CBS scattering-frame snapshots
     * Local frame matrices captured at key scattering vertices to enable the
     * three-stage reverse-path computation. Rows of each matrix are (m, n, s).
     * @{
     */
    Matrix P0  = Matrix(3, 3); ///< Frame at the incidence direction (before first scatter, i.e. the launch frame).
    Matrix P1  = Matrix(3, 3); ///< Frame immediately after the first scattering event.
    Matrix Pn2 = Matrix(3, 3); ///< Frame after the second-to-last (n-2) scattering event.
    Matrix Pn1 = Matrix(3, 3); ///< Frame after the penultimate (n−1) scattering event.
    Matrix Pn  = Matrix(3, 3); ///< Frame after the last (nth) scattering event (exit frame).
    /** @} */

    /**
     * @name CBS scattering-vertex positions
     * @{
     */
    Vec3 r_0{0, 0, 0}; ///< Position of the first scattering event.
    Vec3 r_n{0, 0, 0}; ///< Position of the last scattering event.
    /** @} */

    /**
     * @name Jones transfer matrix (T) and double-buffered updates
     *
     * `matrix_T` accumulates the product of all scattering amplitude matrices
     * along the forward path, expressed in the frame-to-frame basis. It is
     * used by `coherent_calculation()` as the middle-segment operator via
     * Q·T^T·Q (reciprocity shortcut).
     *
     * A double-buffer scheme (active + `_buffer`) avoids overwriting the
     * committed matrix while the current step is being processed.
     * @{
     */
    CMatrix matrix_T        = CMatrix::identity(2); ///< Active Jones transfer matrix for the full path.
    CMatrix matrix_T_buffer = CMatrix::identity(2); ///< Staging buffer; committed to `matrix_T` after each scatter.

    CMatrix matrix_T_raw        = CMatrix::identity(2); ///< Raw accumulated product (without the normalization F factor).
    CMatrix matrix_T_raw_buffer = CMatrix::identity(2); ///< Staging buffer for `matrix_T_raw`.
    /** @} */

    bool has_T_prev{false}; ///< True once at least one scattering event has been committed to `matrix_T`.

    // ─── Construction & methods ──────────────────────────────────────────────

    Photon() = default;

    /**
     * @brief Constructs a photon with a given position, direction, and wavelength.
     *
     * Initializes `pos`, `prev_pos`, `P_local`, wave number `k`, and sets the
     * local frame rows to (m, n, d). The polarization is left at its default
     * (x-linear: `{1, 0}`).
     *
     * @param p  Initial position [mm].
     * @param d  Propagation direction unit vector.
     * @param m  Transverse basis vector m (must be perpendicular to d).
     * @param n  Transverse basis vector n (must be perpendicular to d and m).
     * @param wl Free-space wavelength [nm].
     */
    Photon(const Vec3 &p, const Vec3 &d, const Vec3 &m, const Vec3 &n, const double wl);

    /**
     * @brief Overrides the current Jones vector with an arbitrary polarization state.
     *
     * Sets `polarized = true` and copies the given Jones vector into
     * `polarization`. The local frame (`P_local`) is unchanged.
     *
     * @param pol New Jones vector in the current local frame.
     */
    void set_polarization(CVec2 pol);

    /**
     * @brief Computes the Stokes parameters from the current Jones vector.
     *
     * Returns `{S0, S1, S2, S3}` computed as:
     * - S0 = |Em|² + |En|²   (total intensity)
     * - S1 = |Em|² − |En|²   (linear polarization along m/n axes)
     * - S2 = 2 Re(Em·En*)     (linear polarization at ±45°)
     * - S3 = −2 Im(Em·En*)    (circular polarization; RHC positive)
     *
     * If `polarized` is false, returns `{1, 0, 0, 0}` (unpolarized).
     *
     * @return Array of four Stokes parameters `[S0, S1, S2, S3]`.
     */
    std::array<double, 4> get_stokes_parameters() const;
  };

  // ═══════════════════════════════════════════════════════════════════════════
  // PhotonRecord — immutable detection snapshot
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * @brief Immutable record of a detected photon, written by PhotonRecordSensor.
   *
   * When a photon passes through a `PhotonRecordSensor`, the key physical
   * quantities are copied into a `PhotonRecord` and appended to the sensor's
   * output list. These records are later exposed to Python for analysis.
   *
   * Unlike `Photon`, this struct carries no mutable simulation state; it is
   * intended purely as a data-transfer object for post-processing.
   */
  struct PhotonRecord
  {
    // ─── Transport scalars ───────────────────────────────────────────────────

    uint events{0};               ///< Number of scattering events before detection.
    double penetration_depth{0.0};///< Maximum penetration depth inside the medium [mm].
    double launch_time{0.0};      ///< Emission time of the photon [ns].
    double arrival_time{0.0};     ///< Time of arrival at the detector: launch_time + opticalpath/velocity [ns].
    double opticalpath{0.0};      ///< Total optical path length at detection [mm].
    double weight{0.0};           ///< Statistical weight at detection.
    double k{0.0};                ///< Wave number k = 2π/λ [rad/mm].

    // ─── Spatial information ─────────────────────────────────────────────────

    Vec3 position_first_scattering{0.0, 0.0, 0.0}; ///< Position of the first scattering event (r_0).
    Vec3 position_last_scattering{0.0, 0.0, 0.0};  ///< Position of the last scattering event (r_n).
    Vec3 position_detector{0.0, 0.0, 0.0};          ///< Intersection point on the detector plane.

    // ─── Directional frame at detection ──────────────────────────────────────

    Vec3 direction{0.0, 0.0, 1.0}; ///< Propagation direction unit vector at the detector.
    Vec3 m{1.0, 0.0, 0.0};         ///< m-axis of the local frame at detection.
    Vec3 n{0.0, 1.0, 0.0};         ///< n-axis of the local frame at detection.

    // ─── Polarization snapshots ───────────────────────────────────────────────

    /// Forward-path Jones vector at detection (in the local frame).
    CVec2 polarization_forward{std::complex<double>(1, 0), std::complex<double>(0, 0)};

    /// Reverse-path Jones vector at detection (populated only when CBS tracking is enabled).
    CVec2 polarization_reverse{std::complex<double>(0, 0), std::complex<double>(0, 0)};
  };

} // namespace luminis::core
