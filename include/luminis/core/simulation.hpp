/**
 * @file simulation.hpp
 * @brief Simulation configuration and transport loop entry points.
 *
 * This header exposes the top-level API for running a Monte Carlo photon
 * transport simulation:
 *
 * - **SimConfig** — aggregates all run parameters (photon count, threads,
 *   RNG seed, pointers to medium, laser, detector group, and optional
 *   time-dependent absorption) into a single configuration struct.
 *
 * - **run_simulation()** — single-threaded driver: emits each photon from
 *   the laser and calls `run_photon()` in sequence.
 *
 * - **run_simulation_parallel()** — multi-threaded driver: splits the photon
 *   budget across worker threads, each with its own cloned detector and RNG,
 *   then merges results back into the shared detector after completion.
 *
 * - **run_photon()** — the core transport loop for a single photon: free-path
 *   sampling, detector intersection, scattering angle sampling, polarization
 *   update, CBS frame-history bookkeeping, weight update, and Russian roulette.
 *
 * ## Parallelism model
 * `run_simulation_parallel()` follows a clone-merge pattern: before launching
 * threads, each sensor and absorption object is cloned so threads operate on
 * independent copies. After all threads join, results are merged back into the
 * original (shared) objects via `merge_from()`. The RNG for each thread is
 * seeded with `mix_seed(config.seed, thread_index)` to guarantee statistically
 * independent streams.
 *
 * ## CBS bookkeeping
 * When `SimConfig::track_reverse_paths` is `true`, `run_photon()` maintains
 * the CBS frame history fields on the `Photon` struct (P0, P1, Pn1, Pn, r_0,
 * r_n, matrix_T). These are consumed by `coherent_calculation()` inside
 * `FarFieldCBSSensor::process_hit()` to compute the reverse-path amplitude.
 *
 * @see photon.hpp for the CBS fields on the Photon struct.
 * @see detector.hpp for the CBS detection algorithm.
 */

#pragma once
#include <luminis/core/absortion.hpp>
#include <cstdint>
#include <luminis/core/simulation.hpp>
#include <luminis/core/medium.hpp>
#include <luminis/core/detector.hpp>
#include <luminis/core/photon.hpp>
#include <luminis/core/laser.hpp>
#include <luminis/math/rng.hpp>
#include <luminis/log/logger.hpp>

namespace luminis::core
{

  // ═══════════════════════════════════════════════════════════════════════════
  // SimConfig
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * @brief Aggregates all parameters required to run a Monte Carlo simulation.
   *
   * Pass a fully populated `SimConfig` to `run_simulation()` or
   * `run_simulation_parallel()`. All pointer members are non-owning; the
   * caller is responsible for the lifetime of the referenced objects.
   *
   * @note Setting `track_reverse_paths = true` enables CBS bookkeeping inside
   *       `run_photon()` at a small per-scattering overhead. Only needed when
   *       a `FarFieldCBSSensor` or `PlanarCBSSensor` is in the sensor group.
   */
  struct SimConfig
  {
    std::uint64_t seed = std::random_device{}(); ///< RNG seed. Defaults to a non-deterministic device seed.
    std::size_t n_threads = 1;                   ///< Number of worker threads for parallel execution.
    std::size_t n_photons;                       ///< Total number of photon packets to simulate.

    bool track_reverse_paths{false}; ///< Enable CBS reverse-path tracking (populates P0/P1/Pn/matrix_T on each Photon).

    Medium *medium{nullptr};                      ///< Scattering medium (optical properties, phase function). Must not be null.
    Laser *laser{nullptr};                        ///< Photon source (position, direction, polarization). Must not be null.
    SensorsGroup *detector{nullptr};              ///< Sensor group collecting detection data. Must not be null.
    AbsorptionTimeDependent *absorption{nullptr}; ///< Optional time-dependent absorption recorder; may be null.

    /**
     * @brief Constructs a SimConfig with an auto-generated RNG seed.
     *
     * @param n  Total number of photons to simulate.
     * @param m  Pointer to the scattering medium.
     * @param l  Pointer to the photon source.
     * @param d  Pointer to the sensor group.
     * @param a  Pointer to the absorption recorder (may be null).
     * @param track_reverse_paths  Enable CBS reverse-path bookkeeping.
     */
    SimConfig(std::size_t n, Medium *m = nullptr, Laser *l = nullptr, SensorsGroup *d = nullptr, AbsorptionTimeDependent *a = nullptr, bool track_reverse_paths = false);

    /**
     * @brief Constructs a SimConfig with an explicit RNG seed for reproducibility.
     *
     * @param s  64-bit RNG seed.
     * @param n  Total number of photons to simulate.
     * @param m  Pointer to the scattering medium.
     * @param l  Pointer to the photon source.
     * @param d  Pointer to the sensor group.
     * @param a  Pointer to the absorption recorder (may be null).
     * @param track_reverse_paths  Enable CBS reverse-path bookkeeping.
     */
    SimConfig(std::uint64_t s, std::size_t n, Medium *m = nullptr, Laser *l = nullptr, SensorsGroup *d = nullptr, AbsorptionTimeDependent *a = nullptr, bool track_reverse_paths = false);
  };

  // ═══════════════════════════════════════════════════════════════════════════
  // Simulation entry points
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * @brief Runs the simulation on a single thread.
   *
   * Emits `config.n_photons` photons from the laser in sequence, propagating
   * each through the medium and recording hits in the sensor group.
   * Suitable for debugging and small-photon-count runs.
   *
   * @param config  Fully populated simulation configuration.
   */
  void run_simulation(const SimConfig &config);

  /**
   * @brief Runs the simulation using multiple threads.
   *
   * Splits `config.n_photons` across `config.n_threads` worker threads
   * (or `hardware_concurrency()` if `n_threads == 0`). Each thread works on
   * an independent clone of the detector and absorption objects. After all
   * threads complete, thread-local results are merged back into the shared
   * objects via `merge_from()`.
   *
   * Thread RNG streams are seeded with `mix_seed(config.seed, thread_index)`
   * to guarantee statistical independence.
   *
   * @param config  Fully populated simulation configuration.
   * @throws Any exception thrown inside a worker thread is re-thrown after
   *         all threads have joined.
   */
  void run_simulation_parallel(const SimConfig &config);

  /**
   * @brief Propagates a single photon through the medium until termination.
   *
   * This is the core Monte Carlo transport kernel. For each step it:
   * 1. Samples a free path from the medium's mean-free-path distribution.
   * 2. Moves the photon and checks for detector plane crossings.
   * 3. Checks medium boundaries; terminates the photon if it escapes.
   * 4. Calls next-event estimators on all active sensors.
   * 5. Samples a scattering angle (θ) and azimuthal angle (φ).
   * 6. Updates the local frame `P_local` via the rotation matrix A.
   * 7. Updates the Jones vector with the normalized scattering matrix S·R.
   * 8. Updates CBS frame history (P0, P1, Pn, r_0, r_n, matrix_T) when
   *    `track_reverse_paths` is true.
   * 9. Applies weight splitting (absorption) and Russian roulette.
   *
   * @param photon              Photon packet to transport (modified in place).
   * @param medium              Scattering medium providing optical properties.
   * @param detector            Sensor group for hit recording and estimators.
   * @param rng                 Per-thread random number generator.
   * @param absorption          Optional time-dependent absorption recorder.
   * @param track_reverse_paths Enable CBS reverse-path bookkeeping.
   */
  void run_photon(Photon &photon, Medium &medium, SensorsGroup &detector, Rng &rng, AbsorptionTimeDependent *absorption, bool track_reverse_paths);
  
} // namespace luminis::core
