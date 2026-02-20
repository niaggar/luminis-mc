/**
 * @file simulation.cpp
 * @brief Implementation of the Monte Carlo photon transport engine.
 *
 * Contains the implementations of:
 * - `SimConfig` constructors
 * - `run_simulation()` — single-threaded driver
 * - `run_simulation_parallel()` — multi-threaded driver (clone-merge pattern)
 * - `run_photon()` — per-photon transport kernel
 *
 * @see simulation.hpp for the full API documentation and design notes.
 */

#include <luminis/core/absortion.hpp>
#include <luminis/core/detector.hpp>
#include <luminis/core/simulation.hpp>
#include <luminis/log/logger.hpp>
#include <luminis/math/utils.hpp>
#include <cmath>
#include <thread>
#include <vector>
#include <exception>
#include <atomic>
#include <sstream>

namespace luminis::core
{
  // ═══════════════════════════════════════════════════════════════════════════
  // Constants
  // ═══════════════════════════════════════════════════════════════════════════

  /// Maximum scattering events per photon before forced termination.
  /// Guards against infinite loops in highly scattering media.
  const int MAX_EVENTS = 1000;

  using luminis::math::mix_seed;
  using luminis::math::Rng;

  // ═══════════════════════════════════════════════════════════════════════════
  // SimConfig implementation
  // ═══════════════════════════════════════════════════════════════════════════

  SimConfig::SimConfig(std::size_t n, Medium *m, Laser *l, SensorsGroup *d, AbsorptionTimeDependent *a, bool track_reverse_paths)
      : n_photons(n), medium(m), laser(l), detector(d), absorption(a), track_reverse_paths(track_reverse_paths) {}

  SimConfig::SimConfig(std::uint64_t s, std::size_t n, Medium *m, Laser *l, SensorsGroup *d, AbsorptionTimeDependent *a, bool track_reverse_paths)
      : seed(s), n_photons(n), medium(m), laser(l), detector(d), absorption(a), track_reverse_paths(track_reverse_paths) {}

  // ═══════════════════════════════════════════════════════════════════════════
  // Single-threaded driver
  // ═══════════════════════════════════════════════════════════════════════════

  void run_simulation(const SimConfig &config)
  {
    Rng rng(config.seed);

    for (std::size_t i = 0; i < config.n_photons; ++i)
    {
      Photon photon = config.laser->emit_photon(rng);
      photon.velocity = config.medium->light_speed_in_medium();
      run_photon(photon, *config.medium, *config.detector, rng, config.absorption, config.track_reverse_paths);
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Multi-threaded driver
  // ═══════════════════════════════════════════════════════════════════════════

  void run_simulation_parallel(const SimConfig &config)
  {
    // --- Determine effective thread count ---
    // If n_threads == 0, fall back to the hardware concurrency hint.
    // Never spawn more threads than there are photons.
    std::size_t n_threads = config.n_threads;
    if (n_threads == 0)
      n_threads = std::thread::hardware_concurrency();
    if (n_threads > config.n_photons)
      n_threads = config.n_photons;

    // --- Distribute photons across threads ---
    // Remainder photons are assigned one-per-thread to the first `rem` threads
    // so the total is always exactly n_photons.
    const std::size_t base = config.n_photons / n_threads;
    const std::size_t rem  = config.n_photons % n_threads;

    LLOG_INFO("Running simulation with {} threads for {} photons", n_threads, config.n_photons);

    // --- Clone per-thread detector and absorption objects ---
    // Each thread must work on its own copy to avoid data races.
    // SensorsGroup::clone() / AbsorptionTimeDependent::clone() perform deep
    // copies that preserve sensor configuration but start with zero-filled grids.
    std::vector<std::unique_ptr<SensorsGroup>> thread_detectors;
    thread_detectors.reserve(n_threads);
    std::vector<AbsorptionTimeDependent> thread_absorptions;
    if (config.absorption)
      thread_absorptions.reserve(n_threads);

    for (std::size_t t = 0; t < n_threads; ++t)
    {
      thread_detectors.emplace_back(config.detector->clone());

      if (config.absorption)
        thread_absorptions.emplace_back(config.absorption->clone());
    }

    // --- Launch worker threads ---
    std::vector<std::thread> workers;
    workers.reserve(n_threads);

    std::atomic<bool> any_error{false};
    std::exception_ptr thread_exception = nullptr;

    for (std::size_t t = 0; t < n_threads; ++t)
    {
      const std::size_t my_count = base + (t < rem ? 1u : 0u);

      workers.emplace_back([&, t, my_count]()
      {
        try
        {
          // Each thread gets a deterministic but independent seed derived from
          // the global seed and its thread index via mix_seed().
          const std::uint64_t thread_seed = mix_seed(config.seed, static_cast<std::uint64_t>(t));
          Rng rng(thread_seed);

          std::ostringstream oss;
          oss << "Thread " << t << " (id " << std::this_thread::get_id() << ") processing " << my_count << " photons with seed " << thread_seed;
          LLOG_INFO(oss.str());

          SensorsGroup &det = *thread_detectors[t];
          AbsorptionTimeDependent *abs_ptr = nullptr;
          if (config.absorption)
            abs_ptr = &thread_absorptions[t];

          for (std::size_t i = 0; i < my_count; ++i)
          {
            Photon photon = config.laser->emit_photon(rng);
            photon.velocity = config.medium->light_speed_in_medium();
            run_photon(photon, *config.medium, det, rng, abs_ptr, config.track_reverse_paths);
          }
        }
        catch (...)
        {
          any_error = true;
          thread_exception = std::current_exception();
        }
      });
    }

    LLOG_INFO("All threads launched.");

    // --- Join and propagate any thread exception ---
    for (auto &th : workers)
      th.join();
    if (any_error && thread_exception)
    {
      std::rethrow_exception(thread_exception);
    }

    // --- Merge thread-local results into the shared detector ---
    // merge_from() accumulates each sensor's grids/histograms additively.
    for (std::size_t t = 0; t < n_threads; ++t)
    {
      config.detector->merge_from(*thread_detectors[t]);
    }
    if (config.absorption)
    {
      for (std::size_t t = 0; t < n_threads; ++t)
      {
        config.absorption->merge_from(thread_absorptions[t]);
      }
    }

    LLOG_INFO("Parallel simulation finished");
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Per-photon transport kernel
  // ═══════════════════════════════════════════════════════════════════════════

  void run_photon(Photon &photon, Medium &medium, SensorsGroup &detector, Rng &rng, AbsorptionTimeDependent *absorption, bool track_reverse_paths)
  {
    // --- Initialize CBS frame history ---
    // All CBS fields are reset to the launch state so the photon can be reused
    // across multiple trajectories without carrying stale data.
    photon.r_0 = photon.pos;
    photon.r_n = photon.pos;

    photon.P0  = photon.P_local;
    photon.P1  = photon.P_local;
    photon.Pn2 = photon.P_local;
    photon.Pn1 = photon.P_local;
    photon.Pn  = photon.P_local;

    photon.initial_polarization = photon.polarization;

    // matrix_T accumulates the normalized interior Jones matrices J_2..J_{n-1}.
    // Initialized to identity (no scattering history yet).
    photon.matrix_T        = CMatrix::identity(2);
    photon.matrix_T_buffer = CMatrix::identity(2);
    photon.has_T_prev      = false;

    photon.coherent_path_calculated = false;

    // ─── Main transport loop ─────────────────────────────────────────────────
    while (photon.alive)
    {
      // --- Step 1: Free-path sampling ---
      // Advance the photon by one mean-free-path sampled from the medium.
      const double step = medium.sample_free_path(rng);
      photon.opticalpath += step;
      photon.prev_pos = photon.pos;
      photon.pos.x += photon.P_local(2, 0) * step;
      photon.pos.y += photon.P_local(2, 1) * step;
      photon.pos.z += photon.P_local(2, 2) * step;

      // --- Step 2: Detector intersection ---
      // Check if the step crossed any sensor plane. If a sensor absorbs the
      // photon (absorb_photons == true), terminate immediately.
      const bool hit = detector.record_hit(photon, medium);
      if (hit)
      {
        photon.alive = false;
        break;
      }

      // --- Step 3: Boundary check ---
      // Terminate if the photon has left the medium volume.
      // TODO: Implement boundary interactions and multiple media.
      const bool is_inside = medium.is_inside(photon.pos);
      if (!is_inside)
      {
        photon.alive = false;
        break;
      }

      // --- Step 4: Next-event estimation ---
      // Call all estimators (forced-detection variance reduction) at the
      // current scattering position before sampling the new direction.
      detector.run_estimators(photon, medium);

      // --- Step 5: Sample scattering angles ---
      const double theta = medium.sample_scattering_angle(rng);
      CMatrix S_matrix = medium.scattering_matrix(theta, 0, photon.k);
      const double phi = medium.sample_conditional_azimuthal_angle(rng, S_matrix, photon.polarization, photon.k, theta);

      // --- Step 6: Update local frame (P_local) ---
      // Build the 3×3 rotation matrix A that transforms the current local
      // frame into the new frame after scattering by (θ, φ):
      //   new_P = A * old_P
      // Row layout: (m', n', s') = A * (m, n, s).
      const double cos_theta = std::cos(theta);
      const double sin_theta = std::sin(theta);
      const double cos_phi   = std::cos(phi);
      const double sin_phi   = std::sin(phi);

      Matrix A_update = Matrix(3, 3);
      A_update(0, 0) = cos_theta * cos_phi;
      A_update(0, 1) = cos_theta * sin_phi;
      A_update(0, 2) = -sin_theta;
      A_update(1, 0) = -sin_phi;
      A_update(1, 1) = cos_phi;
      A_update(1, 2) = 0;
      A_update(2, 0) = sin_theta * cos_phi;
      A_update(2, 1) = sin_theta * sin_phi;
      A_update(2, 2) = cos_theta;

      matmul(A_update, photon.P_local, photon.P_local);

      // --- Step 7: Update Jones vector (polarization) ---
      if (photon.polarized)
      {
        // The combined scattering operator for this event is T_current = S * R,
        // where R is the in-plane rotation by φ and S is the amplitude
        // scattering matrix. Both are 2×2 in the local (m, n) basis.
        CMatrix R(2, 2);
        R(0, 0) =  cos_phi;  R(0, 1) = sin_phi;
        R(1, 0) = -sin_phi;  R(1, 1) = cos_phi;
        CMatrix T_current = CMatrix(2, 2);
        matcmul(S_matrix, R, T_current);

        // --- Normalization factor F ---
        // F is the expected intensity after scattering, averaged over all
        // outgoing polarization components weighted by the input Jones vector.
        // Dividing T_current by √F ensures that the scattered Jones vector
        // has unit norm (energy is tracked via photon.weight instead).
        const std::complex<double> Em = photon.polarization.m;
        const std::complex<double> En = photon.polarization.n;

        const double Emm = std::norm(Em);
        const double Enn = std::norm(En);
        const double s22 = std::norm(S_matrix(0, 0));
        const double s11 = std::norm(S_matrix(1, 1));

        const double pow_cos_phi = std::pow(cos_phi, 2);
        const double pow_sin_phi = std::pow(sin_phi, 2);

        const double F =
            Emm * (s22 * pow_cos_phi + s11 * pow_sin_phi) +
            Enn * (s22 * pow_sin_phi + s11 * pow_cos_phi) +
            2.0 * std::real(Em * std::conj(En)) * (s22 - s11) * sin_phi * cos_phi;

        // Apply normalized scattering operator to the Jones vector.
        const double F_inv_sqrt = 1.0 / std::sqrt(F);
        matcmulscalar(F_inv_sqrt, T_current);
        photon.polarization.m = (T_current(0, 0) * Em + T_current(0, 1) * En);
        photon.polarization.n = (T_current(1, 0) * Em + T_current(1, 1) * En);

        // --- Step 8: CBS frame-history update ---
        if (track_reverse_paths)
        {
          // Raw (un-normalized) Jones matrix for this scattering event.
          // Used in matrix_T_raw which tracks the unnormalized path product.
          CMatrix T_current_raw(2, 2);
          T_current_raw(0, 0) =  S_matrix(0, 0) * cos_phi;
          T_current_raw(0, 1) =  S_matrix(0, 0) * sin_phi;
          T_current_raw(1, 0) = -S_matrix(1, 1) * sin_phi;
          T_current_raw(1, 1) =  S_matrix(1, 1) * cos_phi;

          // Event index of the scatter just executed (1-based).
          const uint evt = photon.events + 1;

          // --- Update spatial positions and frame snapshots ---
          if (evt == 1)
          {
            // First scattering event: record the first-scatter position and frame.
            photon.r_0 = photon.pos;
            photon.P1  = photon.P_local;
          }

          photon.r_n  = photon.pos;   // Always update to the most recent scatter site.
          photon.Pn2  = photon.Pn1;
          photon.Pn1  = photon.Pn;
          photon.Pn   = photon.P_local;

          // --- Double-buffer T-matrix update ---
          // matrix_T accumulates J_2 · J_3 · … · J_{n-1} (the interior events).
          // Because we don't know which event is "last" until detection, we
          // use a one-event lookahead buffer:
          //   - matrix_T_buffer always holds the most recent J (candidate "last")
          //   - When the next event arrives, the buffered J is confirmed as NOT
          //     last and is committed into matrix_T.
          if (evt >= 2)
          {
            if (!photon.has_T_prev)
            {
              // Second event: store it as the current "last" candidate.
              photon.matrix_T_buffer     = T_current;
              photon.matrix_T_raw_buffer = T_current_raw;
              photon.has_T_prev          = true;
            }
            else
            {
              // Third event and beyond: the buffered event is confirmed interior,
              // so prepend it to matrix_T (T_mid = J_prev * T_mid).
              CMatrix tmp(2, 2);
              matcmul(photon.matrix_T_buffer, photon.matrix_T, tmp);
              photon.matrix_T        = std::move(tmp);
              photon.matrix_T_buffer = T_current;

              // Update the raw matrix with Frobenius normalization to prevent
              // numerical overflow on very long paths.
              CMatrix tmp_raw(2, 2);
              matcmul(photon.matrix_T_raw_buffer, photon.matrix_T_raw, tmp_raw);
              double frob = std::sqrt(
                  std::norm(tmp_raw(0, 0)) + std::norm(tmp_raw(0, 1)) +
                  std::norm(tmp_raw(1, 0)) + std::norm(tmp_raw(1, 1)));
              if (frob > 1e-300)
              {
                tmp_raw(0, 0) /= frob;
                tmp_raw(0, 1) /= frob;
                tmp_raw(1, 0) /= frob;
                tmp_raw(1, 1) /= frob;
              }
              photon.matrix_T_raw        = std::move(tmp_raw);
              photon.matrix_T_raw_buffer = T_current_raw;
            }
          }
        }
      }

      // --- Step 9: Weight update (absorption splitting + Russian roulette) ---
      // Implicit absorption: fraction mu_a/mu_t is deposited, surviving weight
      // is rescaled by mu_s/mu_t so the photon continues with reduced weight.
      const double d_weight = photon.weight * (medium.mu_absorption / medium.mu_attenuation);
      photon.weight          = photon.weight * (medium.mu_scattering  / medium.mu_attenuation);
      photon.events++;

      if (absorption)
      {
        absorption->record_absorption(photon, d_weight);
      }

      // Russian roulette: photons with very low weight are either boosted
      // (probability 0.1) or terminated to avoid endless low-weight wandering.
      if (photon.weight < 1e-4)
      {
        if (rng.uniform() < 0.1)
        {
          photon.weight /= 0.1;
        }
        else
        {
          photon.alive = false;
          break;
        }
      }

      if (photon.events > MAX_EVENTS)
      {
        photon.alive = false;
        break;
      }
    }

    LLOG_DEBUG("Photon terminated after {} events, final weight: {}, optical path: {}", photon.events, photon.weight, photon.opticalpath);
  }

} // namespace luminis::core