/**
 * @file simulation.cpp
 * @brief Implementation of the Monte Carlo photon transport engine.
 *
 * Contains the implementations of:
 * - `run_simulation_parallel()` — multi-threaded driver (clone-merge pattern)
 * - `run_photon()` — per-photon transport kernel
 *
 * The single-thread path currently routes through the same transport kernel
 * by selecting `n_threads = 1` in `SimConfig`.
 *
 * @see simulation.hpp for the full API documentation and design notes.
 */

#include <luminis/core/absortion.hpp>
#include <luminis/core/detector.hpp>
#include <luminis/core/simulation.hpp>
#include <luminis/core/sample.hpp>
#include <luminis/log/logger.hpp>
#include <luminis/math/utils.hpp>
#include <cmath>
#include <thread>
#include <vector>
#include <exception>
#include <atomic>
#include <sstream>
#include <stdexcept>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#elif defined(__APPLE__)
#include <pthread.h>
#include <mach/mach.h>
#include <mach/thread_act.h>
#include <mach/thread_policy.h>
#endif

namespace luminis::core
{
  bool pin_current_thread_to_core(std::size_t core_index, std::string &error_msg)
    {
#if defined(__linux__)
      const unsigned int hw_threads = std::thread::hardware_concurrency();
      const std::size_t target_core = (hw_threads > 0) ? (core_index % hw_threads) : 0;

      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(static_cast<int>(target_core), &cpuset);

      const int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
      if (rc != 0)
      {
        error_msg = "pthread_setaffinity_np failed with error code " + std::to_string(rc);
        return false;
      }
      return true;
#elif defined(__APPLE__)
      // macOS does not expose strict core pinning in public APIs. The
      // affinity tag is a scheduler hint used as a best-effort alternative.
      const integer_t tag = static_cast<integer_t>((core_index % 65535u) + 1u);
      thread_affinity_policy_data_t policy = {tag};
      const thread_port_t mach_thread = pthread_mach_thread_np(pthread_self());
      const kern_return_t kr = thread_policy_set(
          mach_thread,
          THREAD_AFFINITY_POLICY,
          reinterpret_cast<thread_policy_t>(&policy),
          THREAD_AFFINITY_POLICY_COUNT);
      if (kr != KERN_SUCCESS)
      {
        error_msg = "thread_policy_set(THREAD_AFFINITY_POLICY) failed with code " + std::to_string(kr);
        return false;
      }
      return true;
#else
      (void)core_index;
      error_msg = "thread affinity is not supported on this platform";
      return false;
#endif
    }

  // ═══════════════════════════════════════════════════════════════════════════
  // Constants
  // ═══════════════════════════════════════════════════════════════════════════

  using luminis::math::mix_seed;
  using luminis::math::Rng;

  // ═══════════════════════════════════════════════════════════════════════════
  // Multi-threaded driver
  // ═══════════════════════════════════════════════════════════════════════════

  void validate_sim_config(const SimConfig &config)
  {
    if (config.sample == nullptr)
      throw std::invalid_argument("SimConfig::sample must not be null");
    if (config.laser == nullptr)
      throw std::invalid_argument("SimConfig::laser must not be null");
    if (config.detector == nullptr)
      throw std::invalid_argument("SimConfig::detector must not be null");
  }

  void run_simulation_parallel(const SimConfig &config)
  {
    validate_sim_config(config);

    // --- Determine effective thread count ---
    // If n_threads == 0, fall back to the hardware concurrency hint.
    // Never spawn more threads than there are photons.
    int thread_count = std::thread::hardware_concurrency();

    std::size_t n_threads = config.n_threads;
    if (n_threads == 0)
      n_threads = std::thread::hardware_concurrency();
    if (n_threads > config.n_photons)
      n_threads = config.n_photons;

    LLOG_INFO("Configuring simulation for {} threads of {} available hardware threads", n_threads, thread_count);

    // --- Distribute photons across threads ---
    // Remainder photons are assigned one-per-thread to the first `rem` threads
    // so the total is always exactly n_photons.
    const std::size_t base = config.n_photons / n_threads;
    const std::size_t rem = config.n_photons % n_threads;

    LLOG_INFO("Running simulation with {} threads for {} photons", n_threads, config.n_photons);

    // --- Clone per-thread detector and absorption objects ---
    // Each thread must work on its own copy to avoid data races.
    // SensorsGroup::clone() / Absorption::clone() perform deep
    // copies that preserve sensor configuration but start with zero-filled grids.
    std::vector<std::unique_ptr<SensorsGroup>> thread_detectors;
    thread_detectors.reserve(n_threads);
    std::vector<Absorption> thread_absorptions;
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
          if (config.pin_threads_to_cores)
          {
            std::string affinity_error;
            if (!pin_current_thread_to_core(t, affinity_error))
            {
              LLOG_WARN("Thread {} affinity request failed: {}", t, affinity_error);
            }
          }

          // Each thread gets a deterministic but independent seed derived from
          // the global seed and its thread index via mix_seed().
          const std::uint64_t thread_seed = mix_seed(config.seed, static_cast<std::uint64_t>(t));
          Rng rng(thread_seed);

          std::ostringstream oss;
          oss << "Thread " << t << " (id " << std::this_thread::get_id() << ") processing " << my_count << " photons with seed " << thread_seed;
          LLOG_INFO(oss.str());

          SensorsGroup &det = *thread_detectors[t];
          Absorption *abs_ptr = nullptr;
          if (config.absorption)
            abs_ptr = &thread_absorptions[t];

          // Photon velocity is uniform across all layers (shared host medium).
          const double initial_velocity = config.sample->light_speed_in_medium();

          for (std::size_t i = 0; i < my_count; ++i)
          {
            Photon photon = config.laser->emit_photon(rng);
            photon.velocity = initial_velocity;
            photon.current_layer = config.sample->get_layer_index_at(photon.pos.z);
            run_photon(photon, *config.sample, det, rng, abs_ptr, config.track_reverse_paths, config.MAX_EVENTS);

            if (config.progress)
              config.progress->tick();
          }
        }
        catch (...)
        {
          any_error = true;
          thread_exception = std::current_exception();
        } });
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

  void run_photon(Photon &photon, Sample &sample, SensorsGroup &detector, Rng &rng, Absorption *absorption, bool track_reverse_paths, int max_events)
  {
    // --- Initialize CBS frame history ---
    // All CBS fields are reset to the launch state so the photon can be reused
    // across multiple trajectories without carrying stale data.
    photon.r_1 = photon.pos;
    photon.r_n = photon.pos;

    photon.P0 = photon.P_local;
    photon.P1 = photon.P_local;
    photon.Pn2 = photon.P_local;
    photon.Pn1 = photon.P_local;
    photon.Pn = photon.P_local;

    photon.initial_polarization = photon.polarization;

    // matrix_T accumulates the normalized interior Jones matrices J_2..J_{n-1}.
    // Initialized to identity (no scattering history yet).
    photon.matrix_T = CMatrix::identity(2);
    photon.matrix_T_buffer = CMatrix::identity(2);
    photon.has_T_prev = false;

    photon.coherent_path_calculated = false;

    // ─── Main transport loop ─────────────────────────────────────────────────
    while (photon.alive)
    {
      // --- Resolve current layer ---
      const SampleLayer &current_layer = sample.get_layer(photon.current_layer);
      const ScatteringMedium &medium = *current_layer.medium;

      // --- Step 1: Free-path sampling ---
      // Sample a free-path length from the current layer's medium.
      double step = medium.sample_free_path(rng);

      // Compute candidate new position.
      const double dir_x = photon.P_local(2, 0);
      const double dir_y = photon.P_local(2, 1);
      const double dir_z = photon.P_local(2, 2);
      double new_z = photon.pos.z + dir_z * step;

      // --- Step 1b: Interface crossing check ---
      // If the step crosses a layer boundary, truncate at the interface,
      // convert remaining optical distance into the new layer, and repeat
      // until no more interfaces are crossed or the remaining step is consumed.
      {
        auto iface = sample.find_next_interface(photon.pos.z, new_z);
        while (iface.has_value())
        {
          const double z_iface = iface.value();

          // Parametric intersection: fraction of step to reach the interface.
          const double dz_total = dir_z * step;
          if (std::abs(dz_total) < 1e-15)
            break; // Moving parallel to interface, no crossing.

          const double t = (z_iface - photon.pos.z) / dz_total;
          if (t < 0.0 || t > 1.0)
            break; // Numerical guard.

          // Distance traveled to reach the interface.
          const double step_to_iface = step * t;

          // Remaining optical distance (in the current layer's extinction units).
          const double remaining_optical = (step - step_to_iface) * sample.get_layer(photon.current_layer).medium->mu_attenuation;

          // Move photon exactly to the interface.
          photon.opticalpath += step_to_iface;
          photon.prev_pos = photon.pos;
          photon.pos.x += dir_x * step_to_iface;
          photon.pos.y += dir_y * step_to_iface;
          // Place photon slightly past the interface to land in the new layer.
          const double nudge = (dir_z > 0) ? 1e-12 : -1e-12;
          photon.pos.z = z_iface + nudge;

          // Detect sensor crossings for the sub-step to the interface.
          const bool hit = detector.record_hit(photon, sample);
          if (hit)
          {
            photon.alive = false;
            break;
          }

          // Transition to the new layer.
          std::size_t new_layer_idx = sample.get_layer_index_at(photon.pos.z);
          if (new_layer_idx >= sample.size())
          {
            // Photon has left the stack (top or bottom boundary).
            photon.alive = false;
            break;
          }
          photon.current_layer = new_layer_idx;

          // Convert remaining optical distance to physical distance in the new layer.
          const ScatteringMedium &new_medium = *sample.get_layer(photon.current_layer).medium;
          if (new_medium.mu_attenuation > 0)
            step = remaining_optical / new_medium.mu_attenuation;
          else
            step = remaining_optical; // Fallback for non-attenuating layer.

          // Check for further interface crossings in the new layer.
          new_z = photon.pos.z + dir_z * step;
          iface = sample.find_next_interface(photon.pos.z, new_z);
        }
      }

      // If the photon was killed during interface transitions, stop.
      if (!photon.alive)
        break;

      photon.opticalpath += step;
      photon.prev_pos = photon.pos;
      photon.pos.x += dir_x * step;
      photon.pos.y += dir_y * step;
      photon.pos.z += dir_z * step;

      // Update penetration depth (max z reached inside the medium).
      if (photon.pos.z > photon.penetration_depth)
      {
        photon.penetration_depth = photon.pos.z;
      }

      // --- Step 2: Detector intersection ---
      const bool hit = detector.record_hit(photon, sample);
      if (hit)
      {
        photon.alive = false;
        break;
      }

      // --- Step 3: Boundary check ---
      const bool is_inside = sample.is_inside(photon.pos);
      if (!is_inside)
      {
        photon.alive = false;
        break;
      }
      photon.current_layer = sample.get_layer_index_at(photon.pos.z);

      // --- Step 4: Next-event estimation ---
      const ScatteringMedium &scatter_medium = *sample.get_layer(photon.current_layer).medium;
      detector.run_estimators(photon, sample);

      // --- Step 5: Sample scattering angles ---
      const double theta = scatter_medium.sample_scattering_angle(rng);
      CMatrix S_matrix = scatter_medium.scattering_matrix(theta, 0);
      const double phi = scatter_medium.sample_conditional_azimuthal_angle(rng, S_matrix, photon.polarization, theta);

      // --- Step 6: Update local frame (P_local) ---
      // Build the 3×3 rotation matrix A that transforms the current local
      // frame into the new frame after scattering by (θ, φ):
      //   new_P = A * old_P
      // Row layout: (m', n', s') = A * (m, n, s).
      const double cos_theta = std::cos(theta);
      const double sin_theta = std::sin(theta);
      const double cos_phi = std::cos(phi);
      const double sin_phi = std::sin(phi);

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
        R(0, 0) = cos_phi;
        R(0, 1) = sin_phi;
        R(1, 0) = -sin_phi;
        R(1, 1) = cos_phi;
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
          T_current_raw(0, 0) = S_matrix(0, 0) * cos_phi;
          T_current_raw(0, 1) = S_matrix(0, 0) * sin_phi;
          T_current_raw(1, 0) = -S_matrix(1, 1) * sin_phi;
          T_current_raw(1, 1) = S_matrix(1, 1) * cos_phi;

          // Event index of the scatter just executed (1-based).
          const uint evt = photon.events + 1;

          // --- Update spatial positions and frame snapshots ---
          if (evt == 1)
          {
            // First scattering event: record the first-scatter position and frame.
            photon.r_1 = photon.pos;
            photon.first_scatter_layer = photon.current_layer;
            photon.P1 = photon.P_local;
          }

          photon.r_n = photon.pos; // Always update to the most recent scatter site.
          photon.last_scatter_layer = photon.current_layer;
          photon.Pn2 = photon.Pn1;
          photon.Pn1 = photon.Pn;
          photon.Pn = photon.P_local;

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
              photon.matrix_T_buffer = T_current;
              photon.matrix_T_raw_buffer = T_current_raw;
              photon.has_T_prev = true;
            }
            else
            {
              // Third event and beyond: the buffered event is confirmed interior,
              // so prepend it to matrix_T (T_mid = J_prev * T_mid).
              CMatrix tmp(2, 2);
              matcmul(photon.matrix_T_buffer, photon.matrix_T, tmp);
              photon.matrix_T = std::move(tmp);
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
              photon.matrix_T_raw = std::move(tmp_raw);
              photon.matrix_T_raw_buffer = T_current_raw;
            }
          }
        }
      }

      // --- Step 9: Weight update (absorption splitting + Russian roulette) ---
      // Implicit absorption: fraction mu_a/mu_t is deposited, surviving weight
      // is rescaled by mu_s/mu_t so the photon continues with reduced weight.
      // Uses the medium of the layer where scattering occurred.
      photon.events++;
      double d_weight = 1.0;
      if (scatter_medium.mu_attenuation > 0)
      { 
        const double absorption_prob = scatter_medium.mu_absorption / scatter_medium.mu_attenuation;
        const double scattering_prob = scatter_medium.mu_scattering / scatter_medium.mu_attenuation;
        d_weight = photon.weight * absorption_prob;
        photon.weight *= scattering_prob;
      }

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

      if (photon.events > max_events)
      {
        photon.alive = false;
        break;
      }
    }

    LLOG_DEBUG("Photon terminated after {} events, final weight: {}, optical path: {}", photon.events, photon.weight, photon.opticalpath);
  }

} // namespace luminis::core