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
  const int MAX_EVENTS = 1000;

  using luminis::math::mix_seed;
  using luminis::math::Rng;

  SimConfig::SimConfig(std::size_t n, Medium *m, Laser *l, SensorsGroup *d, AbsorptionTimeDependent *a, bool track_reverse_paths)
      : n_photons(n), medium(m), laser(l), detector(d), absorption(a), track_reverse_paths(track_reverse_paths) {}

  SimConfig::SimConfig(std::uint64_t s, std::size_t n, Medium *m, Laser *l, SensorsGroup *d, AbsorptionTimeDependent *a, bool track_reverse_paths)
      : seed(s), n_photons(n), medium(m), laser(l), detector(d), absorption(a), track_reverse_paths(track_reverse_paths) {}

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

  void run_simulation_parallel(const SimConfig &config)
  {
    // Determine number of threads to use
    std::size_t n_threads = config.n_threads;
    if (n_threads == 0)
      n_threads = std::thread::hardware_concurrency();
    if (n_threads > config.n_photons)
      n_threads = config.n_photons;

    const std::size_t base = config.n_photons / n_threads;
    const std::size_t rem = config.n_photons % n_threads;

    LLOG_INFO("Running simulation with {} threads for {} photons", n_threads, config.n_photons);

    // Create thread-local detectors and absorptions
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

    // Launch threads
    std::vector<std::thread> workers;
    workers.reserve(n_threads);

    std::atomic<bool> any_error{false};
    std::exception_ptr thread_exception = nullptr;

    for (std::size_t t = 0; t < n_threads; ++t)
    {
      const std::size_t my_count = base + (t < rem ? 1u : 0u);

      workers.emplace_back([&, t, my_count]()
                           {
      try {
        const std::uint64_t thread_seed = mix_seed(config.seed, static_cast<std::uint64_t>(t));
        Rng rng(thread_seed);

        std::ostringstream oss;
        oss << "Thread " << t << " (id " << std::this_thread::get_id() << ") processing " << my_count << " photons with seed " << thread_seed;
        LLOG_INFO(oss.str());

        SensorsGroup& det = *thread_detectors[t];
        AbsorptionTimeDependent* abs_ptr = nullptr;
        if (config.absorption) abs_ptr = &thread_absorptions[t];

        for (std::size_t i = 0; i < my_count; ++i) {
          Photon photon = config.laser->emit_photon(rng);
          photon.velocity = config.medium->light_speed_in_medium();
          run_photon(photon, *config.medium, det, rng, abs_ptr, config.track_reverse_paths);
        }

      } catch (...) {
        any_error = true;
        thread_exception = std::current_exception();
      } });
    }

    LLOG_INFO("All threads launched.");

    // Join threads
    for (auto &th : workers)
      th.join();
    if (any_error && thread_exception)
    {
      std::rethrow_exception(thread_exception);
    }

    // Merge thread-local detectors and absorptions
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

  void run_photon(Photon &photon, Medium &medium, SensorsGroup &detector, Rng &rng, AbsorptionTimeDependent *absorption, bool track_reverse_paths)
  {
    const uint first_event = 0;

    // Update incident direction for CBS
    photon.r_0 = photon.pos;
    photon.r_n = photon.pos;

    photon.P0 = photon.P_local;
    photon.P1 = photon.P_local;
    photon.Pn2 = photon.P_local;
    photon.Pn1 = photon.P_local;
    photon.Pn = photon.P_local;

    photon.initial_polarization = photon.polarization;

    photon.matrix_T = CMatrix::identity(2);        // T_mid = I
    photon.matrix_T_buffer = CMatrix::identity(2); // J_prev dummy
    photon.has_T_prev = false;

    photon.coherent_path_calculated = false;

    // Main photon propagation loop
    while (photon.alive)
    {
      // Sample free step
      const double step = medium.sample_free_path(rng);
      photon.opticalpath += step;
      photon.prev_pos = photon.pos;
      photon.pos.x += photon.P_local(2, 0) * step;
      photon.pos.y += photon.P_local(2, 1) * step;
      photon.pos.z += photon.P_local(2, 2) * step;

      // Check for detector hit
      const bool hit = detector.record_hit(photon, medium);
      if (hit)
      {
        photon.alive = false;
        break;
      }

      // Check if photon is still inside the medium
      // TODO: Implement boundary interactions and multiple media
      const bool is_inside = medium.is_inside(photon.pos);
      if (!is_inside)
      {
        photon.alive = false;
        break;
      }

      // Run estimators for current photon state
      detector.run_estimators(photon, medium);

      // Sample scattering angle
      const double theta = medium.sample_scattering_angle(rng);

      // Get scattering matrix
      CMatrix S_matrix = medium.scattering_matrix(theta, 0, photon.k);

      // Sample azimuthal angle
      const double phi = medium.sample_conditional_azimuthal_angle(rng, S_matrix, photon.polarization, photon.k, theta);

      // Precompute trigonometric values for scattering
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

      // Update local scattering plane basis
      matmul(A_update, photon.P_local, photon.P_local);

      // Update photon polarization if needed
      if (photon.polarized)
      {
        // Construct scattering matrix M_current = S * R
        CMatrix R(2, 2);
        R(0, 0) = cos_phi;
        R(0, 1) = sin_phi;
        R(1, 0) = -sin_phi;
        R(1, 1) = cos_phi;
        CMatrix T_current = CMatrix(2, 2);
        matcmul(S_matrix, R, T_current);

        // Calculate normalization factor F (m=1, n=2)
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

        // Update polarization components
        const double F_inv_sqrt = 1.0 / std::sqrt(F);
        matcmulscalar(F_inv_sqrt, T_current);
        photon.polarization.m = (T_current(0, 0) * Em + T_current(0, 1) * En);
        photon.polarization.n = (T_current(1, 0) * Em + T_current(1, 1) * En);

        // Update scatter info for CBS
        if (track_reverse_paths)
        {
          // índice del scattering que acabas de ejecutar (1,2,3,...)
          const uint evt = photon.events + 1;
          if (evt == 1)
          {
            photon.r_0 = photon.pos;
            photon.P1 = photon.P_local; // base después del primer scattering
          }

          photon.r_n = photon.pos; // siempre se actualiza (último scattering)
          photon.Pn2 = photon.Pn1;
          photon.Pn1 = photon.Pn;
          photon.Pn = photon.P_local;

          if (evt >= 2)
          {
            if (!photon.has_T_prev)
            {
              // evt=2: guardo J2 como candidato a "último"
              photon.matrix_T_buffer = T_current;
              photon.has_T_prev = true;
            }
            else
            {
              // evt>=3: ya sé que el evento anterior (J_prev) NO era el último -> entra en T
              CMatrix tmp(2, 2);
              matcmul(photon.matrix_T_buffer, photon.matrix_T, tmp); // T_mid = J_prev * T_mid
              photon.matrix_T = std::move(tmp);
              photon.matrix_T_buffer = T_current;
            }
          }
        }
      }

      LLOG_DEBUG("Photon weight after event {}: {}", photon.events, photon.weight);

      // Update photon events
      const double d_weight = photon.weight * (medium.mu_absorption / medium.mu_attenuation);
      photon.weight = photon.weight * (medium.mu_scattering / medium.mu_attenuation);
      photon.events++;

      // Record absorption
      if (absorption)
      {
        absorption->record_absorption(photon, d_weight);
      }

      // Russian roulette for photon termination
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
