#include "luminis/core/detector.hpp"
#include <cstdint>
#include <luminis/core/absortion.hpp>
#include <luminis/log/logger.hpp>
#include <luminis/core/simulation.hpp>
#include <cmath>
#include <thread>
#include <vector>
#include <exception>
#include <atomic>
#include <sstream>

namespace luminis::core
{

  using luminis::math::mix_seed;
  using luminis::math::Rng;

  SimConfig::SimConfig(std::size_t n, Medium *m, Laser *l, MultiDetector *d, AbsorptionTimeDependent *a, bool track_reverse_paths)
      : n_photons(n), medium(m), laser(l), detector(d), absorption(a), track_reverse_paths(track_reverse_paths) {}

  SimConfig::SimConfig(std::uint64_t s, std::size_t n, Medium *m, Laser *l, MultiDetector *d, AbsorptionTimeDependent *a, bool track_reverse_paths)
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
    std::vector<std::unique_ptr<MultiDetector>> thread_detectors;
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

        MultiDetector& det = *thread_detectors[t];
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

  void run_photon(Photon &photon, Medium &medium, MultiDetector &detector, Rng &rng, AbsorptionTimeDependent *absorption, bool track_reverse_paths)
  {
    const uint first_event = 0;

    // Update incident direction for CBS
    photon.n_0 = photon.n;
    photon.s_0 = photon.dir;
    photon.s_1 = photon.dir;
    photon.s_n2 = photon.dir;
    photon.s_n1 = photon.dir;
    photon.s_n = photon.dir;
    photon.initial_polarization = photon.polarization;

    // Main photon propagation loop
    while (photon.alive)
    {
      // Sample free step
      const double step = medium.sample_free_path(rng);
      photon.opticalpath += step;
      photon.prev_pos = photon.pos;
      photon.pos = photon.pos + photon.dir * step;

      // Check for detector hit
      if (photon.events != first_event)
      {
        const bool hit = detector.record_hit(photon, [&photon, &medium, track_reverse_paths]() { if (track_reverse_paths && photon.events > 1) coherent_calculation(photon, medium); });
        if (hit)
        {
          photon.alive = false;
          break;
        }
      }

      const bool is_inside = medium.is_inside(photon.pos);
      if (!is_inside)
      {
        photon.alive = false;
        break;
      }

      // Scatter the photon
      const double theta = medium.sample_scattering_angle(rng);

      // Get scattering matrix
      CMatrix S_matrix = medium.scattering_matrix(theta, 0, photon.k);

      // const double phi = medium.sample_azimuthal_angle(rng);
      const double phi = medium.sample_conditional_azimuthal_angle(rng, S_matrix, photon.polarization, photon.k, theta);
      const double cos_theta = std::cos(theta);
      const double sin_theta = std::sin(theta);
      const double cos_phi = std::cos(phi);
      const double sin_phi = std::sin(phi);

      // Update photon direction
      const Vec3 old_dir = photon.dir;
      const Vec3 old_m = photon.m;
      const Vec3 old_n = photon.n;

      photon.m = old_m * cos_theta * cos_phi + old_n * cos_theta * sin_phi - old_dir * sin_theta;
      photon.n = old_m * -1 * sin_phi + old_n * cos_phi;
      photon.dir = old_m * sin_theta * cos_phi + old_n * sin_theta * sin_phi + old_dir * cos_theta;

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
        matmul(S_matrix, R, T_current);

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

        // if (F < 1e-18)
        // {
        //   photon.weight = 0.0;
        //   photon.alive = false;
        //   break;
        // }

        // Update polarization components
        const double F_inv_sqrt = 1.0 / std::sqrt(F);
        matmulscalar(F_inv_sqrt, T_current);
        photon.polarization.m = (T_current(0, 0) * Em + T_current(0, 1) * En);
        photon.polarization.n = (T_current(1, 0) * Em + T_current(1, 1) * En);

        const double intensity_unpolarized = 0.5 * (s22 + s11);
        // if (intensity_unpolarized < 1e-18)
        // {
        //   photon.weight = 0.0;
        //   photon.alive = false;
        //   break;
        // }

        const double expected_intensity = F / intensity_unpolarized;

        // Update first scatter info for CBS
        if (track_reverse_paths)
        {
          if (photon.events == first_event)
          {
            photon.r_0 = photon.pos;
            photon.s_1 = photon.dir;
            photon.s_n = photon.dir;
            photon.s_n1 = photon.s_0;
            photon.s_n2 = photon.s_0;
          }
          else
          {
            photon.r_n = photon.pos;
            photon.s_n2 = photon.s_n1;
            photon.s_n1 = photon.s_n;
            photon.s_n = photon.dir;

            // Update matrix T
            matmul(photon.matrix_T_buffer, photon.matrix_T, photon.matrix_T);
            photon.matrix_T_buffer.data.swap(T_current.data);
          }
        }
      }

      LLOG_DEBUG("Photon weight after event {}: {}", photon.events, photon.weight);

      // Update photon events
      const double d_weight = photon.weight * (medium.mu_absorption / medium.mu_attenuation);
      photon.weight = photon.weight * (medium.mu_scattering / medium.mu_attenuation);
      photon.events++;

      // Record absorption
      if (absorption) {
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
    }

    LLOG_DEBUG("Photon terminated after {} events, final weight: {}, optical path: {}", photon.events, photon.weight, photon.opticalpath);
  }

  void coherent_calculation(Photon &photon, Medium &medium)
  {
    if (photon.coherent_path_calculated)
      return;

    photon.coherent_path_calculated = true;

    Vec3 s_0 = photon.s_0;
    Vec3 s_1 = photon.s_1;
    Vec3 s_n2 = photon.s_n2;
    Vec3 s_n1 = photon.s_n1;
    Vec3 s_n = photon.s_n;
    Vec3 n_0 = photon.n_0; // Polarización inicial

    // --- CÁLCULO SEGURO DE NORMALES ---
    // Usamos un epsilon pequeño para detectar colinealidad
    const double EPSILON = 1e-12;

    Vec3 n_1 = cross(s_0, s_1);
    bool n_1_valid = (dot(n_1, n_1) > EPSILON);

    Vec3 n_prime = cross(s_0, s_n1 * (-1.0));
    bool n_prime_valid = (dot(n_prime, n_prime) > EPSILON);

    Vec3 n_n1 = cross(s_n1 * (-1.0), s_n2 * (-1.0));
    double phi_n = 0.0;
    if (n_prime_valid)
    {
      phi_n = calculate_rotation_angle(n_0, n_prime);
    }
    if (std::isnan(phi_n))
      phi_n = 0.0;

    double phi_n_prime = 0.0;
    if (n_prime_valid && n_1_valid)
    {
      phi_n_prime = calculate_rotation_angle(n_prime, n_1 * (-1.0));
    }
    if (std::isnan(phi_n_prime))
      phi_n_prime = 0.0;

    double phi_1_prime = 0.0;
    if (n_1_valid)
    {
      phi_1_prime = calculate_rotation_angle(n_1 * (-1.0), n_n1);
    }
    if (std::isnan(phi_1_prime))
      phi_1_prime = 0.0;

    auto safe_acos_dot = [](Vec3 a, Vec3 b)
    {
      double d = dot(a, b);
      if (d > 1.0)
        d = 1.0;
      if (d < -1.0)
        d = -1.0;
      return std::acos(d);
    };

    double theta_n = safe_acos_dot(s_n1 * (-1.0), s_0);
    double theta_1 = safe_acos_dot(s_n, s_1 * (-1.0));

    // Rotation matrices
    CMatrix R_n(2, 2);
    R_n(0, 0) = std::cos(phi_n);
    R_n(0, 1) = std::sin(phi_n);
    R_n(1, 0) = -std::sin(phi_n);
    R_n(1, 1) = std::cos(phi_n);
    CMatrix R_n_prime(2, 2);
    R_n_prime(0, 0) = std::cos(phi_n_prime);
    R_n_prime(0, 1) = std::sin(phi_n_prime);
    R_n_prime(1, 0) = -std::sin(phi_n_prime);
    R_n_prime(1, 1) = std::cos(phi_n_prime);
    CMatrix R_1_prime(2, 2);
    R_1_prime(0, 0) = std::cos(phi_1_prime);
    R_1_prime(0, 1) = std::sin(phi_1_prime);
    R_1_prime(1, 0) = -std::sin(phi_1_prime);
    R_1_prime(1, 1) = std::cos(phi_1_prime);

    // Scattering matrices
    CMatrix S_n = medium.scattering_matrix(theta_n, 0, photon.k);
    CMatrix S_1 = medium.scattering_matrix(theta_1, 0, photon.k);

    // Auxiliary matrix Q
    CMatrix Q(2, 2);
    Q(0, 0) = 1;
    Q(0, 1) = 0;
    Q(1, 0) = 0;
    Q(1, 1) = -1;

    // Calculate reversed path matrix
    CMatrix T_forward_transposed = CMatrix(2, 2);
    T_forward_transposed(0, 0) = photon.matrix_T(0, 0);
    T_forward_transposed(0, 1) = photon.matrix_T(1, 0);
    T_forward_transposed(1, 0) = photon.matrix_T(0, 1);
    T_forward_transposed(1, 1) = photon.matrix_T(1, 1);

    // LLOG_INFO("CBS forward path matrix T_forward:");
    // LLOG_INFO("[[{}+i{}, {}+i{}],", photon.matrix_T(0,0).real(), photon.matrix_T(0,0).imag(), photon.matrix_T(0,1).real(), photon.matrix_T(0,1).imag());
    // LLOG_INFO(" [{}+i{}, {}+i{}]]", photon.matrix_T(1,0).real(), photon.matrix_T(1,0).imag(), photon.matrix_T(1,1).real(), photon.matrix_T(1,1).imag());

    CMatrix J_reversed = CMatrix::identity(2);
    matmul(S_n, R_n, J_reversed);
    matmul(R_n_prime, J_reversed, J_reversed);
    matmul(Q, J_reversed, J_reversed);
    matmul(T_forward_transposed, J_reversed, J_reversed);
    matmul(Q, J_reversed, J_reversed);
    matmul(R_1_prime, J_reversed, J_reversed);
    matmul(S_1, J_reversed, J_reversed);

    // LLOG_INFO("CBS reversed path matrix T_reversed:");
    // LLOG_INFO("[[{}+i{}, {}+i{}],", J_reversed(0,0).real(), J_reversed(0,0).imag(), J_reversed(0,1).real(), J_reversed(0,1).imag());
    // LLOG_INFO(" [{}+i{}, {}+i{}]]", J_reversed(1,0).real(), J_reversed(1,0).imag(), J_reversed(1,1).real(), J_reversed(1,1).imag());

    CVec2 E_reversed;
    const std::complex<double> Em0 = photon.initial_polarization.m;
    const std::complex<double> En0 = photon.initial_polarization.n;
    E_reversed.m = J_reversed(0, 0) * Em0 + J_reversed(0, 1) * En0;
    E_reversed.n = J_reversed(1, 0) * Em0 + J_reversed(1, 1) * En0;

    // LLOG_INFO("CBS forward path polarization: m = {}+i{}, n = {}+i{}", photon.polarization.m.real(), photon.polarization.m.imag(), photon.polarization.n.real(), photon.polarization.n.imag());
    // LLOG_INFO("CBS reversed path polarization: m = {}+i{}, n = {}+i{}", E_reversed.m.real(), E_reversed.m.imag(), E_reversed.n.real(), E_reversed.n.imag());

    double mag_sq = std::norm(E_reversed.m) + std::norm(E_reversed.n);
    if (mag_sq > 1e-20)
    {
      double inv_norm = 1.0 / std::sqrt(mag_sq);
      E_reversed.m *= inv_norm;
      E_reversed.n *= inv_norm;
    }

    photon.polarization_reverse = E_reversed;
  }

} // namespace luminis::core
