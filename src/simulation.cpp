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


namespace luminis::core {

using luminis::math::Rng;
using luminis::math::mix_seed;

SimConfig::SimConfig(std::size_t n, Medium *m, Laser *l, Detector *d, AbsorptionTimeDependent *a)
    : n_photons(n), medium(m), laser(l), detector(d), absorption(a) {}

SimConfig::SimConfig(std::uint64_t s, std::size_t n, Medium *m, Laser *l, Detector *d, AbsorptionTimeDependent *a)
    : seed(s), n_photons(n), medium(m), laser(l), detector(d), absorption(a) {}

void run_simulation(const SimConfig &config) {
  Rng rng(config.seed);

  for (std::size_t i = 0; i < config.n_photons; ++i) {
    Photon photon = config.laser->emit_photon(rng);
    photon.velocity = config.medium->light_speed_in_medium();
    run_photon(photon, *config.medium, *config.detector, rng, config.absorption);
  }
}

void run_simulation_parallel(const SimConfig &config) {
  // Determine number of threads to use
  std::size_t n_threads = config.n_threads;
  if (n_threads == 0)
    n_threads = std::thread::hardware_concurrency();
  if (n_threads > config.n_photons)
    n_threads = config.n_photons;

  const std::size_t base = config.n_photons / n_threads;
  const std::size_t rem  = config.n_photons % n_threads;

  LLOG_INFO("Running simulation with {} threads for {} photons", n_threads, config.n_photons);


  // Create thread-local detectors and absorptions
  std::vector<Detector> thread_detectors;
  thread_detectors.reserve(n_threads);
  std::vector<AbsorptionTimeDependent> thread_absorptions;
  if (config.absorption)
    thread_absorptions.reserve(n_threads);

  for (std::size_t t = 0; t < n_threads; ++t) {
    thread_detectors.emplace_back(config.detector->copy_start());

    if (config.absorption)
      thread_absorptions.emplace_back(config.absorption->copy_start());
  }


  // Launch threads
  std::vector<std::thread> workers;
  workers.reserve(n_threads);

  std::atomic<bool> any_error{false};
  std::exception_ptr thread_exception = nullptr;

  for (std::size_t t = 0; t < n_threads; ++t) {
    const std::size_t my_count = base + (t < rem ? 1u : 0u);

    workers.emplace_back([&, t, my_count]() {
      try {
        const std::uint64_t thread_seed = mix_seed(config.seed, static_cast<std::uint64_t>(t));
        Rng rng(thread_seed);

        std::ostringstream oss;
        oss << "Thread " << t << " (id " << std::this_thread::get_id() << ") processing " << my_count << " photons with seed " << thread_seed;
        LLOG_INFO(oss.str());

        Detector& det = thread_detectors[t];
        AbsorptionTimeDependent* abs_ptr = nullptr;
        if (config.absorption) abs_ptr = &thread_absorptions[t];

        for (std::size_t i = 0; i < my_count; ++i) {
          Photon photon = config.laser->emit_photon(rng);
          photon.velocity = config.medium->light_speed_in_medium();
          run_photon(photon, *config.medium, det, rng, abs_ptr);
        }

      } catch (...) {
        any_error = true;
        thread_exception = std::current_exception();
      }
    });
  }


  // Join threads
  for (auto &th : workers) th.join();
  if (any_error && thread_exception) {
    std::rethrow_exception(thread_exception);
  }


  // Merge thread-local detectors and absorptions
  for (std::size_t t = 0; t < n_threads; ++t) {
    config.detector->merge_from(thread_detectors[t]);
  }
  if (config.absorption) {
    for (std::size_t t = 0; t < n_threads; ++t) {
      config.absorption->merge_from(thread_absorptions[t]);
    }
  }

  LLOG_INFO("Parallel simulation finished. Total hits: {}", config.detector->hits);
}

void run_photon(Photon &photon, Medium &medium, Detector &detector, Rng &rng, AbsorptionTimeDependent *absorption) {
  const uint first_event = 0;

  // Update incident direction for CBS
  photon.s_0 = photon.dir;


  while (photon.alive) {
    // Sample free step
    const double step = medium.sample_free_path(rng);
    photon.opticalpath += step;
    photon.prev_pos = photon.pos;
    photon.pos = photon.pos + photon.dir * step;

    // Update first scatter info for CBS
    if (photon.events == first_event) {
      photon.r_0 = photon.pos;
    }

    // Check for detector hit
    if (photon.events != first_event) {
      detector.record_hit(photon);
    }
    if (!photon.alive)
      break;

    // Scatter the photon
    const double theta = medium.sample_scattering_angle(rng);

    // Get scattering matrix
    CVec2 S_vec = medium.scattering_matrix(theta, 0, photon.k);

    const double phi = medium.sample_azimuthal_angle(rng);
    // const double phi = medium.sample_conditional_azimuthal_angle(rng, S_vec, photon.polarization, photon.k, theta);
    const double cos_theta = std::cos(theta);
    const double sin_theta = std::sin(theta);
    const double cos_phi = std::cos(phi);
    const double sin_phi = std::sin(phi);



    // Update photon polarization if needed
    if (photon.polarized) {
      // Construct scattering matrix M_current = S * R
      CMatrix R(2, 2);
      R(0, 0) = cos_phi;   R(0, 1) = sin_phi;
      R(1, 0) = -sin_phi;  R(1, 1) = cos_phi;
      CMatrix S(2, 2);
      S(0, 0) = S_vec.m;   S(0, 1) = 0.0;
      S(1, 0) = 0.0;       S(1, 1) = S_vec.n;
      CMatrix M_current = matmul(S, R);

      // Calculate normalization factor F
      const std::complex<double> Em = photon.polarization.m;
      const std::complex<double> En = photon.polarization.n;

      const double Emm = std::norm(Em);
      const double Enn = std::norm(En);
      const double s22 = std::norm(S_vec.m);
      const double s11 = std::norm(S_vec.n);

      const double pow_cos_phi = std::pow(cos_phi, 2);
      const double pow_sin_phi = std::pow(sin_phi, 2);

      const double F =
          Emm * (s22 * pow_cos_phi + s11 * pow_sin_phi) +
          Enn * (s22 * pow_sin_phi + s11 * pow_cos_phi) +
          2.0 * std::real(Em * std::conj(En)) * (s22 - s11) * sin_phi * cos_phi;

      // Update polarization components
      const double F_inv_sqrt = 1.0 / std::sqrt(F);
      photon.polarization.m = (M_current(0, 0) * Em + M_current(0, 1) * En) * F_inv_sqrt;
      photon.polarization.n = (M_current(1, 0) * Em + M_current(1, 1) * En) * F_inv_sqrt;




    }

    // Update photon direction
    const Vec3 old_dir = photon.dir;
    const Vec3 old_m = photon.m;
    const Vec3 old_n = photon.n;

    photon.m = old_m * cos_theta * cos_phi + old_n * cos_theta * sin_phi - old_dir * sin_theta;
    photon.n = old_m * -1 * sin_phi + old_n * cos_phi;
    photon.dir = old_m * sin_theta * cos_phi + old_n * sin_theta * sin_phi + old_dir * cos_theta;

    // Update photon events
    const double d_weight = photon.weight * (medium.mu_absorption / medium.mu_attenuation);
    photon.weight = photon.weight - d_weight;
    photon.events++;

    // Record absorption
    if (absorption) {
      absorption->record_absorption(photon, d_weight);
    }

    // Russian roulette for photon termination
    if (photon.weight < 1e-4) {
      if (rng.uniform() < 0.1) {
        photon.weight /= 0.1;
      } else {
        photon.alive = false;
        break;
      }
    }
  }

  // LLOG_DEBUG(
  //     "Photon terminated after {} events, final weight: {}, optical path: {}",
  //     photon.events, photon.weight, photon.opticalpath);
}

} // namespace luminis::core
