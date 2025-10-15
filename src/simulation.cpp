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
  while (photon.alive) {
    // Sample free step
    const double step = medium.sample_free_path(rng);
    photon.opticalpath += step;
    photon.prev_pos = photon.pos;
    photon.pos.x += step * photon.dir.x;
    photon.pos.y += step * photon.dir.y;
    photon.pos.z += step * photon.dir.z;

    // Check for detector hit
    if (photon.events != 0) {
      detector.record_hit(photon);
    }
    if (!photon.alive)
      break;

    // Scatter the photon
    const double theta = medium.sample_scattering_angle(rng);

    // Get scattering matrix
    CVec2 S = medium.scattering_matrix(theta, 0, photon.k);

    // const double phi = medium.sample_azimuthal_angle(rng);
    const double phi = medium.sample_conditional_azimuthal_angle(rng, S, photon.polarization, photon.k, theta);
    const double cos_theta = std::cos(theta);
    const double sin_theta = std::sin(theta);
    const double cos_phi = std::cos(phi);
    const double sin_phi = std::sin(phi);

    // Update photon polarization if needed
    if (photon.polarized) {
      const std::complex<double> Em = photon.polarization.m;
      const std::complex<double> En = photon.polarization.n;

      const double Emm = std::norm(Em);
      const double Enn = std::norm(En);
      const double s22 = std::norm(S.m);
      const double s11 = std::norm(S.n);

      const double pow_cos_phi = std::pow(cos_phi, 2);
      const double pow_sin_phi = std::pow(sin_phi, 2);

      const double F =
          Emm * (s22 * pow_cos_phi + s11 * pow_sin_phi) +
          Enn * (s22 * pow_sin_phi + s11 * pow_cos_phi) +
          2.0 * std::real(Em * std::conj(En)) * (s22 - s11) * sin_phi * cos_phi;

      const double F_inv_sqrt = 1.0 / std::sqrt(F);
      photon.polarization.m = F_inv_sqrt * S.m * (Em * cos_phi + En * sin_phi);
      photon.polarization.n = F_inv_sqrt * S.n * (-Em * sin_phi + En * cos_phi);
    }

    // Update photon direction
    const Vec3 old_dir = photon.dir;
    const Vec3 old_m = photon.m;
    const Vec3 old_n = photon.n;

    photon.m.x = old_m.x * cos_theta * cos_phi + old_n.x * cos_theta * sin_phi - old_dir.x * sin_theta;
    photon.m.y = old_m.y * cos_theta * cos_phi + old_n.y * cos_theta * sin_phi - old_dir.y * sin_theta;
    photon.m.z = old_m.z * cos_theta * cos_phi + old_n.z * cos_theta * sin_phi - old_dir.z * sin_theta;
    photon.n.x = old_m.x * -1 * sin_phi + old_n.x * cos_phi;
    photon.n.y = old_m.y * -1 * sin_phi + old_n.y * cos_phi;
    photon.n.z = old_m.z * -1 * sin_phi + old_n.z * cos_phi;
    photon.dir.x = old_m.x * sin_theta * cos_phi + old_n.x * sin_theta * sin_phi + old_dir.x * cos_theta;
    photon.dir.y = old_m.y * sin_theta * cos_phi + old_n.y * sin_theta * sin_phi + old_dir.y * cos_theta;
    photon.dir.z = old_m.z * sin_theta * cos_phi + old_n.z * sin_theta * sin_phi + old_dir.z * cos_theta;

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
