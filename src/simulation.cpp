#include "luminis/log/logger.hpp"
#include <cmath>
#include <luminis/core/simulation.hpp>

namespace luminis::core {

SimConfig::SimConfig(std::size_t n) : n_photons(n) {}

SimConfig::SimConfig(std::uint64_t s, std::size_t n) : seed(s), n_photons(n) {}

void run_simulation(SimConfig &config, Medium &medium, Detector &detector, Laser &laser) {
  Rng rng(config.seed);

  for (std::size_t i = 0; i < config.n_photons; ++i) {
    Photon photon = laser.emit_photon(rng);
    photon.velocity = medium.light_speed_in_medium();

    run_photon(photon, medium, detector, rng);
  }
}

void run_photon(Photon &photon, Medium &medium, Detector &detector, Rng &rng) {
  while (photon.alive) {
    // Sample free step
    const double step = medium.sample_free_path(rng);
    photon.opticalpath += step;
    photon.prev_pos = photon.pos;
    photon.pos = photon.prev_pos + photon.dir * step;

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
      const std::complex<double> Em = photon.polarization[0];
      const std::complex<double> En = photon.polarization[1];

      const double Emm = std::norm(Em);
      const double Enn = std::norm(En);
      const double s22 = std::norm(S[0]);
      const double s11 = std::norm(S[1]);

      const double pow_cos_phi = std::pow(cos_phi, 2);
      const double pow_sin_phi = std::pow(sin_phi, 2);

      const double F =
          Emm * (s22 * pow_cos_phi + s11 * pow_sin_phi) +
          Enn * (s22 * pow_sin_phi + s11 * pow_cos_phi) +
          2.0 * std::real(Em * std::conj(En)) * (s22 - s11) * sin_phi * cos_phi;

      const double F_inv_sqrt = 1.0 / std::sqrt(F);
      photon.polarization[0] =
          F_inv_sqrt * S[0] * (Em * cos_phi + En * sin_phi);
      photon.polarization[1] =
          F_inv_sqrt * S[1] * (-Em * sin_phi + En * cos_phi);
    }

    // Update photon direction
    const Vec3 old_dir = photon.dir;
    const Vec3 old_m = photon.m;
    const Vec3 old_n = photon.n;

    photon.dir = old_m * sin_theta * cos_phi + old_n * sin_theta * sin_phi + old_dir * cos_theta;
    photon.m = old_m * cos_theta * cos_phi + old_n * cos_theta * sin_phi - old_dir * sin_theta;
    photon.n = old_n * cos_phi - old_m * sin_phi;

    // Update photon events
    const double d_weight = photon.weight * (medium.mu_absorption / medium.mu_attenuation);
    photon.weight = photon.weight - d_weight;
    photon.events++;

    // Record absorption
    if (medium.absorption) {
      medium.absorption->record_absorption(photon, d_weight);
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
