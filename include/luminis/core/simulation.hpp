#pragma once
#include <cstdint>
#include <luminis/core/detector.hpp>
#include <luminis/core/material.hpp>
#include <luminis/core/photon.hpp>
#include <print>

namespace luminis::core {

struct SimConfig {
  std::uint64_t seed = std::random_device{}();
  std::size_t n_photons = 10000;
  std::size_t max_scatter = 1000;
  double world_z_min = 0.0; // terminate if photon goes below
  double world_z_max = 1e6; // safety cap
};

struct SimStats {
  std::size_t emitted = 0;
  std::size_t detected = 0;
  double detection_rate() const {
    return emitted ? double(detected) / double(emitted) : 0.0;
  }
};

class Simulation {
public:
  Simulation(SimConfig cfg, Material mat, PlaneDetector det)
      : cfg_(cfg), mat_(mat), det_(det), rng_(cfg.seed) {}

  SimStats run() {
    SimStats st{};
    st.emitted = cfg_.n_photons;

    for (std::size_t i = 0; i < cfg_.n_photons; ++i) {
      Photon ph({0, 0, 0}, {0, 0, 1}, 532.0); // source at origin, shooting +z
      bool got_hit = trace(ph);
      if (got_hit)
        ++st.detected;
    }

    return st;
  }

  const PlaneDetector &detector() const { return det_; }

private:
  bool trace(Photon &ph) {
    for (std::size_t k = 0; k < cfg_.max_scatter; ++k) {
      const double s = mat_.sample_free_path(rng_);
      // Check intersection with the plane during this free-flight
      if (det_.intersect_and_record(ph.pos, ph.dir, s)) {
        // We can early-return on first hit
        ph.alive = false;
        return true;
      }
      // Move
      ph.move(s);
      // Bounds
      if (ph.pos[2] < cfg_.world_z_min || ph.pos[2] > cfg_.world_z_max)
        break;
      // Scatter
      ph.dir = mat_.scatter(rng_);
    }
    ph.alive = false;
    return false;
  }

  SimConfig cfg_;
  Material mat_;
  PlaneDetector det_;
  Rng rng_;
};

} // namespace luminis::core
