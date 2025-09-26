#pragma once
#include <cstdint>
#include <luminis/core/detector.hpp>
#include <luminis/core/medium.hpp>
#include <luminis/core/photon.hpp>

namespace luminis::core {

struct SimConfig {
  std::uint64_t seed = std::random_device{}();
  std::size_t n_photons = 10000;
  std::size_t max_scatter = 1000;
};

void run_simulation(const SimConfig &config, Medium &medium, Detector &detector);

void run_photon(Photon &photon, Medium &medium, Detector &detector);

} // namespace luminis::core
