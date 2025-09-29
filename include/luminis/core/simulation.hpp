#pragma once
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

  struct SimConfig
  {
    std::uint64_t seed = std::random_device{}();
    std::size_t n_photons;

    SimConfig(std::size_t n);
    SimConfig(std::uint64_t s, std::size_t n);
  };

  void run_simulation(SimConfig &config, Medium &medium, Detector &detector, Laser &laser);

  void run_photon(Photon &photon, Medium &medium, Detector &detector, Rng &rng);

} // namespace luminis::core
