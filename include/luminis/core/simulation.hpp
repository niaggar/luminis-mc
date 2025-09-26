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

struct SimStats {

};

} // namespace luminis::core
