#pragma once
#include <luminis/core/absortion.hpp>
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
    std::size_t n_threads = 1;
    std::size_t n_photons;

    bool track_reverse_paths{false};

    Medium *medium{nullptr};
    Laser *laser{nullptr};
    SensorsGroup *detector{nullptr};
    AbsorptionTimeDependent *absorption{nullptr};

    SimConfig(std::size_t n, Medium *m = nullptr, Laser *l = nullptr, SensorsGroup *d = nullptr, AbsorptionTimeDependent *a = nullptr, bool track_reverse_paths = false); 
    SimConfig(std::uint64_t s, std::size_t n, Medium *m = nullptr, Laser *l = nullptr, SensorsGroup *d = nullptr, AbsorptionTimeDependent *a = nullptr, bool track_reverse_paths = false);
  };

  void run_simulation(const SimConfig &config);

  void run_simulation_parallel(const SimConfig &config);

  void run_photon(Photon &photon, Medium &medium, SensorsGroup &detector, Rng &rng, AbsorptionTimeDependent *absorption, bool track_reverse_paths);

  void coherent_calculation(Photon &photon, Medium &medium);

} // namespace luminis::core
