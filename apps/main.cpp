#include "luminis/simulation.hpp"
#include <print>

int main() {
  using namespace luminis;

  SimConfig cfg;
  cfg.n_photons = 10000;
  cfg.max_scatter = 1000;

  Material mat(/*mean_free_path=*/5.0);
  PlaneDetector det(/*z_plane=*/100.0, /*radius=*/10.0);

  Simulation sim(cfg, mat, det);
  auto stats = sim.run();

  std::print("Final: hits={} emitted={} rate={:.6f}\n", stats.detected,
             stats.emitted, stats.detection_rate());
  return 0;
}
