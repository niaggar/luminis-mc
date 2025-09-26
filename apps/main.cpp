#include <luminis/core/simulation.hpp>
#include <luminis/log/logger.hpp>

int main() {
  // using namespace luminis::core;
  // using luminis::log::Level;
  // using luminis::log::Logger;

  // Logger::instance().set_level(Level::debug);

  // LLOG_DEBUG("Log debug message");
  // LLOG_INFO("Log info message");
  // LLOG_WARN("Log warning message");
  // LLOG_ERROR("Log error message");

  // SimConfig cfg;
  // cfg.n_photons = 10000;
  // cfg.max_scatter = 1000;

  // Material mat(5.0);
  // PlaneDetector det(100.0, 10.0);

  // Simulation sim(cfg, mat, det);
  // auto stats = sim.run();

  // LLOG_INFO("Simulation complete: hits={} emitted={} rate={:.6f}",
  //           stats.detected, stats.emitted, stats.detection_rate());
  // return 0;
}
