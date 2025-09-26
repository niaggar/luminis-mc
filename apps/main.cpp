#include <luminis/log/logger.hpp>
#include <luminis/core/simulation.hpp>
#include <luminis/core/detector.hpp>
#include <luminis/core/laser.hpp>
#include <luminis/core/medium.hpp>
#include <luminis/core/photon.hpp>
#include <luminis/math/vec.hpp>

int main() {
  using namespace luminis::core;
  using luminis::log::Level;
  using luminis::log::Logger;

  Logger::instance().set_level(Level::debug);

  LLOG_DEBUG("Log debug message");
  LLOG_INFO("Log info message");
  LLOG_WARN("Log warning message");
  LLOG_ERROR("Log error message");

  Laser laser({0, 0, 0}, {0, 0, 1}, {1, 0}, 532.0, 1.0, LaserSource::Gaussian);
  Detector detector({0, 0, 0}, {0, 0, 1});
  HenyeyGreensteinPhaseFunction phase_func(0.9);
  SimpleMedium medium(0.01, 0.1, &phase_func, 1.33, 0.1);

  SimConfig config;
  config.n_photons = 10;
  config.seed = 3942;

  run_simulation(config, medium, detector, laser);
  
  return 0;
}
