from luminis_mc import (
    Laser,
    SimpleMedium,
    Detector,
    SimConfig,
    RayleighDebyePhaseFunction,
    CVec2,
    Vec3,
)
from luminis_mc import LogLevel, LaserSource
from luminis_mc import run_simulation_parallel, set_log_level
import numpy as np
import time

set_log_level(LogLevel.debug)


start_time = time.time()

# Global frame of reference
m_global = Vec3(1, 0, 0)
n_global = Vec3(0, 1, 0)
s_global = Vec3(0, 0, 1)
light_speed = 299792458e-6

# Medium parameters
radius = 0.46 # in micrometers
mean_free_path = 2.8 # in micrometers
wavelength = 0.525  # in micrometers
inv_mfp = 1 / mean_free_path
mu_absortion = 0.0003 * inv_mfp
mu_scattering = inv_mfp - mu_absortion
# mean_free_path = 1 / (mu_absortion + mu_scattering)

print(f"Mean free path: {mean_free_path}")
print(f"Medium radius: {radius}")

# Time parameters
t_ref = mean_free_path / light_speed
dt = 0
max_time = 50 * t_ref

# Phase function parameters
thetaMin = 0.00001
thetaMax = np.pi
nDiv = 1000
n_photons = 5_000_000

# Laser parameters
origin = Vec3(0, 0, 0)
polarization = CVec2(1, 0)
laser_radius = 0.1 * mean_free_path
laser_type = LaserSource.Gaussian

laser_source = Laser(origin, s_global, n_global, m_global, polarization, wavelength, laser_radius, laser_type)
detector = Detector(0)
phase_function = RayleighDebyePhaseFunction(wavelength, radius, nDiv, thetaMin, thetaMax)
medium = SimpleMedium(mu_absortion, mu_scattering, phase_function, mean_free_path, radius)

config = SimConfig(
    n_photons=n_photons,
    medium=medium,
    detector=detector,
    laser=laser_source,
)
config.n_threads = 8

run_simulation_parallel(config)

end_time = time.time()
print(f"Simulation time: {end_time - start_time:.2f} seconds")

detector.save_recorded_photons("test-data-phi-random.dat")
