from luminis_mc import (
    Laser,
    SimpleMedium,
    Detector,
    SimConfig,
    RayleighDebyeEMCPhaseFunction,
    CVec2,
    Vec3,
    compute_events_histogram,
    load_recorded_photons,
    save_recorded_photons
)
from luminis_mc import LogLevel, LaserSource
from luminis_mc import run_simulation_parallel, set_log_level
import numpy as np
import time

set_log_level(LogLevel.warn)


start_time = time.time()

# Global frame of reference
m_global = Vec3(1, 0, 0)
n_global = Vec3(0, 1, 0)
s_global = Vec3(0, 0, 1)
light_speed = 1

# Medium parameters in micrometers
radius = 0.1
mean_free_path = 100
wavelength = 0.532
inv_mfp = 1 / mean_free_path
mu_absortion = 0.0003 * inv_mfp
mu_scattering = inv_mfp - mu_absortion
n_particle = 1.59
n_medium = 1.33

print(f"Mean free path: {mean_free_path}")
print(f"Particle radius: {radius}")
print(f"Wavelength: {wavelength}")
print(f"Scattering coefficient: {mu_scattering}")
print(f"Absorption coefficient: {mu_absortion}")
print(f"Albedo: {mu_scattering / (mu_scattering + mu_absortion)}")
print(f"Refractive index particle: {n_particle}")
print(f"Refractive index medium: {n_medium}")
print(f"Relative refractive index: {n_particle / n_medium}")

# Time parameters
t_ref = mean_free_path / light_speed
dt = 0
max_time = 50 * t_ref

# Phase function parameters
thetaMin = 0.00001
thetaMax = np.pi
nDiv = 1000
n_photons = 10_000_000

# Laser parameters
origin = Vec3(0, 0, 0)
polarization = CVec2(1, 0)
laser_radius = 0.1 * mean_free_path
laser_type = LaserSource.Gaussian

laser_source = Laser(origin, polarization, wavelength, laser_radius, laser_type)
detector = Detector(0)
phase_function = RayleighDebyeEMCPhaseFunction(wavelength, radius, n_particle, n_medium, nDiv, thetaMin, thetaMax)
medium = SimpleMedium(mu_absortion, mu_scattering, phase_function, mean_free_path, radius, n_particle, n_medium)

config = SimConfig(
    n_photons=n_photons,
    medium=medium,
    detector=detector,
    laser=laser_source,
    parallel=True,
)
config.n_threads = 8

run_simulation_parallel(config)

end_time = time.time()
print(f"Simulation time: {end_time - start_time:.2f} seconds")

save_recorded_photons("test-data-phi-random.dat", detector)
