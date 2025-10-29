# %%

from luminis_mc import (
    Laser,
    SimpleMedium,
    Detector,
    SimConfig,
    AbsorptionTimeDependent,
    RayleighDebyePhaseFunction,
    Rng,
    CVec2,
    Vec3,
)
from luminis_mc import LogLevel, LaserSource
from luminis_mc import run_simulation, set_log_level
import matplotlib.pyplot as plt
import numpy as np
import time

set_log_level(LogLevel.debug)

# %%

start_time = time.time()

# Global frame of reference
m_global = Vec3(1, 0, 0)
n_global = Vec3(0, 1, 0)
s_global = Vec3(0, 0, 1)
light_speed = 299792458e-6

# Medium parameters
radius = 1.6 # in micrometers
mean_free_path = 5.8 # in micrometers
wavelength = 0.525  # in micrometers
inv_mfp = 1 / mean_free_path
mu_absortion = 0.03 * inv_mfp
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
n_photons = 100000

# Laser parameters
origin = Vec3(0, 0, 0)
polarization = CVec2(1, 0)
laser_radius = 5 * mean_free_path
laser_type = LaserSource.Gaussian

# Absortion parameters
r_size = 500 * mean_free_path
z_size = 600 * mean_free_path
dr = mean_free_path / 5
dz = mean_free_path / 5

# %%

# Initialize components
rng = Rng()
config = SimConfig(n_photons=n_photons)
laser_source = Laser(origin, s_global, n_global, m_global, polarization, wavelength, laser_radius, laser_type)
detector = Detector(origin, s_global, n_global, m_global)
phase_function = RayleighDebyePhaseFunction(wavelength, radius, nDiv, thetaMin, thetaMax)
medium = SimpleMedium(mu_absortion, mu_scattering, phase_function, mean_free_path, radius)
print("Anysotropic factor g:", phase_function.get_anisotropy_factor(rng))
medium = SimpleMedium(mu_absortion, mu_scattering, phase_function, mean_free_path, radius)
absorption = AbsorptionTimeDependent(r_size, z_size, dr, dz, dt, max_time)
medium.absorption = absorption





rng = Rng()
print("Anystropy g:", phase_function.get_anisotropy_factor(rng))

# %%

run_simulation(config, medium, detector, laser_source)

# %%

print("Recorded photons:", len(detector.recorded_photons))

x_min, x_max = 0.0, r_size
y_min, y_max = 0.0, z_size

for i in range(len(medium.absorption.time_slices)):
    abs_image = medium.absorption.get_absorption_image(n_photons, i)
    extent = [0, z_size / mean_free_path, -r_size / mean_free_path, r_size / mean_free_path]
    plt.figure(figsize=(7, 7))
    plt.imshow(abs_image, cmap="viridis", origin="lower", extent=extent, aspect='equal')
    cbar = plt.colorbar(label="Absorption Values")
    plt.xlabel("z  [1/l]")
    plt.ylabel("x  [1/l]")
    # plt.title(f"Absorption Values Grid - Time Slice {i}")
    plt.tight_layout()
    plt.show()
