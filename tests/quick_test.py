# %%

from luminis_mc import (
    Laser,
    SimpleMedium,
    Detector,
    SimConfig,
    HenyeyGreensteinPhaseFunction,
    AbsortionTimeDependent,
    RayleighDebyePhaseFunction,
)
from luminis_mc import LogLevel, LaserSource
from luminis_mc import run_simulation, set_log_level
import matplotlib.pyplot as plt
import numpy as np

set_log_level(LogLevel.debug)

# %%

# Global frame of reference
n_global = [1, 0, 0]
m_global = [0, 1, 0]
s_global = [0, 0, 1]
light_speed = 299792458e-6

# Medium parameters
wavelength = 532.0
mu_absortion = 0.07
mu_scattering = 0.05
mean_free_path = 1 / (mu_absortion + mu_scattering)
radius = 0.1 * mean_free_path

print(f"Mean free path: {mean_free_path}")
print(f"Medium radius: {radius}")

# Time parameters
t_ref = mean_free_path / light_speed
dt = 0
max_time = 50 * t_ref

# Absortion parameters
r_size = 30 * mean_free_path
z_size = 50 * mean_free_path
dr = mean_free_path / 10
dz = mean_free_path / 10

# Phase function parameters
thetaMin = 0.00001
thetaMax = np.pi
nDiv = 1000
n_photons = 4000000

# Laser parameters
origin = [0, 0, 0]
polarization = [1, 0]
laser_radius = 0.1 * mean_free_path
laser_type = LaserSource.Point

# %%

# Initialize components
config = SimConfig(n_photons=n_photons)
laser_source = Laser(origin, s_global, n_global, m_global, polarization, wavelength, laser_radius, laser_type)
detector = Detector(origin, s_global, n_global, m_global)
phase_function = HenyeyGreensteinPhaseFunction(0.99)
# phase_function = RayleighDebyePhaseFunction(wavelength, radius, nDiv, thetaMin, thetaMax)
absorption = AbsortionTimeDependent(r_size, z_size, dr, dz, dt, max_time)
medium = SimpleMedium(mu_absortion, mu_scattering, phase_function, mean_free_path, radius)
medium.absorption = absorption

# %%

run_simulation(config, medium, detector, laser_source)

# %%

print("Recorded photons:", len(detector.recorded_photons))

# emited_positions = []
# for i in range(len(detector.recorded_photons)):
#     photon = detector.recorded_photons[i]
#     emited_positions.append((photon.pos[0], photon.pos[1], photon.pos[2]))

# emited_positions = np.array(emited_positions)
# plt.figure(figsize=(8, 8))
# plt.scatter(emited_positions[:, 0], emited_positions[:, 1], s=1, alpha=0.5)
# plt.xlabel("X position")
# plt.ylabel("Y position")
# plt.title("Photon Emission Positions")
# plt.grid(True)
# plt.show()

# %%

for i in range(len(medium.absorption.time_slices)):
    abs_image = medium.absorption.get_absorption_image(n_photons, i)
    plt.figure(figsize=(8, 8))
    plt.imshow(abs_image, cmap="viridis", origin="lower")
    plt.colorbar(label="Absorption Values")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Absorption Values Grid - Time Slice {i}")
    plt.show()
