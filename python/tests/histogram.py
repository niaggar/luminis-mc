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
mu_absortion = 0.03
mu_scattering = 0.05
mean_free_path = 1 / (mu_absortion + mu_scattering)
radius = 0.1 * mean_free_path

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
n_photons = 5000000

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
phase_function = HenyeyGreensteinPhaseFunction(0.1)
# phase_function = RayleighDebyePhaseFunction(wavelength, radius, nDiv, thetaMin, thetaMax)
medium = SimpleMedium(mu_absortion, mu_scattering, phase_function, mean_free_path, radius)

# %%

run_simulation(config, medium, detector, laser_source)

# %%

print("Recorded photons:", len(detector.recorded_photons))

min_hist_angle = 0
max_hist_angle = 180
hit_histogram_raw_data = detector.get_hit_histogram(min_hist_angle, max_hist_angle)
event_counts = np.asarray(hit_histogram_raw_data, dtype=int)

k = np.arange(len(event_counts))

plt.figure(figsize=(7,4))
plt.bar(k, event_counts, width=0.9)
plt.xlabel("Número de eventos (scatterings) antes de llegar al detector")
plt.ylabel("Conteo de fotones")
plt.tight_layout()
plt.show()
