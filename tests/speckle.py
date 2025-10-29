# %%

from luminis_mc import (
    Laser,
    SimpleMedium,
    Detector,
    SimConfig,
    RayleighDebyePhaseFunction,
    Rng,
    CVec2,
    Vec3,
)
from luminis_mc import LogLevel, LaserSource
from luminis_mc import run_simulation, run_simulation_parallel, set_log_level
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
n_photons = 1_000

# Laser parameters
origin = Vec3(0, 0, 0)
polarization = CVec2(1, 0)
laser_radius = 0.1 * mean_free_path
laser_type = LaserSource.Gaussian

# %%

# Initialize components
rng = Rng()
laser_source = Laser(origin, s_global, n_global, m_global, polarization, wavelength, laser_radius, laser_type)
detector = Detector(origin, s_global, n_global, m_global)
phase_function = RayleighDebyePhaseFunction(wavelength, radius, nDiv, thetaMin, thetaMax)
medium = SimpleMedium(mu_absortion, mu_scattering, phase_function, mean_free_path, radius)
print("Anysotropic factor g:", phase_function.get_anisotropy_factor(rng))

config = SimConfig(
    n_photons=n_photons,
    medium=medium,
    detector=detector,
    laser=laser_source,
)
config.n_threads = 8

# %%

run_simulation_parallel(config)
end_time = time.time()
print(f"Simulation time: {end_time - start_time:.2f} seconds")

# %%

print(f"Number of hits recorded: {detector.hits}")

min_hist_angle = 0
max_hist_angle = 180
hit_histogram_raw_data = detector.compute_events_histogram(min_hist_angle, max_hist_angle)
event_counts = np.asarray(hit_histogram_raw_data, dtype=int)

k = np.arange(len(event_counts))

plt.figure(figsize=(7,4))
plt.bar(k, event_counts, width=0.9)
plt.xlabel("Número de eventos (scatterings) antes de llegar al detector")
plt.ylabel("Conteo de fotones")
plt.tight_layout()
plt.show()


speckle = detector.compute_speckle()
plt.figure(figsize=(8, 8))
plt.imshow(np.log10(np.array(speckle.I) + 1e-20), cmap="gray", origin="lower")
plt.colorbar(label="Absorption Values")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

I = np.array(speckle.I)  # o speckle.I, según quieras comparar

# Flatten y quitar ceros o NaN
I = I[np.isfinite(I)]
I = I[I > 0]

# Normaliza a la media
mean_I = np.mean(I)
eta = I / mean_I

# Construye histograma normalizado (probability density)
bins = np.linspace(0, 10, 60)
hist, edges = np.histogram(eta, bins=bins, density=True)
centers = 0.5 * (edges[1:] + edges[:-1])

# Curva teórica e^{-η}
eta_theory = np.linspace(0, 10, 500)
p_theory = np.exp(-eta_theory)

C = np.std(I) / np.mean(I)
print(f"Speckle contrast C = {C:.3f}")

# Gráfica logarítmica (escala semilog)
plt.figure(figsize=(6, 5))
plt.semilogy(centers, hist, 'o', label='Speckle Histogram')
plt.semilogy(eta_theory, p_theory, '-', label=r'$e^{-\eta}$')
plt.xlabel(r'$\eta = I / \langle I \rangle$')
plt.ylabel('Probability Density (log scale)')
plt.title('Normalized Speckle Intensity Distribution')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.4)
plt.show()
