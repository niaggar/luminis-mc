# %%

from luminis_mc import (
    Laser,
    SimpleMedium,
    Detector,
    SimConfig,
    HenyeyGreensteinPhaseFunction,
    AbsortionTimeDependent,
    RayleighDebyeEMCPhaseFunction,
    Rng,
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
n_global = [1, 0, 0]
m_global = [0, 1, 0]
s_global = [0, 0, 1]
light_speed = 299792458e-6

# Medium parameters
radius = 0.1 # in micrometers
mean_free_path = 2.8 # in micrometers
wavelength = 0.525  # in micrometers
inv_mfp = 1 / mean_free_path
mu_absortion = 0.02 * inv_mfp
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
n_photons = 1000000

# Laser parameters
origin = [0, 0, 0]
polarization = [1, 0] # linear polarization
# polarization = [1/np.sqrt(2), (0-1j)/np.sqrt(2)] # circular polarization
laser_radius = 0.1 * mean_free_path
laser_type = LaserSource.Point

# %%

# Initialize components
rng = Rng()
config = SimConfig(n_photons=n_photons)
laser_source = Laser(origin, s_global, n_global, m_global, polarization, wavelength, laser_radius, laser_type)
detector = Detector(origin, s_global, n_global, m_global)
phase_function = RayleighDebyeEMCPhaseFunction(wavelength, radius, nDiv, thetaMin, thetaMax)
medium = SimpleMedium(mu_absortion, mu_scattering, phase_function, mean_free_path, radius)
print("Anysotropic factor g:", phase_function.get_anisotropy_factor(rng))

# %%

run_simulation(config, medium, detector, laser_source)
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


speckle = detector.compute_spatial_intensity(max_hist_angle)
plt.figure(figsize=(8, 8))
plt.imshow(np.log10(np.array(speckle.I) + 1e-20), cmap="gray", origin="lower")
plt.colorbar(label="Absorption Values")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

plt.figure(figsize=(8, 8))
plt.imshow(np.log10(np.array(speckle.Ix) + 1e-20), cmap="gray", origin="lower")
plt.colorbar(label="Absorption Values")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

plt.figure(figsize=(8, 8))
plt.imshow(np.log10(np.array(speckle.Iy) + 1e-20), cmap="gray", origin="lower")
plt.colorbar(label="Absorption Values")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# --- Rebin correcto: θ-uniforme -> μ-uniforme (ponderado por ΔΩ) ---
def rebin_theta_to_mu(Z, theta_max, n_mu):
    """
    Z: array (n_theta, n_phi) en bins uniformes en θ
    Devuelve: Z_mu (n_mu, n_phi) en bins uniformes en μ=cosθ,
              y bordes de θ equivalentes para pcolormesh.
    """
    n_theta, n_phi = Z.shape
    th_edges = np.linspace(0.0, theta_max, n_theta + 1)
    mu_edges = np.linspace(np.cos(theta_max), 1.0, n_mu + 1)

    # centros y pesos por fila (área sólida por fila ∝ Δcosθ; Δφ cancelará)
    mu_centers = 0.5 * (np.cos(th_edges[:-1]) + np.cos(th_edges[1:]))
    w_rows = (np.cos(th_edges[:-1]) - np.cos(th_edges[1:]))  # > 0

    # asigna cada fila θ a un bin μ por su centro
    i_mu = np.digitize(mu_centers, mu_edges) - 1
    i_mu = np.clip(i_mu, 0, n_mu - 1)

    Zmu = np.zeros((n_mu, n_phi), dtype=float)
    W   = np.zeros(n_mu, dtype=float)
    for r in range(n_theta):
        Zmu[i_mu[r], :] += Z[r, :] * w_rows[r]
        W[i_mu[r]]      += w_rows[r]
    W[W == 0] = 1.0
    Zmu /= W[:, None]

    # bordes en θ correspondientes a esos μ-bins (para el plot polar)
    th_edges_mu = np.arccos(mu_edges)
    return Zmu, th_edges_mu

# --- Plot limpio en ejes polares usando la malla μ-uniforme ---
def plot_ai(ai):
    Ix_mu, th_edges_mu = rebin_theta_to_mu(np.asarray(ai.Ix, float), ai.theta_max, 256)
    Iy_mu, _           = rebin_theta_to_mu(np.asarray(ai.Iy, float), ai.theta_max, 256)
    I_mu,  _           = rebin_theta_to_mu(np.asarray(ai.I,  float), ai.theta_max, 256)

    # Ix_mu = np.asarray(ai.Ix, float)
    # Iy_mu = np.asarray(ai.Iy, float)
    # I_mu  = np.asarray(ai.I,  float)

    # th_edges_mu = np.linspace(0.0, ai.theta_max, ai.N_theta + 1)
    ph_edges = np.linspace(0.0, ai.phi_max, ai.N_phi + 1)
    PH, TH = np.meshgrid(ph_edges, th_edges_mu)

    def show(Z, title):
        fig = plt.figure(figsize=(5,5), dpi=150)
        ax = fig.add_subplot(111, projection="polar")
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_rlim(0, ai.theta_max)
        ax.grid(False); ax.set_xticks([]); ax.set_yticks([])

        im = ax.pcolormesh(PH, TH, Z, shading="auto")

        cbar = fig.colorbar(im, ax=ax, pad=0.08)
        cbar.set_label("Intensity / sr")
        ax.set_title(title, pad=8)
        plt.show()

    show(I_mu,  "I(θ, φ)")
    show(Ix_mu, "Ix(θ, φ)")
    show(Iy_mu, "Iy(θ, φ)")

# --- Ejemplo de uso ---
max_grad = 60
max_rad = np.radians(max_grad)
angular_intensity = detector.compute_angular_intensity(max_rad, 2*np.pi)
plot_ai(angular_intensity)   # usa log=True si quieres escala logarítmica
