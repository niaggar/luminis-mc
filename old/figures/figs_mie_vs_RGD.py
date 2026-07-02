from utils.styles import apply
import numpy as np

from luminis_mc import (
    RGDMedium, MieMedium, RayleighDebyeEMCPhaseFunction, MiePhaseFunction
)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

apply(context="paper", col="single")

save_path = "/Users/niaggar/Documents/Thesis/Results"


# Things to study:
#   - Particle size vs Size parameter
#   - Particle size vs Anisotropy factor
#   - Particle size vs Mean free path
#   - Volume fraction vs Mean free path
#   - Volume fraction vs Max scattering angle for CBS
#   - Particle size vs Max scattering angle for CBS

# Units are micrometers (µm)

# RGD have two conditions for validity:
#   1. |m-1| << 1
#   2. size parameter * |m-1| << 1


n_medium = 1.33
wavelength = 0.514
k_medium = 2 * np.pi * n_medium / wavelength

# Silica particles
n_particle_silica = 1.46
r_min_silica = 0.030
r_max_silica = 0.500

# Biological particles
n_particle_bio = 1.38
r_min_bio = 0.050
r_max_bio = 0.500

# Polystyrene particles
n_particle_poly = 1.59
r_min_poly = 0.020
r_max_poly = 0.500


volume_fraction_sweep = np.linspace(0.01, 0.2, 5)

# Phase function parameters
phasef_theta_min = 0.0
phasef_theta_max = np.pi
phasef_ndiv = 100_000

# Simulation parameters
n_photons = 50_000_000

# Phase function parameters
phasef_theta_min = 0.0
phasef_theta_max = np.pi
phasef_ndiv = 500_000



def plot_phase_function(radius, n_particle, label):
    phase_mie = MiePhaseFunction(wavelength, radius, n_particle, n_medium, phasef_ndiv, phasef_theta_min, phasef_theta_max)
    phase = RayleighDebyeEMCPhaseFunction(wavelength, radius, n_particle, n_medium, phasef_ndiv, phasef_theta_min, phasef_theta_max)

    x_axis = np.linspace(0, 180, phasef_ndiv)
    y_axis_mie = [phase_mie.rho_phase_function(theta) for theta in np.radians(x_axis)]
    y_axis_rgd = [phase.rho_phase_function(theta) for theta in np.radians(x_axis)]

    plt.plot(x_axis, y_axis_mie, label=f"Mie: {label}")
    plt.plot(x_axis, y_axis_rgd, label=f"RGD: {label}", linestyle="dashed")
    plt.xlabel("Scattering angle (degrees)")
    plt.ylabel("Phase function")
    plt.title(f"Phase function for radius = {radius} µm")


def plot_scattering_matrix(radius, n_particle, label):
    phase_mie = MiePhaseFunction(wavelength, radius, n_particle, n_medium, phasef_ndiv, phasef_theta_min, phasef_theta_max)
    phase = RayleighDebyeEMCPhaseFunction(wavelength, radius, n_particle, n_medium, phasef_ndiv, phasef_theta_min, phasef_theta_max)

    medium_mie = MieMedium(phase_mie, radius, n_particle, n_medium, wavelength)
    medium = RGDMedium(phase, radius, n_particle, n_medium, wavelength)

    x_axis = np.linspace(0, 180, phasef_ndiv)
    matrix_mie = [medium_mie.scattering_matrix(theta, 0) for theta in np.radians(x_axis)]
    matrix_rgd = [medium.scattering_matrix(theta, 0) for theta in np.radians(x_axis)]

    s2_mie = np.array([s.get(0,0) for s in matrix_mie])
    s1_mie = np.array([s.get(1,1) for s in matrix_mie])

    s2_rgd = np.array([s.get(0,0) for s in matrix_rgd])
    s1_rgd = np.array([s.get(1,1) for s in matrix_rgd])

    ratio_mie = np.abs(s2_mie) / np.abs(s1_mie)
    ratio_rgd = np.abs(s2_rgd) / np.abs(s1_rgd)

    plt.plot(x_axis, ratio_mie, label=f"Mie: {label}")
    plt.plot(x_axis, ratio_rgd, label=f"RGD: {label}", linestyle="dashed")
    plt.xlabel("Scattering angle (degrees)")
    plt.ylabel("Scattering matrix element S11")
    plt.title(f"Scattering matrix element S11 for radius = {radius} µm")

    # Use log scale for y-axis
    # plt.yscale("log")



# # Plot phase functions for different particle sizes
# plt.figure(figsize=(8, 6))
# plot_phase_function(0.05, 1.01, "Biological (0.05 µm)")
# plot_phase_function(0.1, 1.01, "Biological (0.1 µm)")
# plot_phase_function(0.2, 1.01, "Biological (0.2 µm)")
# plt.legend()
# plt.xlim(0, 180)
# plt.grid()
# plt.tight_layout()


plt.figure(figsize=(8, 6))
plot_scattering_matrix(0.05, 1.01, "Biological (0.05 µm)")
plot_scattering_matrix(0.1, 1.01, " Biological (0.1 µm)")
plot_scattering_matrix(0.2, 1.01, "Biological (0.2 µm)")
plt.legend()
plt.xlim(0, 180)
plt.grid()
plt.tight_layout()

plt.show()
