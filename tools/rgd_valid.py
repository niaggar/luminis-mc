import numpy as np

from luminis_mc import (
    RGDMedium, RayleighDebyeEMCPhaseFunction, MiePhaseFunction
)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter




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

# Funcion de fase
phasef_theta_min = 0.0
phasef_theta_max = np.pi
phasef_ndiv = 100_000

def print_info_particle(radius, volume_fraction, n_particle, d_theta):
    phase = RayleighDebyeEMCPhaseFunction(wavelength, radius, n_particle, n_medium, phasef_ndiv, phasef_theta_min, phasef_theta_max)
    medium = RGDMedium(phase, radius, n_particle, n_medium, wavelength)

    size_parameter = k_medium * radius
    anisotropy_factor = phase.get_anisotropy_factor()[0]
    scattering_efficiency = medium.phase_function.scattering_efficiency()
    mean_free_path = (4.0 * radius) / (3.0 * volume_fraction * scattering_efficiency)
    transport_mean_free_path = mean_free_path / (1 - anisotropy_factor)
    theta_max_cbs = 1 / (k_medium * transport_mean_free_path)
    condition_1 = np.abs(n_particle / n_medium - 1)
    condition_2 = size_parameter * condition_1

    print(f"Radius: {radius:.3f} µm, Volume fraction: {volume_fraction:.3f}")
    print(f"Size parameter: {size_parameter:.3f}")
    print(f"Anisotropy factor: {anisotropy_factor:.3f}")
    print(f"Scattering efficiency: {scattering_efficiency:.3f}")
    print(f"Mean free path: {mean_free_path:.3f} µm")
    print(f"Transport mean free path: {transport_mean_free_path:.3f} µm")
    print(f"Max CBS angle: {theta_max_cbs:.4f} radians ({np.degrees(theta_max_cbs):.4f} degrees)")
    print(f"Number of bins for CBS: {int(np.ceil(np.degrees(theta_max_cbs) / d_theta))}")
    print(f"Condition 1 (|m-1|): {condition_1:.3f}")
    print(f"Condition 2 (size parameter * |m-1|): {condition_2:.3f}")
    print("-" * 30)


# radius_values = [0.020, 0.035, 0.055, 0.075, 0.175]
# d_thetas = [0.5/1000, 0.5/500, 1/500, 1/500, 1/500]
# for radius, d_theta in zip(radius_values, d_thetas):
#     print_info_particle(radius=radius, volume_fraction=0.10, n_particle=1.59, d_theta=d_theta)



print_info_particle(radius=0.055, volume_fraction=0.50, n_particle=1.59, d_theta=1/500)
