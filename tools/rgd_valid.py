import numpy as np

from luminis_mc import (
    RGDMedium, MieMedium, RayleighDebyeEMCPhaseFunction, MiePhaseFunction, derived_quantities
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

    dq = derived_quantities(medium, volume_fraction)

    size_parameter = dq["size_parameter"]
    anisotropy_factor = dq["anisotropy_g"]
    scattering_efficiency = dq["scattering_efficiency"]
    mean_free_path = dq["mean_free_path"]
    transport_mean_free_path = dq["transport_mean_free_path"]
    theta_max_cbs = dq["theta_coherent"]
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



def print_same_mus(rad1, rad2, volume_fraction, n_particle):
    phase1 = RayleighDebyeEMCPhaseFunction(wavelength, rad1, n_particle, n_medium, phasef_ndiv, phasef_theta_min, phasef_theta_max)
    medium1 = RGDMedium(phase1, rad1, n_particle, n_medium, wavelength)

    phase2 = MiePhaseFunction(wavelength, rad2, n_particle, n_medium, phasef_ndiv, phasef_theta_min, phasef_theta_max)
    medium2 = MieMedium(phase2, rad2, n_particle, n_medium, wavelength)

    dq1 = derived_quantities(medium1, volume_fraction)
    dq2 = derived_quantities(medium2, volume_fraction)


    reference_mus = 1 / dq1["mean_free_path"]
    relative_difference_g = (1 - dq2["anisotropy_g"]) / (1 - dq1["anisotropy_g"])
    volume_fraction2 = (4*rad2*reference_mus) / (3 * dq2["scattering_efficiency"])

    dq2 = derived_quantities(medium2, volume_fraction2)

    print(f"Radius 1: {rad1:.3f} µm -> g1: {dq1['anisotropy_g']:.3f}, mus1: {1/dq1['mean_free_path']:.3f} µm^-1")
    print(f"Radius 2: {rad2:.3f} µm -> g2: {dq2['anisotropy_g']:.3f}, mus2: {1/dq2['mean_free_path']:.3f} µm^-1")
    print(f"Relative difference in (1-g): {relative_difference_g:.3f}")
    print(f"Volume fraction for radius 2 to match mus1: {volume_fraction2:.3f}")
    print(f"Mean fre paths ls1: {dq1['mean_free_path']:.3f} µm, ls2: {dq2['mean_free_path']:.3f} µm")
    print(f"Transport mean free paths lts1: {dq1['transport_mean_free_path']:.3f} µm, lts2: {dq2['transport_mean_free_path']:.3f} µm")
    print(f"Max CBS angles theta_cbs1: {np.degrees(dq1['theta_coherent']):.4f} deg, theta_cbs2: {np.degrees(dq2['theta_coherent']):.4f} deg")
    print("-" * 30)



print_same_mus(rad1=0.035, rad2=0.100, volume_fraction=0.10, n_particle=1.59)


# print_info_particle(radius=0.035, volume_fraction=0.10, n_particle=1.59, d_theta=1/500)
# print_info_particle(radius=0.075, volume_fraction=0.10, n_particle=1.59, d_theta=1/500)
# print_info_particle(radius=0.100, volume_fraction=0.10, n_particle=1.59, d_theta=1/500)

