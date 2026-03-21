from utils.styles import apply
import numpy as np

from luminis_mc import (
    RGDMedium, RayleighDebyeEMCPhaseFunction, MiePhaseFunction
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



RGD_EPSILON = 0.2
COLOR_VALID   = "#2CA02C"   # muted green
COLOR_INVALID = "#D62728"   # muted red
COLOR_LINE    = "#0057e7"   # steel blue for data
COLOR_LIMIT   = "#8B0000"   # dark red for the cutoff line
COLOR_GRAY   = "#6B6B6B"   # gray for auxiliary lines and text


# --------- Particle size vs Size parameter

def plot_size_parameter_vs_radius(r_min, r_max, n_particle, path):
    fig, ax = plt.subplots(figsize=(4, 4))
    size_parameters = []
    condition_2_values = []

    radius_sweep = np.linspace(r_min, r_max, 50)
    m_relative = n_particle / n_medium

    for radius in radius_sweep:
        size_parameter = k_medium * radius
        size_parameters.append(size_parameter)

        condition_2 = size_parameter * np.abs(m_relative - 1)
        condition_2_values.append(condition_2)

    ax.plot(radius_sweep, size_parameters, color=COLOR_LINE, linewidth=2, marker="o", markersize=3.5, markevery=5, zorder=4)

    condition_1_ref = np.abs(m_relative - 1)
    radius_limit = RGD_EPSILON / (k_medium * condition_1_ref)

    # ax.axhline(
    #     RGD_EPSILON,
    #     color="black",
    #     linestyle="--",
    #     linewidth=1.0,
    #     label=rf"Validity threshold: $\epsilon={RGD_EPSILON}$",
    # )
    ax.axvline(
        radius_limit,
        color=COLOR_LIMIT,
        linestyle=":",
        linewidth=1.5,
        label=rf"$a_{{max}}={radius_limit:.3f}\,\mu m$",
    )

    radius_min = float(np.min(radius_sweep))
    radius_max = float(np.max(radius_sweep))
    ax.axvspan(radius_min, min(radius_limit, radius_max), color=COLOR_VALID, alpha=0.12)
    if radius_limit < radius_max:
        ax.axvspan(max(radius_limit, radius_min), radius_max, color=COLOR_INVALID, alpha=0.08)

    plus = radius_limit * 10 / 100
    if radius_limit + plus > radius_max:
        ax.set_xlim(radius_min, radius_max)
    else:
        ax.set_xlim(radius_min, radius_limit + plus)

    ax.set_xlabel("Particle radius (µm)")
    ax.set_ylabel("Size parameter")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.legend()

    plt.tight_layout()
    plt.savefig(path)


# plot_size_parameter_vs_radius(r_min_silica, r_max_silica, n_particle_silica, f"{save_path}/size_parameter_vs_radius_silica.pdf")
# plot_size_parameter_vs_radius(r_min_bio, r_max_bio, n_particle_bio, f"{save_path}/size_parameter_vs_radius_bio.pdf")
# plot_size_parameter_vs_radius(r_min_poly, r_max_poly, n_particle_poly, f"{save_path}/size_parameter_vs_radius_poly.pdf")


# --------- Particle size vs Anisotropy factor

def plot_anisotropy_factor_vs_radius(r_min, r_max, n_particle, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    anisotropy_factors = []
    anisotropy_factors_mie = []
    condition_2_values = []


    radius_sweep = np.linspace(r_min, r_max, 50)

    for radius in radius_sweep:
        phase_mie = MiePhaseFunction(wavelength, radius, n_particle, n_medium, phasef_ndiv, phasef_theta_min, phasef_theta_max)
        anisotropy_factor_mie = phase_mie.get_anisotropy_factor()[0]
        anisotropy_factors_mie.append(anisotropy_factor_mie)

        phase = RayleighDebyeEMCPhaseFunction(wavelength, radius, n_particle, n_medium, phasef_ndiv, phasef_theta_min, phasef_theta_max)
        anisotropy_factor = phase.get_anisotropy_factor()[0]
        anisotropy_factors.append(anisotropy_factor)

        condition_2 = k_medium * radius * np.abs(n_particle / n_medium - 1)
        condition_2_values.append(condition_2)

    m_relative = n_particle / n_medium
    condition_1_ref = np.abs(m_relative - 1)
    radius_limit = RGD_EPSILON / (k_medium * condition_1_ref)

    # Vertical line for radius limit
    ax.axvline(
        radius_limit,
        color=COLOR_LIMIT,
        linestyle=":",
        linewidth=1.5,
    )


    ax.plot(radius_sweep, anisotropy_factors_mie, color=COLOR_GRAY, linewidth=2, linestyle="--", marker="s", markersize=3.5, markevery=5, zorder=4, label="Mie")
    ax.plot(radius_sweep, anisotropy_factors, color=COLOR_LINE, linewidth=2, marker="o", markersize=3.5, markevery=5, zorder=4, label="RGD")
    ax.set_xlabel("Particle radius (µm)")
    ax.set_ylabel("Anisotropy factor")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))



    # --- inset zoomed to the valid RGD region ---
    x1, x2 = r_min, min(radius_limit, r_max)
    mask = (radius_sweep >= x1) & (radius_sweep <= x2)

    # Position and size of the inset: [left, bottom, width, height] in axes fraction
    axins = ax.inset_axes((0.55, 0.08, 0.42, 0.42))

    axins.plot(radius_sweep[mask], np.array(anisotropy_factors_mie)[mask], color=COLOR_GRAY, linewidth=1.5, linestyle="--", marker="s", markersize=3, label="Mie")
    axins.plot(radius_sweep[mask], np.array(anisotropy_factors)[mask], color=COLOR_LINE, linewidth=1.5, marker="o", markersize=3, label="RGD")

    axins.set_xlim(x1, x2)
    y_vals = np.array(anisotropy_factors)[mask]
    axins.set_ylim(y_vals.min() * 0.95, y_vals.max() * 1.05)
    axins.tick_params(labelsize=7)
    axins.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axins.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    # Draw the zoom lines connecting inset to main plot
    ax.indicate_inset_zoom(axins, edgecolor="#555555", linewidth=0.8)

    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(path)


# plot_anisotropy_factor_vs_radius(r_min_silica, r_max_silica, n_particle_silica, f"{save_path}/anisotropy_factor_vs_radius_silica.pdf")
# plot_anisotropy_factor_vs_radius(r_min_bio, r_max_bio, n_particle_bio, f"{save_path}/anisotropy_factor_vs_radius_bio.pdf")
# plot_anisotropy_factor_vs_radius(r_min_poly, r_max_poly, n_particle_poly, f"{save_path}/anisotropy_factor_vs_radius_poly.pdf")



# --------- Particle size vs Mean free path

def plot_mean_free_path_vs_radius(r_min, r_max, n_particle, volume_fractions, path):
    fig, ax = plt.subplots(figsize=(4, 4))
    mean_free_paths = {}
    condition_2_values = []

    radius_sweep = np.linspace(r_min, r_max, 50)

    for radius in radius_sweep:
        phase = RayleighDebyeEMCPhaseFunction(wavelength, radius, n_particle, n_medium, phasef_ndiv, phasef_theta_min, phasef_theta_max)
        medium = RGDMedium(phase, radius, n_particle, n_medium, wavelength)

        scattering_efficiency = medium.scattering_efficiency()
        for volume_fraction in volume_fractions:
            mean_free_path = (4.0 * radius) / (3.0 * volume_fraction * scattering_efficiency)
            if volume_fraction not in mean_free_paths:
                mean_free_paths[volume_fraction] = []
            mean_free_paths[volume_fraction].append(mean_free_path)

        condition_2 = k_medium * radius * np.abs(n_particle / n_medium - 1)
        condition_2_values.append(condition_2)


    m_relative = n_particle / n_medium
    condition_1_ref = np.abs(m_relative - 1)
    radius_limit = RGD_EPSILON / (k_medium * condition_1_ref)

    # Vertical line for radius limit
    ax.axvline(
        radius_limit,
        color=COLOR_LIMIT,
        linestyle=":",
        linewidth=1.5,
    )

    for volume_fraction, mean_free_path_list in mean_free_paths.items():
        ax.plot(radius_sweep, mean_free_path_list, linewidth=2, marker="o", markersize=3.5, markevery=5, zorder=4, label=rf"$f={volume_fraction:.2f}$")
    ax.set_yscale("log")
    ax.set_xlabel("Particle radius (µm)")
    ax.set_ylabel("Mean free path (µm)")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.legend()

    radius_min = float(np.min(radius_sweep))
    radius_max = float(np.max(radius_sweep))
    plus = radius_limit * 10 / 100
    if radius_limit + plus > radius_max:
        ax.set_xlim(radius_min, radius_max)
    else:
        ax.set_xlim(radius_min, radius_limit + plus)

    plt.tight_layout()
    plt.savefig(path)

# plot_mean_free_path_vs_radius(r_min_silica, r_max_silica, n_particle_silica, volume_fraction_sweep, f"{save_path}/mean_free_path_vs_radius_silica.pdf")
# plot_mean_free_path_vs_radius(r_min_bio, r_max_bio, n_particle_bio, volume_fraction_sweep, f"{save_path}/mean_free_path_vs_radius_bio.pdf")
# plot_mean_free_path_vs_radius(r_min_poly, r_max_poly, n_particle_poly, volume_fraction_sweep, f"{save_path}/mean_free_path_vs_radius_poly.pdf")



# --------- Phase function vs Scattering angle for different particle sizes

n_samples = 5_000_000

def plot_phase_function_vs_angle(r_min, r_max, n_particle, path):
    fig, ax = plt.subplots(figsize=(6, 4))

    radius_sweep = np.linspace(r_min, r_max, 5)
    histogram_data = {}

    mus = np.random.default_rng().random(n_samples)

    for radius in radius_sweep:
        phase = RayleighDebyeEMCPhaseFunction(wavelength, radius, n_particle, n_medium, phasef_ndiv, phasef_theta_min, phasef_theta_max)
        theta_emc = np.array([phase.sample_theta(x) for x in mus])

        bins = np.linspace(phasef_theta_min, phasef_theta_max, 500)
        hist_emc, _ = np.histogram(theta_emc, bins=bins, density=True)
        histogram_data[radius] = hist_emc

    for radius in radius_sweep:
        ax.semilogy(np.linspace(phasef_theta_min, phasef_theta_max, 499), histogram_data[radius], label=rf"$a={radius:.3f}\,\mu m$", linewidth=2, marker="o", markersize=3.5, markevery=50)

    ax.set_xlabel("$\\theta$ (Degrees)")
    ax.set_ylabel("$P(\\theta)$")
    ax.legend()

    plt.tight_layout()
    plt.savefig(path)


r_max_silica = 0.126
r_max_bio = 0.327
r_max_poly = 0.063

# plot_phase_function_vs_angle(r_min_silica, r_max_silica, n_particle_silica, f"{save_path}/phase_function_vs_angle_silica.pdf")
# plot_phase_function_vs_angle(r_min_bio, r_max_bio, n_particle_bio, f"{save_path}/phase_function_vs_angle_bio.pdf")
# plot_phase_function_vs_angle(r_min_poly, r_max_poly, n_particle_poly, f"{save_path}/phase_function_vs_angle_poly.pdf")




def print_info_particle(radius, volume_fraction, n_particle):
    phase = RayleighDebyeEMCPhaseFunction(wavelength, radius, n_particle, n_medium, phasef_ndiv, phasef_theta_min, phasef_theta_max)
    medium = RGDMedium(phase, radius, n_particle, n_medium, wavelength)

    size_parameter = k_medium * radius
    anisotropy_factor = phase.get_anisotropy_factor()[0]
    scattering_efficiency = medium.scattering_efficiency()
    mean_free_path = (4.0 * radius) / (3.0 * volume_fraction * scattering_efficiency)
    transport_mean_free_path = mean_free_path / (1 - anisotropy_factor)
    theta_max_cbs = 1 / (k_medium * transport_mean_free_path)

    condition_1 = np.abs(n_particle / n_medium - 1)
    condition_2 = size_parameter * condition_1
    valid_condition_1 = condition_1 < RGD_EPSILON
    valid_condition_2 = condition_2 < RGD_EPSILON


    print(f"Radius: {radius:.3f} µm")
    print(f"Size parameter: {size_parameter:.3f}")
    print(f"Anisotropy factor: {anisotropy_factor:.3f}")
    print(f"Scattering efficiency: {scattering_efficiency:.3f}")
    print(f"Mean free path: {mean_free_path:.3f} µm")
    print(f"Transport mean free path: {transport_mean_free_path:.3f} µm")
    print(f"Max CBS angle: {theta_max_cbs:.4f} radians ({np.degrees(theta_max_cbs):.4f} degrees)")
    print(f"Condition 1 (|m-1|): {condition_1:.3f} - {'Valid' if valid_condition_1 else 'Invalid'}")
    print(f"Condition 2 (size parameter * |m-1|): {condition_2:.3f} - {'Valid' if valid_condition_2 else 'Invalid'}")
    print("-" * 30)


# # Test with a specific particle size and volume fraction
# test_radius = 0.070
# # test_volume_fraction_s = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.20, 0.30]
# test_volume_fraction_s = [0.05, 0.07, 0.10, 0.20, 0.30]
# test_n_particle = n_particle_poly
# for test_volume_fraction in test_volume_fraction_s:
#     print_info_particle(test_radius, test_volume_fraction, test_n_particle)


test_radius = 0.100
density_1 = 0.07
density_2 = 0.10
test_n_particle = n_particle_poly
print_info_particle(test_radius, density_1, test_n_particle)
print_info_particle(test_radius, density_2, test_n_particle)

# 2.0 nm -> Reference
min_depth = 2.0
max_depth = 100.0
depths_first_layers = np.linspace(min_depth, max_depth, 15)


# print_info_particle(0.350 / 2, 0.2, n_particle_poly)


# plt.show()
