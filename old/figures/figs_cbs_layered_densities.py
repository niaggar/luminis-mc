from utils.loaders import load_sweep
from utils.styles import apply
import utils.figures as figures
import utils.cbs as cbs

from luminis_mc import MiePhaseFunction

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

apply(context="paper", col="single")
save_path = "/Users/niaggar/Documents/Thesis/Results"






folder_circular = "data/layers/cbs_layers_scaled_highdetail"
data = {}
sweep_data = load_sweep(folder_circular)

N_div_layer = 3
min_depth = 0.2
max_depth = 4.0
depths_first_layers = np.linspace(min_depth, max_depth, N_div_layer)


reference_names = [
    "reference_0.10",
    "reference_0.70",
]
layered_names = [
    f"{i}_depth_{n_layers:.2f}" for i, n_layers in enumerate(depths_first_layers)
]

print(layered_names)

radius_real_a = 0.15
radius_real_b = 1.0


f_scale = 50
theta_degrees = np.linspace(0, 30, 600) / f_scale
theta_mrad = theta_degrees * (1000 * np.pi / 180)
phi_degrees = np.linspace(0, 360, 60)

N_PHOTONS_CIRCULAR = 200_000_000
t_max = 50.0
d_time = 5.0
n_time_bins = int(t_max / d_time)

for run_name, result_loader in sweep_data.items():
    radius = result_loader.params["radius_um"]
    n_particle = result_loader.params["n_particle"]
    n_medium = result_loader.params["n_medium"]
    m_relative = result_loader.params["m_relative"]

    wavelength = result_loader.params["wavelength_um"]

    mean_free_path_a = 0
    mean_free_path_b = 0
    transport_mean_free_path_a = 0
    transport_mean_free_path_b = 0
    volume_fraction_a = 0
    volume_fraction_b = 0
    if "mean_free_path_ls_um_a" in result_loader.params:
        mean_free_path_a = result_loader.params["mean_free_path_ls_um_a"] * f_scale
        mean_free_path_b = result_loader.params["mean_free_path_ls_um_b"] * f_scale
        transport_mean_free_path_a = result_loader.params["transport_mean_free_path_lstar_um_a"] * f_scale
        transport_mean_free_path_b = result_loader.params["transport_mean_free_path_lstar_um_b"] * f_scale
        volume_fraction_a = result_loader.params["volume_fraction_a"] / f_scale
        volume_fraction_b = result_loader.params["volume_fraction_b"] / f_scale
    else:
        mean_free_path_a = result_loader.params["mean_free_path_ls_um"] * f_scale
        transport_mean_free_path_a = result_loader.params["transport_mean_free_path_lstar_um"] * f_scale
        volume_fraction_a = result_loader.params["volume_fraction"] / f_scale

    scattering_efficiency = result_loader.params["scattering_efficiency"]
    anisotropy_factor = result_loader.params["anisotropy_factor"]
    size_parameter = result_loader.params["size_parameter"]
    condition_1 = result_loader.params["condition_1"]
    condition_2 = result_loader.params["condition_2"]

    k = (2 * np.pi / wavelength)

    theta_coherent_a = 1 / (k * transport_mean_free_path_a) * 180 / np.pi  # Convertir a grados
    theta_coherent_milirad_a = theta_coherent_a * (1000 * np.pi / 180)
    theta_coherent_b = 0
    theta_coherent_milirad_b = 0
    if transport_mean_free_path_b > 0:
        theta_coherent_b = 1 / (k * transport_mean_free_path_b) * 180 / np.pi  # Convertir a grados
        theta_coherent_milirad_b = theta_coherent_b * (1000 * np.pi / 180)

    data[run_name] = {
        "volume_fraction_a": volume_fraction_a,
        "volume_fraction_b": volume_fraction_b,
        "anisotropy": anisotropy_factor,
        "mean_free_path_a": mean_free_path_a,
        "mean_free_path_b": mean_free_path_b,
        "transport_mean_free_path_a": transport_mean_free_path_a,
        "transport_mean_free_path_b": transport_mean_free_path_b,
        "scattering_efficiency": scattering_efficiency,
        "radius": radius,
        "size_parameter": size_parameter,
        "theta_coherent_a": theta_coherent_a,
        "theta_coherent_b": theta_coherent_b,
        "theta_coherent_mrad_a": theta_coherent_milirad_a,
        "theta_coherent_mrad_b": theta_coherent_milirad_b,
        "ff_circular_timed": [{
            "coherent": {
                "s0": result_loader.derived(f"farfield_cbs_timed_{t}/coherent/s0"),
                "s1": result_loader.derived(f"farfield_cbs_timed_{t}/coherent/s1"),
                "s2": result_loader.derived(f"farfield_cbs_timed_{t}/coherent/s2"),
                "s3": result_loader.derived(f"farfield_cbs_timed_{t}/coherent/s3"),
            },
            "incoherent": {
                "s0": result_loader.derived(f"farfield_cbs_timed_{t}/incoherent/s0"),
                "s1": result_loader.derived(f"farfield_cbs_timed_{t}/incoherent/s1"),
                "s2": result_loader.derived(f"farfield_cbs_timed_{t}/incoherent/s2"),
                "s3": result_loader.derived(f"farfield_cbs_timed_{t}/incoherent/s3"),
            },
            "enhancement": {},
        } for t in range(n_time_bins)],
    }


    eps = 1e-30
    for t in range(n_time_bins):
        s0_coh = data[run_name]["ff_circular_timed"][t]["coherent"]["s0"] + eps
        s1_coh = data[run_name]["ff_circular_timed"][t]["coherent"]["s1"] + eps
        s2_coh = data[run_name]["ff_circular_timed"][t]["coherent"]["s2"] + eps
        s3_coh = data[run_name]["ff_circular_timed"][t]["coherent"]["s3"] + eps

        s0_inc = data[run_name]["ff_circular_timed"][t]["incoherent"]["s0"] + eps
        s1_inc = data[run_name]["ff_circular_timed"][t]["incoherent"]["s1"] + eps
        s2_inc = data[run_name]["ff_circular_timed"][t]["incoherent"]["s2"] + eps
        s3_inc = data[run_name]["ff_circular_timed"][t]["incoherent"]["s3"] + eps

        data[run_name]["ff_circular_timed"][t]["coherent"]["Ico"] = (s0_coh - s3_coh) / 2
        data[run_name]["ff_circular_timed"][t]["coherent"]["Icross"] = (s0_coh + s3_coh) / 2
        data[run_name]["ff_circular_timed"][t]["coherent"]["Ix"] = (s0_inc + s1_inc) / 2
        data[run_name]["ff_circular_timed"][t]["coherent"]["Iy"] = (s0_inc - s1_inc) / 2
        data[run_name]["ff_circular_timed"][t]["coherent"]["Itotal"] = s0_inc

        data[run_name]["ff_circular_timed"][t]["incoherent"]["Ico"] = (s0_inc - s3_inc) / 2
        data[run_name]["ff_circular_timed"][t]["incoherent"]["Icross"] = (s0_inc + s3_inc) / 2
        data[run_name]["ff_circular_timed"][t]["incoherent"]["Ix"] = (s0_inc + s1_inc) / 2
        data[run_name]["ff_circular_timed"][t]["incoherent"]["Iy"] = (s0_inc - s1_inc) / 2
        data[run_name]["ff_circular_timed"][t]["incoherent"]["Itotal"] = s0_inc

        data[run_name]["ff_circular_timed"][t]["enhancement"]["Ico"] = (
            data[run_name]["ff_circular_timed"][t]["coherent"]["Ico"] /
            data[run_name]["ff_circular_timed"][t]["incoherent"]["Ico"]
        )
        data[run_name]["ff_circular_timed"][t]["enhancement"]["Icross"] = (
            data[run_name]["ff_circular_timed"][t]["coherent"]["Icross"] /
            data[run_name]["ff_circular_timed"][t]["incoherent"]["Icross"]
        )
        data[run_name]["ff_circular_timed"][t]["enhancement"]["Ix"] = (
            data[run_name]["ff_circular_timed"][t]["coherent"]["Ix"] /
            data[run_name]["ff_circular_timed"][t]["incoherent"]["Ix"]
        )
        data[run_name]["ff_circular_timed"][t]["enhancement"]["Iy"] = (
            data[run_name]["ff_circular_timed"][t]["coherent"]["Iy"] /
            data[run_name]["ff_circular_timed"][t]["incoherent"]["Iy"]
        )
        data[run_name]["ff_circular_timed"][t]["enhancement"]["Itotal"] = (
            data[run_name]["ff_circular_timed"][t]["coherent"]["Itotal"] /
            data[run_name]["ff_circular_timed"][t]["incoherent"]["Itotal"]
        )


LAYERED_COLORS = {
    2:  "#6B6B6B",   # dark gray
    5:  "#6B6B6B",   # same — dash pattern does the work
    10: "#6B6B6B",
}
LAYERED_DASHES = {
    2:  "-.",   # dash-dot
    5:  "--",   # dashed
    10: ":",   # dotted
}
REF_COLORS = {
    "small": "#0057e7",
    "large": "#d62d20",
}



# ------ Radial profiles Circular
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# reference single-particle curves - 0.15
Ico = np.mean(data[reference_names[0]]["ff_circular_timed"][0]["enhancement"]["Ico"], axis=1)
Icros = np.mean(data[reference_names[0]]["ff_circular_timed"][0]["enhancement"]["Icross"], axis=1)

theta_coherent_milirad_a = data[reference_names[0]]["theta_coherent_mrad_a"]
ax_left.axvline(theta_coherent_milirad_a, color=REF_COLORS["small"], linestyle="--", alpha=0.7)
ax_right.axvline(theta_coherent_milirad_a, color=REF_COLORS["small"], linestyle="--", alpha=0.7)

ax_left.plot(theta_mrad, Ico, color=REF_COLORS["small"], linestyle="-", label="Single particle (a=0.15)", linewidth=2)
ax_right.plot(theta_mrad, Icros, color=REF_COLORS["small"], linestyle="-", label="Single particle (a=0.15)", linewidth=2)

# reference single-particle curves - 1.0
Ico = np.mean(data[reference_names[1]]["ff_circular_timed"][0]["enhancement"]["Ico"], axis=1)
Icros = np.mean(data[reference_names[1]]["ff_circular_timed"][0]["enhancement"]["Icross"], axis=1)

theta_coherent_milirad_b = data[reference_names[1]]["theta_coherent_mrad_b"]
ax_left.axvline(theta_coherent_milirad_b, color=REF_COLORS["large"], linestyle="--", alpha=0.7)
ax_right.axvline(theta_coherent_milirad_b, color=REF_COLORS["large"], linestyle="--", alpha=0.7)

ax_left.plot(theta_mrad, Ico, color=REF_COLORS["large"], linestyle="-", label="Single particle (a=1.0)", linewidth=2)
ax_right.plot(theta_mrad, Icros, color=REF_COLORS["large"], linestyle="-", label="Single particle (a=1.0)", linewidth=2)

# layered curves
for layered_name in layered_names:
    n_layers = layered_name

    Ico = np.mean(data[layered_name]["ff_circular_timed"][0]["enhancement"]["Ico"], axis=1)
    ax_left.plot(theta_mrad, Ico, label=f"Layered (n={n_layers})", color="#6B6B6B", linestyle=":", linewidth=1.5)
    Icros = np.mean(data[layered_name]["ff_circular_timed"][0]["enhancement"]["Icross"], axis=1)
    ax_right.plot(theta_mrad, Icros, label=f"Layered (n={n_layers})", color="#6B6B6B", linestyle=":", linewidth=1.5)


ax_left.set_title("Co-polarized")
ax_left.set_xlabel("Scattering angle (mrad)")
ax_left.set_ylabel("Enhancement factor")

ax_right.set_title("Cross-polarized")
ax_right.set_xlabel("Scattering angle (mrad)")
ax_right.legend()

# MAX_MRAD = 100
# ax_left.set_xlim(0, MAX_MRAD)
# ax_right.set_xlim(0, MAX_MRAD)

plt.tight_layout()
plt.savefig(f"{save_path}/quantic/cbs-rgd-layered-enhancement-profiles.png", dpi=300)




# ------ Time evolution
# MAX_MRAD = 100
# T_indexes = [0, 3, 6, 16]
# fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)

# for i, T_index in enumerate(T_indexes):
#     ax = axes[i // 2, i % 2]

#     for layered_name in layered_names:
#         n_layers = data_circular[layered_name]["n_depth_layer"]

#         Ico = data_circular[layered_name]["ff_timed"][T_index]["enhancement"]["Ico"]
#         ax.plot(theta_mrad, Ico, label=f"Layered (n={n_layers})", color=LAYERED_COLORS[n_layers], linestyle=LAYERED_DASHES[n_layers], linewidth=1.5)
    
#     Ico = data_circular["0005_n_radius_0.15"]["ff_timed"][T_index]["enhancement"]["Ico"]
#     ax.plot(theta_mrad, Ico, color=REF_COLORS["small"], linestyle="-", label="Single particle (a=0.15)", linewidth=2)

#     Ico = data_circular["0006_n_radius_1.0"]["ff_timed"][T_index]["enhancement"]["Ico"]
#     ax.plot(theta_mrad, Ico, color=REF_COLORS["large"], linestyle="-", label="Single particle (a=1.0)", linewidth=2)

#     t_min = (T_index) * dt
#     t_max = (T_index + 1) * dt

#     ax.set_title(fr"$t \in [{t_min:.0f}, {t_max:.0f}]\,\ell/c$")
    
#     # if is the last subplot, add the legend
#     if i == len(T_indexes) - 1:
#         ax.legend(loc="upper right")

#     # if are the last row show the x-axis labels
#     if i // 2 == 1:
#         ax.set_xlabel("Scattering angle (mrad)")

#     # if are the first column, show the y-axis labels
#     if i % 2 == 0:
#         ax.set_ylabel("Enhancement factor")

#     ax.set_xlim(0, MAX_MRAD)
# plt.tight_layout()
# plt.savefig(f"{save_path}/cbs-mie-layered-enhancement-profiles-timed.png", dpi=300)


# fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)

# for i, T_index in enumerate(T_indexes):
#     ax = axes[i // 2, i % 2]

#     for layered_name in layered_names:
#         n_layers = data_circular[layered_name]["n_depth_layer"]

#         Ico = data_circular[layered_name]["ff_timed"][T_index]["coherent"]["Ico"]
#         ax.plot(theta_mrad, Ico, label=f"Layered (n={n_layers})", color=LAYERED_COLORS[n_layers], linestyle=LAYERED_DASHES[n_layers], linewidth=1.5)
    
#     Ico = data_circular["0005_n_radius_0.15"]["ff_timed"][T_index]["coherent"]["Ico"]
#     ax.plot(theta_mrad, Ico, color=REF_COLORS["small"], linestyle="-", label="Single particle (a=0.15)", linewidth=2)

#     Ico = data_circular["0006_n_radius_1.0"]["ff_timed"][T_index]["coherent"]["Ico"]
#     ax.plot(theta_mrad, Ico, color=REF_COLORS["large"], linestyle="-", label="Single particle (a=1.0)", linewidth=2)

#     t_min = (T_index) * dt
#     t_max = (T_index + 1) * dt

#     ax.set_title(fr"$t \in [{t_min:.0f}, {t_max:.0f}]\,\ell/c$")
    
#     # if is the last subplot, add the legend
#     if i == len(T_indexes) - 1:
#         ax.legend(loc="upper right")

#     # if are the last row show the x-axis labels
#     if i // 2 == 1:
#         ax.set_xlabel("Scattering angle (mrad)")

#     # if are the first column, show the y-axis labels
#     if i % 2 == 0:
#         ax.set_ylabel("Co-polarized intensity (a.u.)")

#     ax.set_xlim(0, MAX_MRAD)
# plt.tight_layout()
# plt.savefig(f"{save_path}/cbs-mie-layered-coherent-profiles-timed.png", dpi=300)

# def extract_boundary(enhancement_matrix, theta_mrad, threshold=1.5, smooth_sigma=1.5):
#     theta_boundary = np.full(enhancement_matrix.shape[0], np.nan)

#     for it in range(enhancement_matrix.shape[0]):
#         profile = enhancement_matrix[it, :]          # E(theta) at this time
#         mask    = profile > threshold
#         if mask.any():
#             # largest theta still above threshold
#             theta_boundary[it] = theta_mrad[np.where(mask)[0][-1]]

#     # Smooth to remove shot-noise jitter
#     valid = ~np.isnan(theta_boundary)
#     theta_boundary[valid] = gaussian_filter1d(
#         theta_boundary[valid], sigma=smooth_sigma
#     )
#     return theta_boundary


# # ------ Time evolution heatmaps
# MAX_MRAD = 100
# VMIN, VMAX = 1.0, 2.0

# norm = plt.matplotlib.colors.Normalize(vmin=VMIN, vmax=VMAX)
# cmap = "RdYlBu_r" 

# time_axis = np.arange(N_t_max) * dt
# boundaries = {}

# fig, axes = plt.subplots(1, 5, figsize=(15, 3), sharex=True, sharey=True, gridspec_kw={"wspace": 0.04})

# enhancement_matrix = np.array([data_circular["0005_n_radius_0.15"]["ff_timed"][it]["enhancement"]["Ico"] for it in range(N_t_max)])
# im = axes[0].imshow(enhancement_matrix.T, aspect="auto", origin="lower", extent=[0, N_t_max * dt, theta_mrad[0], theta_mrad[-1]], norm=norm, cmap=cmap)
# axes[0].set_title(r"Single particle ($a=0.15$)")
# axes[0].set_xlabel(r"Time $[\ell/c]$")
# axes[0].set_ylabel(r"Scattering angle (mrad)")
# axes[0].set_ylim(0, MAX_MRAD)

# boundaries["0005_n_radius_0.15"] = extract_boundary(enhancement_matrix, theta_mrad)
# axes[0].plot(time_axis, boundaries["0005_n_radius_0.15"], color="white", linestyle="-", label="Boundary", linewidth=2)

# enhancement_matrix = np.array([data_circular["0006_n_radius_1.0"]["ff_timed"][it]["enhancement"]["Ico"] for it in range(N_t_max)])
# im = axes[1].imshow(enhancement_matrix.T, aspect="auto", origin="lower", extent=[0, N_t_max * dt, theta_mrad[0], theta_mrad[-1]], norm=norm, cmap=cmap)
# axes[1].set_title(r"Single particle ($a=1.0$)")
# axes[1].set_xlabel(r"Time $[\ell/c]$")
# axes[1].set_ylim(0, MAX_MRAD)

# boundaries["0006_n_radius_1.0"] = extract_boundary(enhancement_matrix, theta_mrad)
# axes[1].plot(time_axis, boundaries["0006_n_radius_1.0"], color="white", linestyle="-", label="Boundary", linewidth=2)

# for i, layered_name in enumerate(layered_names):
#     n_layers = data_circular[layered_name]["n_depth_layer"]

#     enhancement_matrix = np.array([data_circular[layered_name]["ff_timed"][it]["enhancement"]["Ico"] for it in range(N_t_max)])
    
#     im = axes[i + 2].imshow(enhancement_matrix.T, aspect="auto", origin="lower", extent=[0, N_t_max * dt, theta_mrad[0], theta_mrad[-1]], norm=norm, cmap=cmap)
#     axes[i + 2].set_title(rf"Layered ($n={n_layers}$)")
#     axes[i + 2].set_xlabel(r"Time $[\ell/c]$")
#     axes[i + 2].set_ylim(0, MAX_MRAD)

#     boundaries[layered_name] = extract_boundary(enhancement_matrix, theta_mrad)
#     axes[i + 2].plot(time_axis, boundaries[layered_name], color="white", linestyle="-", label="Boundary", linewidth=2)

# cbar = fig.colorbar(
#     im, ax=axes,
#     orientation="vertical",
#     fraction=0.012,
#     pad=0.01,
#     extend="max"        # arrow at top signals values can exceed VMAX (noisy data)
# )
# cbar.set_label(r"Enhancement factor $E(\theta,\,t)$", fontsize=9)
# cbar.ax.tick_params(labelsize=8.5)
# cbar.set_ticks([1.0, 1.25, 1.5, 1.75, 2.0])

# plt.tight_layout()
# plt.savefig(f"{save_path}/cbs-mie-layered-enhancement-heatmaps.png", dpi=300)



# fig, ax = plt.subplots(figsize=(6, 4))

# for run_name, boundary in boundaries.items():
#     label = None
#     color = None
#     line_style = None
#     line_width = 1.5
#     if run_name == "0005_n_radius_0.15":
#         label = "Single particle (a=0.15)"
#         color = REF_COLORS["small"]
#         line_style = "-"
#         line_width = 2
#     elif run_name == "0006_n_radius_1.0":
#         label = "Single particle (a=1.0)"
#         color = REF_COLORS["large"]
#         line_style = "-"
#         line_width = 2
#     else:
#         n_layers = data_circular[run_name]["n_depth_layer"]
#         label = f"Layered ($n={n_layers}$)"
#         color = LAYERED_COLORS[n_layers]
#         line_style = LAYERED_DASHES[n_layers]

#     ax.plot(time_axis, boundary, label=label, color=color, linestyle=line_style, linewidth=line_width)

# ax.set_xlabel(r"Time $[\ell/c]$")
# ax.set_ylabel(r"Boundary angle $\theta_b$ (mrad)")
# ax.legend()
# ax.set_xlim(0, N_t_max * dt)
# ax.set_ylim(0, MAX_MRAD)
# plt.tight_layout()
# plt.savefig(f"{save_path}/cbs-mie-layered-enhancement-boundaries.png", dpi=300)






# def boundary_model(t, theta0, tau, alpha):
#     """Power-law decay: theta*(t) = theta0 * (t/tau)^(-alpha)"""
#     return theta0 * (t / tau) ** (-alpha)

# # Fit — skip t=0 to avoid division by zero
# t_fit = time_axis[2:]
# for label, curve in boundaries.items():
#     b_fit = curve[2:]
#     valid = ~np.isnan(b_fit) & (b_fit > 0)
#     try:
#         popt, _ = curve_fit(boundary_model, t_fit[valid], b_fit[valid],
#                             p0=[80, 5, 0.8], maxfev=5000)
#         print(f"{label}: theta0={popt[0]:.1f}, tau={popt[1]:.2f}, alpha={popt[2]:.3f}")
#     except Exception as e:
#         print(f"{label}: fit failed — {e}")




# ------ Show all figures
plt.show()