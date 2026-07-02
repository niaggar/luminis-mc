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



base_names = [
    "0005_n_radius_0.15",
    "0006_n_radius_1.0",
]

folder_circular = "09Mar26/cbs_sweep_layered"
data_circular = {}
sweep_data_circular = load_sweep(folder_circular)
n_layers_sweep = [2, 5, 10, 20, 30]

radius_real_a = 0.15
radius_real_b = 1.0

theta_degrees_circular = np.linspace(0, 45, 500)
phi_degrees_circular = np.linspace(0, 360, 1)
run_names_circular = sorted(sweep_data_circular.keys())

print(run_names_circular)

N_PHOTONS_CIRCULAR = 2_500_000_000
dt = 2.0
N_t_max = 20

for run_name, result_loader in sweep_data_circular.items():
    wavelength_real = result_loader.params["wavelength_real"]
    n_particle_real = result_loader.params["n_particle_real"]
    n_medium_real = result_loader.params["n_medium_real"]
    phasef_theta_min = result_loader.params["phasef_theta_min"]
    phasef_theta_max = result_loader.params["phasef_theta_max"]
    phasef_ndiv = result_loader.params["phasef_ndiv"]

    phase_a = MiePhaseFunction(wavelength_real, radius_real_a, n_particle_real, n_medium_real, phasef_ndiv, phasef_theta_min, phasef_theta_max)
    phase_b = MiePhaseFunction(wavelength_real, radius_real_b, n_particle_real, n_medium_real, phasef_ndiv, phasef_theta_min, phasef_theta_max)
    anysotropy_a = phase_a.get_anisotropy_factor()
    anysotropy_b = phase_b.get_anisotropy_factor()


    transport_mean_free_path_a = result_loader.params["mean_free_path_sim"] / (1 - anysotropy_a[0])
    transport_mean_free_path_b = result_loader.params["mean_free_path_sim"] / (1 - anysotropy_b[0])
    theta_coherent_a = wavelength_real / (2 * np.pi * transport_mean_free_path_a) * 180 / np.pi  # Convertir a grados
    theta_coherent_b = wavelength_real / (2 * np.pi * transport_mean_free_path_b) * 180 / np.pi  # Convertir a grados
    size_parameter_a = 2 * np.pi * radius_real_a / wavelength_real
    size_parameter_b = 2 * np.pi * radius_real_b / wavelength_real
    theta_coherent_milirad_a = theta_coherent_a * (1000 * np.pi / 180)
    theta_coherent_milirad_b = theta_coherent_b * (1000 * np.pi / 180)

    ff_timed_sensor_params = result_loader.sensor_meta("farfieldcbs_timed")
    N_t = ff_timed_sensor_params["N_t"]


    n_depth_layer = None
    if "n_depth_layer" in result_loader.params:
        n_depth_layer = result_loader.params["n_depth_layer"]

    
    single = None
    if "radius_real" in result_loader.params:
        single = result_loader.params["radius_real"]
    

    data_circular[run_name] = {
        "n_depth_layer": n_depth_layer,
        "anisotropy_a": anysotropy_a[0],
        "anisotropy_b": anysotropy_b[0],
        "mean_free_path": result_loader.params["mean_free_path_sim"],
        "radius_a": radius_real_a,
        "radius_b": radius_real_b,
        "single_particle": single,
        "size_parameter_a": size_parameter_a,
        "size_parameter_b": size_parameter_b,
        "theta_coherent_a": theta_coherent_a,
        "theta_coherent_b": theta_coherent_b,
        "theta_coherent_mrad_a": theta_coherent_milirad_a,
        "theta_coherent_mrad_b": theta_coherent_milirad_b,
        "ff": {
            "coherent": {
                "s0": np.mean(result_loader.derived("farfieldcbs_total/coherent/s0"), axis=1) / N_PHOTONS_CIRCULAR,
                "s1": np.mean(result_loader.derived("farfieldcbs_total/coherent/s1"), axis=1) / N_PHOTONS_CIRCULAR,
                "s2": np.mean(result_loader.derived("farfieldcbs_total/coherent/s2"), axis=1) / N_PHOTONS_CIRCULAR,
                "s3": np.mean(result_loader.derived("farfieldcbs_total/coherent/s3"), axis=1) / N_PHOTONS_CIRCULAR,
            },
            "incoherent": {
                "s0": np.mean(result_loader.derived("farfieldcbs_total/incoherent/s0"), axis=1) / N_PHOTONS_CIRCULAR,
                "s1": np.mean(result_loader.derived("farfieldcbs_total/incoherent/s1"), axis=1) / N_PHOTONS_CIRCULAR,
                "s2": np.mean(result_loader.derived("farfieldcbs_total/incoherent/s2"), axis=1) / N_PHOTONS_CIRCULAR,
                "s3": np.mean(result_loader.derived("farfieldcbs_total/incoherent/s3"), axis=1) / N_PHOTONS_CIRCULAR,
            },
            "enhancement": {},
        },
        "ff_timed": [
            {
                "time": it * dt,
                "coherent": {
                    "s0": np.mean(result_loader.derived(f"farfieldcbs_timed/coherent/t{it}_s0"), axis=1) / N_PHOTONS_CIRCULAR,
                    "s1": np.mean(result_loader.derived(f"farfieldcbs_timed/coherent/t{it}_s1"), axis=1) / N_PHOTONS_CIRCULAR,
                    "s2": np.mean(result_loader.derived(f"farfieldcbs_timed/coherent/t{it}_s2"), axis=1) / N_PHOTONS_CIRCULAR,
                    "s3": np.mean(result_loader.derived(f"farfieldcbs_timed/coherent/t{it}_s3"), axis=1) / N_PHOTONS_CIRCULAR,
                },
                "incoherent": {
                    "s0": np.mean(result_loader.derived(f"farfieldcbs_timed/incoherent/t{it}_s0"), axis=1) / N_PHOTONS_CIRCULAR,
                    "s1": np.mean(result_loader.derived(f"farfieldcbs_timed/incoherent/t{it}_s1"), axis=1) / N_PHOTONS_CIRCULAR,
                    "s2": np.mean(result_loader.derived(f"farfieldcbs_timed/incoherent/t{it}_s2"), axis=1) / N_PHOTONS_CIRCULAR,
                    "s3": np.mean(result_loader.derived(f"farfieldcbs_timed/incoherent/t{it}_s3"), axis=1) / N_PHOTONS_CIRCULAR,
                },
                "enhancement": {},
            } for it in range(N_t)
        ]
    }

    data_circular[run_name]["ff"]["coherent"]["Ico"] = (data_circular[run_name]["ff"]["coherent"]["s0"] - data_circular[run_name]["ff"]["coherent"]["s3"]) / 2
    data_circular[run_name]["ff"]["coherent"]["Icross"] = (data_circular[run_name]["ff"]["coherent"]["s0"] + data_circular[run_name]["ff"]["coherent"]["s3"]) / 2
    data_circular[run_name]["ff"]["coherent"]["Itotal"] = data_circular[run_name]["ff"]["coherent"]["Ico"] + data_circular[run_name]["ff"]["coherent"]["Icross"]
    data_circular[run_name]["ff"]["incoherent"]["Ico"] = (data_circular[run_name]["ff"]["incoherent"]["s0"] - data_circular[run_name]["ff"]["incoherent"]["s3"]) / 2
    data_circular[run_name]["ff"]["incoherent"]["Icross"] = (data_circular[run_name]["ff"]["incoherent"]["s0"] + data_circular[run_name]["ff"]["incoherent"]["s3"]) / 2
    data_circular[run_name]["ff"]["incoherent"]["Itotal"] = data_circular[run_name]["ff"]["incoherent"]["Ico"] + data_circular[run_name]["ff"]["incoherent"]["Icross"]
    
    eps = 1e-30  # pure numerical guard, no physical effect
    for key in ["Ico", "Icross", "Itotal"]:
        data_circular[run_name]["ff"]["enhancement"][key] = (
            (data_circular[run_name]["ff"]["coherent"][key] + eps)
            / (data_circular[run_name]["ff"]["incoherent"][key] + eps)
        )

    for it in range(N_t):
        data_circular[run_name]["ff_timed"][it]["coherent"]["Ico"] = (data_circular[run_name]["ff_timed"][it]["coherent"]["s0"] - data_circular[run_name]["ff_timed"][it]["coherent"]["s3"]) / 2
        data_circular[run_name]["ff_timed"][it]["coherent"]["Icross"] = (data_circular[run_name]["ff_timed"][it]["coherent"]["s0"] + data_circular[run_name]["ff_timed"][it]["coherent"]["s3"]) / 2
        data_circular[run_name]["ff_timed"][it]["coherent"]["Itotal"] = data_circular[run_name]["ff_timed"][it]["coherent"]["Ico"] + data_circular[run_name]["ff_timed"][it]["coherent"]["Icross"]
        data_circular[run_name]["ff_timed"][it]["incoherent"]["Ico"] = (data_circular[run_name]["ff_timed"][it]["incoherent"]["s0"] - data_circular[run_name]["ff_timed"][it]["incoherent"]["s3"]) / 2
        data_circular[run_name]["ff_timed"][it]["incoherent"]["Icross"] = (data_circular[run_name]["ff_timed"][it]["incoherent"]["s0"] + data_circular[run_name]["ff_timed"][it]["incoherent"]["s3"]) / 2
        data_circular[run_name]["ff_timed"][it]["incoherent"]["Itotal"] = data_circular[run_name]["ff_timed"][it]["incoherent"]["Ico"] + data_circular[run_name]["ff_timed"][it]["incoherent"]["Icross"]
        
        for key in ["Ico", "Icross", "Itotal"]:
            data_circular[run_name]["ff_timed"][it]["enhancement"][key] = (
                (data_circular[run_name]["ff_timed"][it]["coherent"][key] + eps)
                / (data_circular[run_name]["ff_timed"][it]["incoherent"][key] + eps)
            )



theta_mrad = theta_degrees_circular * (1000 * np.pi / 180)
layered_names = [
    '0000_n_layers_2', '0001_n_layers_5', '0002_n_layers_10'
]

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
# fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# # reference single-particle curves - 0.15
# Ico = data_circular["0005_n_radius_0.15"]["ff"]["enhancement"]["Ico"]
# Icros = data_circular["0005_n_radius_0.15"]["ff"]["enhancement"]["Icross"]

# theta_coherent_milirad_a = data_circular["0005_n_radius_0.15"]["theta_coherent_mrad_a"]
# ax_left.axvline(theta_coherent_milirad_a, color=REF_COLORS["small"], linestyle="--", alpha=0.7)
# ax_right.axvline(theta_coherent_milirad_a, color=REF_COLORS["small"], linestyle="--", alpha=0.7)

# ax_left.plot(theta_mrad, Ico, color=REF_COLORS["small"], linestyle="-", label="Single particle (a=0.15)", linewidth=2)
# ax_right.plot(theta_mrad, Icros, color=REF_COLORS["small"], linestyle="-", label="Single particle (a=0.15)", linewidth=2)

# # reference single-particle curves - 1.0
# Ico = data_circular["0006_n_radius_1.0"]["ff"]["enhancement"]["Ico"]
# Icros = data_circular["0006_n_radius_1.0"]["ff"]["enhancement"]["Icross"]

# theta_coherent_milirad_b = data_circular["0005_n_radius_0.15"]["theta_coherent_mrad_b"]
# ax_left.axvline(theta_coherent_milirad_b, color=REF_COLORS["large"], linestyle="--", alpha=0.7)
# ax_right.axvline(theta_coherent_milirad_b, color=REF_COLORS["large"], linestyle="--", alpha=0.7)

# ax_left.plot(theta_mrad, Ico, color=REF_COLORS["large"], linestyle="-", label="Single particle (a=1.0)", linewidth=2)
# ax_right.plot(theta_mrad, Icros, color=REF_COLORS["large"], linestyle="-", label="Single particle (a=1.0)", linewidth=2)

# # layered curves
# for layered_name in layered_names:
#     n_layers = data_circular[layered_name]["n_depth_layer"]

#     Ico = data_circular[layered_name]["ff"]["enhancement"]["Ico"]
#     ax_left.plot(theta_mrad, Ico, label=f"Layered (n={n_layers})", color=LAYERED_COLORS[n_layers], linestyle=LAYERED_DASHES[n_layers], linewidth=1.5)
#     Icros = data_circular[layered_name]["ff"]["enhancement"]["Icross"]
#     ax_right.plot(theta_mrad, Icros, label=f"Layered (n={n_layers})", color=LAYERED_COLORS[n_layers], linestyle=LAYERED_DASHES[n_layers], linewidth=1.5)


# ax_left.set_title("Co-polarized")
# ax_left.set_xlabel("Scattering angle (mrad)")
# ax_left.set_ylabel("Enhancement factor")

# ax_right.set_title("Cross-polarized")
# ax_right.set_xlabel("Scattering angle (mrad)")
# ax_right.legend()

# MAX_MRAD = 100
# ax_left.set_xlim(0, MAX_MRAD)
# ax_right.set_xlim(0, MAX_MRAD)

# plt.tight_layout()
# plt.savefig(f"{save_path}/cbs-mie-layered-enhancement-profiles.png", dpi=300)




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

def extract_boundary(enhancement_matrix, theta_mrad, threshold=1.5, smooth_sigma=1.5):
    theta_boundary = np.full(enhancement_matrix.shape[0], np.nan)

    for it in range(enhancement_matrix.shape[0]):
        profile = enhancement_matrix[it, :]          # E(theta) at this time
        mask    = profile > threshold
        if mask.any():
            # largest theta still above threshold
            theta_boundary[it] = theta_mrad[np.where(mask)[0][-1]]

    # Smooth to remove shot-noise jitter
    valid = ~np.isnan(theta_boundary)
    theta_boundary[valid] = gaussian_filter1d(
        theta_boundary[valid], sigma=smooth_sigma
    )
    return theta_boundary


# ------ Time evolution heatmaps
MAX_MRAD = 100
VMIN, VMAX = 1.0, 2.0

norm = plt.matplotlib.colors.Normalize(vmin=VMIN, vmax=VMAX)
cmap = "RdYlBu_r" 

time_axis = np.arange(N_t_max) * dt
boundaries = {}

fig, axes = plt.subplots(1, 5, figsize=(15, 3), sharex=True, sharey=True, gridspec_kw={"wspace": 0.04})

enhancement_matrix = np.array([data_circular["0005_n_radius_0.15"]["ff_timed"][it]["enhancement"]["Ico"] for it in range(N_t_max)])
im = axes[0].imshow(enhancement_matrix.T, aspect="auto", origin="lower", extent=[0, N_t_max * dt, theta_mrad[0], theta_mrad[-1]], norm=norm, cmap=cmap)
axes[0].set_title(r"Single particle ($a=0.15$)")
axes[0].set_xlabel(r"Time $[\ell/c]$")
axes[0].set_ylabel(r"Scattering angle (mrad)")
axes[0].set_ylim(0, MAX_MRAD)

boundaries["0005_n_radius_0.15"] = extract_boundary(enhancement_matrix, theta_mrad)
axes[0].plot(time_axis, boundaries["0005_n_radius_0.15"], color="white", linestyle="-", label="Boundary", linewidth=2)

enhancement_matrix = np.array([data_circular["0006_n_radius_1.0"]["ff_timed"][it]["enhancement"]["Ico"] for it in range(N_t_max)])
im = axes[1].imshow(enhancement_matrix.T, aspect="auto", origin="lower", extent=[0, N_t_max * dt, theta_mrad[0], theta_mrad[-1]], norm=norm, cmap=cmap)
axes[1].set_title(r"Single particle ($a=1.0$)")
axes[1].set_xlabel(r"Time $[\ell/c]$")
axes[1].set_ylim(0, MAX_MRAD)

boundaries["0006_n_radius_1.0"] = extract_boundary(enhancement_matrix, theta_mrad)
axes[1].plot(time_axis, boundaries["0006_n_radius_1.0"], color="white", linestyle="-", label="Boundary", linewidth=2)

for i, layered_name in enumerate(layered_names):
    n_layers = data_circular[layered_name]["n_depth_layer"]

    enhancement_matrix = np.array([data_circular[layered_name]["ff_timed"][it]["enhancement"]["Ico"] for it in range(N_t_max)])
    
    im = axes[i + 2].imshow(enhancement_matrix.T, aspect="auto", origin="lower", extent=[0, N_t_max * dt, theta_mrad[0], theta_mrad[-1]], norm=norm, cmap=cmap)
    axes[i + 2].set_title(rf"Layered ($n={n_layers}$)")
    axes[i + 2].set_xlabel(r"Time $[\ell/c]$")
    axes[i + 2].set_ylim(0, MAX_MRAD)

    boundaries[layered_name] = extract_boundary(enhancement_matrix, theta_mrad)
    axes[i + 2].plot(time_axis, boundaries[layered_name], color="white", linestyle="-", label="Boundary", linewidth=2)

cbar = fig.colorbar(
    im, ax=axes,
    orientation="vertical",
    fraction=0.012,
    pad=0.01,
    extend="max"        # arrow at top signals values can exceed VMAX (noisy data)
)
cbar.set_label(r"Enhancement factor $E(\theta,\,t)$", fontsize=9)
cbar.ax.tick_params(labelsize=8.5)
cbar.set_ticks([1.0, 1.25, 1.5, 1.75, 2.0])

plt.tight_layout()
plt.savefig(f"{save_path}/cbs-mie-layered-enhancement-heatmaps.png", dpi=300)



fig, ax = plt.subplots(figsize=(6, 4))

for run_name, boundary in boundaries.items():
    label = None
    color = None
    line_style = None
    line_width = 1.5
    if run_name == "0005_n_radius_0.15":
        label = "Single particle (a=0.15)"
        color = REF_COLORS["small"]
        line_style = "-"
        line_width = 2
    elif run_name == "0006_n_radius_1.0":
        label = "Single particle (a=1.0)"
        color = REF_COLORS["large"]
        line_style = "-"
        line_width = 2
    else:
        n_layers = data_circular[run_name]["n_depth_layer"]
        label = f"Layered ($n={n_layers}$)"
        color = LAYERED_COLORS[n_layers]
        line_style = LAYERED_DASHES[n_layers]

    ax.plot(time_axis, boundary, label=label, color=color, linestyle=line_style, linewidth=line_width)

ax.set_xlabel(r"Time $[\ell/c]$")
ax.set_ylabel(r"Boundary angle $\theta_b$ (mrad)")
ax.legend()
ax.set_xlim(0, N_t_max * dt)
ax.set_ylim(0, MAX_MRAD)
plt.tight_layout()
plt.savefig(f"{save_path}/cbs-mie-layered-enhancement-boundaries.png", dpi=300)






def boundary_model(t, theta0, tau, alpha):
    """Power-law decay: theta*(t) = theta0 * (t/tau)^(-alpha)"""
    return theta0 * (t / tau) ** (-alpha)

# Fit — skip t=0 to avoid division by zero
t_fit = time_axis[2:]
for label, curve in boundaries.items():
    b_fit = curve[2:]
    valid = ~np.isnan(b_fit) & (b_fit > 0)
    try:
        popt, _ = curve_fit(boundary_model, t_fit[valid], b_fit[valid],
                            p0=[80, 5, 0.8], maxfev=5000)
        print(f"{label}: theta0={popt[0]:.1f}, tau={popt[1]:.2f}, alpha={popt[2]:.3f}")
    except Exception as e:
        print(f"{label}: fit failed — {e}")




# ------ Show all figures
plt.show()