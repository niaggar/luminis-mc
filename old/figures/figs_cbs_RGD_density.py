from scipy.sparse.linalg._isolve.lsqr import eps
from utils.loaders import load_sweep
from utils.styles import apply
import utils.figures as figures
import utils.cbs as cbs
from scipy.ndimage import uniform_filter1d
from utils.cbs_fit import fit_cbs, plot_cbs_fit

import numpy as np
import matplotlib.pyplot as plt


apply(context="paper", col="single")

save_path = "/Users/niaggar/Documents/Thesis/Results"
folder_base = "Data/cbs/"


def get_sweep_name(radius, polarization):
    name = f"cbs_escaled_{radius:.3f}_{polarization}"
    return name

def get_run_name(volume_fraction):
    if volume_fraction == 5.00:
        return "0002_volume_fraction_5.00"
    elif volume_fraction == 3.50:
        return "0001_volume_fraction_3.50"
    elif volume_fraction == 2.50:
        return "0000_volume_fraction_2.50"


volume_fraction_s = [2.50, 3.50, 5.00]
polarization = ["linear", "circular"]
radius_list = [0.070, 0.085, 0.100]


data = {}

f_scale = 50
theta_degrees = np.linspace(0, 30, 600) / f_scale
theta_mrad = theta_degrees * (1000 * np.pi / 180)
phi_degrees = np.linspace(0, 360, 60)

MAX_MRAD = 70
N_PHOTONS = 200_000_000

t_max = 50.0
d_time = 5.0
n_time_bins = int(t_max / d_time)

n_medium = 1.33

for r in radius_list:
    linear = get_sweep_name(r, polarization[0])
    circular = get_sweep_name(r, polarization[1])
    linear_raw = load_sweep(f"{folder_base}/{linear}")
    circular_raw = load_sweep(f"{folder_base}/{circular}")

    data[f"{r}"] = {}
    data_current = {}

    for run_name in circular_raw.keys():
        circular_loader = circular_raw[run_name]
        linear_loader = linear_raw[run_name]

        radius = circular_loader.params["radius_um"]
        volume_fraction = circular_loader.params["volume_fraction"] / f_scale
        n_particle = circular_loader.params["n_particle"]
        n_medium = circular_loader.params["n_medium"]
        m_relative = circular_loader.params["m_relative"]

        wavelength = circular_loader.params["wavelength_um"]
        mean_free_path = circular_loader.params["mean_free_path_ls_um"] * f_scale
        transport_mean_free_path = circular_loader.params["transport_mean_free_path_lstar_um"] * f_scale

        scattering_efficiency = circular_loader.params["scattering_efficiency"]
        mu_scattering = circular_loader.params["mu_scattering_um_inv"]
        mu_absortion = circular_loader.params["mu_absortion_um_inv"]
        anisotropy_factor = circular_loader.params["anisotropy_factor"]
        size_parameter = circular_loader.params["size_parameter"]
        condition_1 = circular_loader.params["condition_1"]
        condition_2 = circular_loader.params["condition_2"]


        k = (2 * np.pi / wavelength)

        theta_coherent = 1 / (k * transport_mean_free_path) * 180 / np.pi  # Convertir a grados
        theta_coherent_milirad = theta_coherent * (1000 * np.pi / 180)
        print(f"Radius: {radius:.3f} um, Volume fraction: {volume_fraction:.3f}, Anisotropy factor: {anisotropy_factor:.4f}, Mean free path: {mean_free_path:.4f} um, Transport mean free path: {transport_mean_free_path:.4f} um, Coherent backscattering angle: {theta_coherent:.4f} degrees ({theta_coherent_milirad:.4f} mrad)")

        # print(f"Anisotropy factor for radius {radius:.3f}: {anisotropy_factor:.4f}")
        # print(f"Condition 1 (|m-1|): {condition_1:.4f}")
        # print(f"Condition 2 (size parameter * |m-1|): {condition_2:.4f}")
        # print(f"Transport mean free path: {transport_mean_free_path:.4f}")
        # print(f"Coherent backscattering angle (degrees): {theta_coherent:.4f}")
        # print(f"Coherent backscattering angle (mrad): {theta_coherent_milirad:.4f}")

        params = circular_loader.params
        data_current = {
            "anisotropy": anisotropy_factor,
            "mean_free_path": mean_free_path,
            "transport_mean_free_path": transport_mean_free_path,
            "scattering_efficiency": scattering_efficiency,
            "volume_fraction": volume_fraction,
            "radius": radius,
            "size_parameter": size_parameter,
            "theta_coherent": theta_coherent,
            "theta_coherent_mrad": theta_coherent_milirad,
            "ff_circular_timed": [{
                "coherent": {
                    "s0": circular_loader.derived(f"farfield_cbs_timed_{t}/coherent/s0"),
                    "s1": circular_loader.derived(f"farfield_cbs_timed_{t}/coherent/s1"),
                    "s2": circular_loader.derived(f"farfield_cbs_timed_{t}/coherent/s2"),
                    "s3": circular_loader.derived(f"farfield_cbs_timed_{t}/coherent/s3"),
                },
                "incoherent": {
                    "s0": circular_loader.derived(f"farfield_cbs_timed_{t}/incoherent/s0"),
                    "s1": circular_loader.derived(f"farfield_cbs_timed_{t}/incoherent/s1"),
                    "s2": circular_loader.derived(f"farfield_cbs_timed_{t}/incoherent/s2"),
                    "s3": circular_loader.derived(f"farfield_cbs_timed_{t}/incoherent/s3"),
                },
                "enhancement": {},
            } for t in range(n_time_bins)],
            "ff_linear_timed": [{
                "coherent": {
                    "s0": linear_loader.derived(f"farfield_cbs_timed_{t}/coherent/s0"),
                    "s1": linear_loader.derived(f"farfield_cbs_timed_{t}/coherent/s1"),
                    "s2": linear_loader.derived(f"farfield_cbs_timed_{t}/coherent/s2"),
                    "s3": linear_loader.derived(f"farfield_cbs_timed_{t}/coherent/s3"),
                },
                "incoherent": {
                    "s0": linear_loader.derived(f"farfield_cbs_timed_{t}/incoherent/s0"),
                    "s1": linear_loader.derived(f"farfield_cbs_timed_{t}/incoherent/s1"),
                    "s2": linear_loader.derived(f"farfield_cbs_timed_{t}/incoherent/s2"),
                    "s3": linear_loader.derived(f"farfield_cbs_timed_{t}/incoherent/s3"),
                },
                "enhancement": {},
            } for t in range(n_time_bins)],
        }

        data[f"{r}"][run_name] = data_current

        eps = 1e-30
        for t in range(n_time_bins):
            s0_coh = data[f"{r}"][run_name]["ff_circular_timed"][t]["coherent"]["s0"]
            s1_coh = data[f"{r}"][run_name]["ff_circular_timed"][t]["coherent"]["s1"]
            s3_coh = data[f"{r}"][run_name]["ff_circular_timed"][t]["coherent"]["s3"]

            s0_inc = data[f"{r}"][run_name]["ff_circular_timed"][t]["incoherent"]["s0"]
            s1_inc = data[f"{r}"][run_name]["ff_circular_timed"][t]["incoherent"]["s1"]
            s3_inc = data[f"{r}"][run_name]["ff_circular_timed"][t]["incoherent"]["s3"]

            data[f"{r}"][run_name]["ff_circular_timed"][t]["coherent"]["Ico"] = (s0_coh - s3_coh) / 2
            data[f"{r}"][run_name]["ff_circular_timed"][t]["coherent"]["Icross"] = (s0_coh + s3_coh) / 2
            data[f"{r}"][run_name]["ff_circular_timed"][t]["coherent"]["Ix"] = (s0_coh + s1_coh) / 2
            data[f"{r}"][run_name]["ff_circular_timed"][t]["coherent"]["Iy"] = (s0_coh - s1_coh) / 2
            data[f"{r}"][run_name]["ff_circular_timed"][t]["coherent"]["Itotal"] = s0_coh

            data[f"{r}"][run_name]["ff_circular_timed"][t]["incoherent"]["Ico"] = (s0_inc - s3_inc) / 2
            data[f"{r}"][run_name]["ff_circular_timed"][t]["incoherent"]["Icross"] = (s0_inc + s3_inc) / 2
            data[f"{r}"][run_name]["ff_circular_timed"][t]["incoherent"]["Ix"] = (s0_inc + s1_inc) / 2
            data[f"{r}"][run_name]["ff_circular_timed"][t]["incoherent"]["Iy"] = (s0_inc - s1_inc) / 2
            data[f"{r}"][run_name]["ff_circular_timed"][t]["incoherent"]["Itotal"] = s0_inc

            data[f"{r}"][run_name]["ff_circular_timed"][t]["enhancement"]["Ico"] = (
                (data[f"{r}"][run_name]["ff_circular_timed"][t]["coherent"]["Ico"] + eps)
                / (data[f"{r}"][run_name]["ff_circular_timed"][t]["incoherent"]["Ico"] + eps)
            )
            data[f"{r}"][run_name]["ff_circular_timed"][t]["enhancement"]["Icross"] = (
                (data[f"{r}"][run_name]["ff_circular_timed"][t]["coherent"]["Icross"] + eps)
                / (data[f"{r}"][run_name]["ff_circular_timed"][t]["incoherent"]["Icross"] + eps)
            )
            data[f"{r}"][run_name]["ff_circular_timed"][t]["enhancement"]["Ix"] = (
                (data[f"{r}"][run_name]["ff_circular_timed"][t]["coherent"]["Ix"] + eps)
                / (data[f"{r}"][run_name]["ff_circular_timed"][t]["incoherent"]["Ix"] + eps)
            )
            data[f"{r}"][run_name]["ff_circular_timed"][t]["enhancement"]["Iy"] = (
                (data[f"{r}"][run_name]["ff_circular_timed"][t]["coherent"]["Iy"] + eps)
                / (data[f"{r}"][run_name]["ff_circular_timed"][t]["incoherent"]["Iy"] + eps)
            )
            data[f"{r}"][run_name]["ff_circular_timed"][t]["enhancement"]["total"] = (
                (data[f"{r}"][run_name]["ff_circular_timed"][t]["coherent"]["Itotal"] + eps)
                / (data[f"{r}"][run_name]["ff_circular_timed"][t]["incoherent"]["Itotal"] + eps)
            )

        for t in range(n_time_bins):
            s0_coh = data[f"{r}"][run_name]["ff_linear_timed"][t]["coherent"]["s0"]
            s1_coh = data[f"{r}"][run_name]["ff_linear_timed"][t]["coherent"]["s1"]
            s3_coh = data[f"{r}"][run_name]["ff_linear_timed"][t]["coherent"]["s3"]

            s0_inc = data[f"{r}"][run_name]["ff_linear_timed"][t]["incoherent"]["s0"]
            s1_inc = data[f"{r}"][run_name]["ff_linear_timed"][t]["incoherent"]["s1"]
            s3_inc = data[f"{r}"][run_name]["ff_linear_timed"][t]["incoherent"]["s3"]

            data[f"{r}"][run_name]["ff_linear_timed"][t]["coherent"]["Ico"] = (s0_coh - s3_coh) / 2
            data[f"{r}"][run_name]["ff_linear_timed"][t]["coherent"]["Icross"] = (s0_coh + s3_coh) / 2
            data[f"{r}"][run_name]["ff_linear_timed"][t]["coherent"]["Ix"] = (s0_coh + s1_coh) / 2
            data[f"{r}"][run_name]["ff_linear_timed"][t]["coherent"]["Iy"] = (s0_coh - s1_coh) / 2
            data[f"{r}"][run_name]["ff_linear_timed"][t]["coherent"]["Itotal"] = s0_coh

            data[f"{r}"][run_name]["ff_linear_timed"][t]["incoherent"]["Ico"] = (s0_inc - s3_inc) / 2
            data[f"{r}"][run_name]["ff_linear_timed"][t]["incoherent"]["Icross"] = (s0_inc + s3_inc) / 2
            data[f"{r}"][run_name]["ff_linear_timed"][t]["incoherent"]["Ix"] = (s0_inc + s1_inc) / 2
            data[f"{r}"][run_name]["ff_linear_timed"][t]["incoherent"]["Iy"] = (s0_inc - s1_inc) / 2
            data[f"{r}"][run_name]["ff_linear_timed"][t]["incoherent"]["Itotal"] = s0_inc

            data[f"{r}"][run_name]["ff_linear_timed"][t]["enhancement"]["Ico"] = (
                (data[f"{r}"][run_name]["ff_linear_timed"][t]["coherent"]["Ico"] + eps)
                / (data[f"{r}"][run_name]["ff_linear_timed"][t]["incoherent"]["Ico"] + eps)
            )
            data[f"{r}"][run_name]["ff_linear_timed"][t]["enhancement"]["Icross"] = (
                (data[f"{r}"][run_name]["ff_linear_timed"][t]["coherent"]["Icross"] + eps)
                / (data[f"{r}"][run_name]["ff_linear_timed"][t]["incoherent"]["Icross"] + eps)
            )
            data[f"{r}"][run_name]["ff_linear_timed"][t]["enhancement"]["Ix"] = (
                (data[f"{r}"][run_name]["ff_linear_timed"][t]["coherent"]["Ix"] + eps)
                / (data[f"{r}"][run_name]["ff_linear_timed"][t]["incoherent"]["Ix"] + eps)
            )
            data[f"{r}"][run_name]["ff_linear_timed"][t]["enhancement"]["Iy"] = (
                (data[f"{r}"][run_name]["ff_linear_timed"][t]["coherent"]["Iy"] + eps)
                / (data[f"{r}"][run_name]["ff_linear_timed"][t]["incoherent"]["Iy"] + eps)
            )
            data[f"{r}"][run_name]["ff_linear_timed"][t]["enhancement"]["total"] = (
                (data[f"{r}"][run_name]["ff_linear_timed"][t]["coherent"]["Itotal"] + eps)
                / (data[f"{r}"][run_name]["ff_linear_timed"][t]["incoherent"]["Itotal"] + eps)
            )


COLOR_DENSE_HIGH = "#1f77b4"
COLOR_DENSE_MEDIUM = "#2ca02c"
COLOR_DENSE_LOW = "#ff7f0e"

def get_color_for_volume_fraction(volume_fraction):
    if np.isclose(volume_fraction, 0.100):
        return COLOR_DENSE_HIGH
    elif np.isclose(volume_fraction, 0.070):
        return COLOR_DENSE_MEDIUM
    elif np.isclose(volume_fraction, 0.050):
        return COLOR_DENSE_LOW
    else:
        return "black"
    

def smooth(y, size=7):
    return uniform_filter1d(y, size=size)

def deg_to_mrad(deg):
    return np.asarray(deg) * (1000 * np.pi / 180)



print("Data loaded and processed successfully.")

# ------ Disk plots

# for r in radius_list:
#     data_radius = data[f"{r}"]
#     for run_name in data_radius.keys():
#         data_run = data_radius[run_name]
#         figures.plot_disk(
#             Icoh=(
#                 data_run["ff_circular_timed"][0]["coherent"]["Itotal"],
#                 data_run["ff_circular_timed"][0]["coherent"]["Ico"],
#                 data_run["ff_circular_timed"][0]["coherent"]["Icross"],
#             ),
#             Iinc=(
#                 data_run["ff_circular_timed"][0]["incoherent"]["Itotal"],
#                 data_run["ff_circular_timed"][0]["incoherent"]["Ico"],
#                 data_run["ff_circular_timed"][0]["incoherent"]["Icross"],
#             ),
#             labels=[r"$I_\mathrm{total}$", r"$I_\mathrm{co}$", r"$I_\mathrm{cross}$"],
#             theta_deg=theta_degrees,
#             phi_data=np.deg2rad(phi_degrees),
#             title=f"Circular Polarization - Radius ${data_run['radius']:.3f}$, Volume fraction ${data_run['volume_fraction']:.3f}$",
#             save_path=f"{save_path}/quantic/cbs-rgd-disk-plot-circular-radius-{data_run['radius']:.3f}-vf-{data_run['volume_fraction']:.3f}.png",
#         )

#         figures.plot_disk(
#             Icoh=(
#                 data_run["ff_linear_timed"][0]["coherent"]["Itotal"],
#                 data_run["ff_linear_timed"][0]["coherent"]["Ix"],
#                 data_run["ff_linear_timed"][0]["coherent"]["Iy"],
#             ),
#             Iinc=(
#                 data_run["ff_linear_timed"][0]["incoherent"]["Itotal"],
#                 data_run["ff_linear_timed"][0]["incoherent"]["Ix"],
#                 data_run["ff_linear_timed"][0]["incoherent"]["Iy"],
#             ),
#             labels=[r"$I_\mathrm{total}$", r"$I_\mathrm{x}$", r"$I_\mathrm{y}$"],
#             theta_deg=theta_degrees,
#             phi_data=np.deg2rad(phi_degrees),
#             title=f"Linear Polarization - Radius ${data_run['radius']:.3f}$, Volume fraction ${data_run['volume_fraction']:.3f}$",
#             save_path=f"{save_path}/quantic/cbs-rgd-disk-plot-linear-radius-{data_run['radius']:.3f}-vf-{data_run['volume_fraction']:.3f}.png",
#         )



# ------ Slices

# for r in radius_list:
#     data_radius = data[f"{r}"]
#     fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

#     for run_name in data_radius.keys():
#         data_run = data_radius[run_name]
#         label = f"$a={data_run['radius']:.3f}$, $f={data_run['volume_fraction']:.3f}$"
#         color = get_color_for_volume_fraction(data_run["volume_fraction"])

#         theta_max_coherent = data_run["theta_coherent_mrad"]

#         theta_deg_slice, data_left_00 = cbs.get_slice(data_run["ff_linear_timed"][0]["enhancement"]["Ix"], theta_degrees, phi_degrees, phi_cut=0.0)
#         _, data_left_90 = cbs.get_slice(data_run["ff_linear_timed"][0]["enhancement"]["Ix"], theta_degrees, phi_degrees, phi_cut=90.0)

#         _, data_right_00 = cbs.get_slice(data_run["ff_linear_timed"][0]["enhancement"]["Iy"], theta_degrees, phi_degrees, phi_cut=0.0)
#         _, data_right_90 = cbs.get_slice(data_run["ff_linear_timed"][0]["enhancement"]["Iy"], theta_degrees, phi_degrees, phi_cut=90.0)

#         theta_mrad_slice = theta_deg_slice * (1000 * np.pi / 180)

#         ax_left.plot(theta_mrad_slice, data_left_00, label=label, color=color)
#         ax_right.plot(theta_mrad_slice, data_right_00, label=label, color=color)

#         ax_left.plot(theta_mrad_slice, data_left_90, label=f"{label} - 90°", color=color, ls="--")
#         ax_right.plot(theta_mrad_slice, data_right_90, label=f"{label} - 90°", color=color, ls="--")

#         # Mark the coherent backscattering angle
#         ax_left.axvline(theta_max_coherent, color=color, lw=0.8, ls=":")
#         ax_left.axvline(-theta_max_coherent, color=color, lw=0.8, ls=":")
#         ax_right.axvline(theta_max_coherent, color=color, lw=0.8, ls=":")
#         ax_right.axvline(-theta_max_coherent, color=color, lw=0.8, ls=":")

#     ax_left.set_xlabel("Scattering angle (mrad)")
#     ax_left.set_ylabel("Enhancement factor (a.u.)")
#     ax_left.set_title(r"$I_\mathrm{x}$ enhancement")
#     ax_right.set_xlabel("Scattering angle (mrad)")
#     ax_right.set_title(r"$I_\mathrm{y}$ enhancement")
#     ax_right.legend()

#     ax_left.set_ylim(1, None)
#     ax_right.set_ylim(1, None)

#     plt.tight_layout()
#     plt.savefig(f"{save_path}/quantic/cbs-RGD-slices-enhancement-ico-icross-radius-{r:.3f}.pdf")



# ------ Anisotropy panel

# for r in radius_list:
#     data_radius = data[f"{r}"]
    
#     # --- Figure layout: 3 rows (one per volume fraction), 3 cols ---
#     # Col 0: Ix(φ=0) and Ix(φ=90) overlaid
#     # Col 1: Iy(φ=0) and Iy(φ=90) overlaid  
#     # Col 2: Anisotropy = I(φ=0) - I(φ=90)   ← NEW: highlights the difference
    
#     run_items = list(data_radius.items())
#     n_runs = len(run_items)
#     fig, axes = plt.subplots(n_runs, 3, figsize=(12, 3 * n_runs), sharey="col")
#     if n_runs == 1:
#         axes = axes[np.newaxis, :]

#     for row, (run_name, data_run) in enumerate(run_items):
#         f_val = data_run["volume_fraction"]
#         a_val = data_run["radius"]
#         theta_coh = data_run["theta_coherent_mrad"]

#         enh_t0 = data_run["ff_circular_timed"][0]["enhancement"]

#         for col, (quantity, title, ylabel) in enumerate([
#             ("Ico", r"$I_\mathrm{co}$ enhancement", "Enhancement"),
#             ("Icross", r"$I_\mathrm{cross}$ enhancement", "Enhancement"),
#             (None, r"Anisotropy $I(\phi{=}0°) - I(\phi{=}90°)$", "Δ Enhancement"),
#         ]):
#             ax = axes[row, col]

#             if quantity is not None:
#                 theta_deg, d0 = cbs.get_slice(enh_t0[quantity], theta_degrees, phi_degrees, phi_cut=0.0)
#                 _, d90 = cbs.get_slice(enh_t0[quantity], theta_degrees, phi_degrees, phi_cut=90.0)
#                 theta_mrad = deg_to_mrad(theta_deg)

#                 ax.plot(theta_mrad, smooth(d0),  label=r"$\phi=0°$",  color="tab:blue",   lw=1.5)
#                 ax.plot(theta_mrad, smooth(d90), label=r"$\phi=90°$", color="tab:orange", lw=1.5, ls="--")
#                 ax.axhline(1.0, color="gray", lw=0.7, ls=":")
#             else:
#                 # Anisotropy panel: plot both Ix and Iy anisotropy
#                 theta_deg, ix0  = cbs.get_slice(enh_t0["Ico"], theta_degrees, phi_degrees, phi_cut=0.0)
#                 _, ix90 = cbs.get_slice(enh_t0["Ico"], theta_degrees, phi_degrees, phi_cut=90.0)
#                 _, iy0  = cbs.get_slice(enh_t0["Icross"], theta_degrees, phi_degrees, phi_cut=0.0)
#                 _, iy90 = cbs.get_slice(enh_t0["Icross"], theta_degrees, phi_degrees, phi_cut=90.0)
#                 theta_mrad = deg_to_mrad(theta_deg)

#                 aniso_x = smooth(np.array(ix0) - np.array(ix90), size=11)
#                 aniso_y = smooth(np.array(iy0) - np.array(iy90), size=11)

#                 ax.plot(theta_mrad, aniso_x, label=r"$\Delta I_\mathrm{co}$", color="tab:blue",  lw=1.5)
#                 ax.plot(theta_mrad, aniso_y, label=r"$\Delta I_\mathrm{cross}$", color="tab:red",   lw=1.5, ls="--")
#                 ax.axhline(0.0, color="gray", lw=0.7, ls=":")

#             # Mark CBS angle
#             for sign in [-1, 1]:
#                 ax.axvline(sign * theta_coh, color="gray", lw=0.8, ls=":", alpha=0.6)

#             ax.set_xlim(-12, 12)
#             ax.set_xlabel("Scattering angle (mrad)")
#             if col == 0:
#                 ax.set_ylabel(f"$a={a_val:.3f}$, $f={f_val:.3f}$\n{ylabel}")
#             if row == 0:
#                 ax.set_title(title)
#             ax.legend(fontsize=8)

#     fig.suptitle(f"Circular polarization CBS — $a = {r}$ µm", fontsize=12, y=1.01)
#     plt.tight_layout()
#     plt.savefig(f"{save_path}/quantic/cbs-RGD-anisotropy-circular-radius-{r:.3f}.pdf", bbox_inches="tight")




# Plot to show the CBS
COLOR_RADIUS_070 = "#2ca02c"
COLOR_RADIUS_085 = "#1f77b4"
COLOR_RADIUS_100 = "#ff7f0e"

def get_color_for_radius(radius):
    if np.isclose(radius, 0.070):
        return COLOR_RADIUS_070
    elif np.isclose(radius, 0.085):
        return COLOR_RADIUS_085
    elif np.isclose(radius, 0.100):
        return COLOR_RADIUS_100
    else:
        return "black"

# fig, ax = plt.subplots(figsize=(6, 4))
# for r in radius_list:
#     data_radius = data[f"{r}"]

#     run_name = get_run_name(volume_fraction_s[2])
#     data_run = data_radius[run_name]

#     label = f"$a={data_run['radius']:.3f}$"
#     color = get_color_for_radius(data_run["radius"])

#     slice_data = cbs.get_sym_slice(data_run["ff_linear_timed"][0]["enhancement"]["Iy"], theta_degrees, phi_degrees, phi_cut=0.0)
#     theta_mrad_slice = slice_data[0]
#     mean_info = slice_data[1]

#     ax.plot(theta_mrad_slice, mean_info, label=label, color=color)
#     ax.axvline(data_run["theta_coherent_mrad"], color=color, lw=0.8, ls=":", alpha=0.6)
#     ax.axvline(-data_run["theta_coherent_mrad"], color=color, lw=0.8, ls=":", alpha=0.6)

# ax.set_xlabel("Scattering angle (mrad)")
# ax.set_ylabel(r"$I_\mathrm{y}$ enhancement")
# ax.set_title("Linear polarization CBS")
# ax.legend()

# plt.tight_layout()
# plt.savefig(f"{save_path}/quantic/cbs-RGD-linear-slice-y-radius-comparison.pdf", bbox_inches="tight")



# def cbs_profile(theta_rad, kl):
#     return 1 + 1 / (1 + kl * np.abs(theta_rad))**2

# def fit_cbs_profile(theta_mrad, enhancement, theta_max_mrad=1.5):
#     # Solo ajusta la región de ángulos pequeños (punta del cono)
#     mask = theta_mrad <= theta_max_mrad
#     theta_rad = theta_mrad[mask] * 1e-3
#     enh_masked = enhancement[mask]

#     popt, pcov = curve_fit(
#         cbs_profile,
#         theta_rad,
#         enh_masked,
#         p0=[50],
#         bounds=(0, np.inf)
#     )
#     kl_fit  = popt[0]
#     kl_err  = np.sqrt(pcov[0, 0])
#     l_fit_um = kl_fit / k * 1e6
#     return kl_fit, kl_err, l_fit_um


# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

# r = 0.070
# data_radius = data[f"{r}"]

# for run_name in data_radius.keys():
#     data_run = data_radius[run_name]

#     label = f"$a={data_run['radius']:.3f}$, $f={data_run['volume_fraction']:.3f}$"
#     color = get_color_for_volume_fraction(data_run["volume_fraction"])

#     mean_info = np.mean(data_run["ff_circular_timed"][0]["enhancement"]["Ico"], axis=1)
#     theta_coherent_mrad = data_run["theta_coherent_mrad"]

#     # Fit del perfil CBS para obtener kl* y l*
#     kl_fit, kl_err, l_fit_um = fit_cbs_profile(theta_mrad, mean_info, theta_max_mrad=1.5)
#     enhancement_fit = cbs_profile(theta_mrad * 1e-3, kl_fit)
#     print(f"Radius {r:.3f} µm: kl* = {kl_fit:.1f}, l* = {l_fit_um:.2f} µm")

#     if run_name == get_run_name(volume_fraction_s[0]):
#         ax1.plot(theta_mrad, mean_info, label=label, color=color)
#         ax1.axvline(theta_coherent_mrad, color=color, lw=0.8, ls=":", alpha=0.6)
#         ax1.plot(theta_mrad, enhancement_fit, label=f"Ajuste: kl*={kl_fit:.1f}", color=color, ls="--")
#     elif run_name == get_run_name(volume_fraction_s[1]):
#         ax2.plot(theta_mrad, mean_info, label=label, color=color)
#         ax2.axvline(theta_coherent_mrad, color=color, lw=0.8, ls=":", alpha=0.6)
#         ax2.plot(theta_mrad, enhancement_fit, label=f"Ajuste: kl*={kl_fit:.1f}", color=color, ls="--")
#     elif run_name == get_run_name(volume_fraction_s[2]):
#         ax3.plot(theta_mrad, mean_info, label=label, color=color)
#         ax3.axvline(theta_coherent_mrad, color=color, lw=0.8, ls=":", alpha=0.6)
#         ax3.plot(theta_mrad, enhancement_fit, label=f"Ajuste: kl*={kl_fit:.1f}", color=color, ls="--")

#     # print(f"Radius {r:.3f} µm: kl* = {kl_fit:.1f}, l* = {l_fit_um:.2f} µm")

# ax1.set_title(r"Volume fraction: 0.05")
# ax2.set_title(r"Volume fraction: 0.07")
# ax3.set_title(r"Volume fraction: 0.10")

# ax1.set_xlim(0, 10)
# ax2.set_xlim(0, 10)
# ax3.set_xlim(0, 10)

# plt.tight_layout()





# # ------ Simple plots

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for r in radius_list:
    data_radius = data[f"{r}"]

    for run_name in data_radius.keys():
        data_run = data_radius[run_name]

        label = f"$a={data_run['radius']:.3f}$, $f={data_run['volume_fraction']:.3f}$"
        color = get_color_for_volume_fraction(data_run["volume_fraction"])

        mean_info = np.mean(data_run["ff_circular_timed"][0]["enhancement"]["Ico"], axis=1)
        theta_coherent_mrad = data_run["theta_coherent_mrad"]

        if np.isclose(data_run["radius"], 0.070):
            ax1.plot(theta_mrad, mean_info, label=label, color=color)
            ax1.axvline(theta_coherent_mrad, color=color, lw=0.8, ls=":", alpha=0.6)

        elif np.isclose(data_run["radius"], 0.085):
            ax2.plot(theta_mrad, mean_info, label=label, color=color)
            ax2.axvline(theta_coherent_mrad, color=color, lw=0.8, ls=":", alpha=0.6)

        elif np.isclose(data_run["radius"], 0.100):
            ax3.plot(theta_mrad, mean_info, label=label, color=color)
            ax3.axvline(theta_coherent_mrad, color=color, lw=0.8, ls=":", alpha=0.6)

ax1.set_title(r"$a=0.070$ µm")
ax2.set_title(r"$a=0.085$ µm")
ax3.set_title(r"$a=0.100$ µm")

ax1.set_xlabel("Scattering angle (mrad)")
ax2.set_xlabel("Scattering angle (mrad)")
ax3.set_xlabel("Scattering angle (mrad)")
ax1.set_ylabel(r"$I_\mathrm{co}$ enhancement")
# ax1.legend()
# ax2.legend()
ax3.legend()
ax1.set_xlim(0, 10)
ax2.set_xlim(0, 10)
ax3.set_xlim(0, 10)

plt.tight_layout()
plt.savefig(f"{save_path}/quantic/cbs-RGD-circular-mean-radius-comparison.pdf", bbox_inches="tight")


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

rad_1_points = []
rad_2_points = []
rad_3_points = []

for r in radius_list:
    data_radius = data[f"{r}"]

    for run_name in data_radius.keys():
        data_run = data_radius[run_name]

        label = f"$a={data_run['radius']:.3f}$, $f={data_run['volume_fraction']:.3f}$"
        color = get_color_for_volume_fraction(data_run["volume_fraction"])

        mean_info = np.mean(data_run["ff_circular_timed"][0]["enhancement"]["Ico"], axis=1)
        theta_coherent_mrad = data_run["theta_coherent_mrad"]

        if np.isclose(data_run["radius"], 0.070):
            r = plot_cbs_fit(ax1, theta_mrad, mean_info, wavelength_nm=514, model="diff", label_prefix=label + " ", show_fwhm=False)
            print(f"a 0.070 µm, f {data_run['volume_fraction']:.3f}, l {r.l_um:.2f} µm, Theory l* {data_run["mean_free_path"]}, {data_run['transport_mean_free_path']} µm")
            print(f"error l: {r.l_um - data_run['transport_mean_free_path']:.2f} µm ({(r.l_um - data_run['transport_mean_free_path']) / data_run['transport_mean_free_path'] * 100:.1f}%)")
            rad_1_points.append((r.fwhm_mrad, r.l_um))

        if np.isclose(data_run["radius"], 0.085):
            r = plot_cbs_fit(ax2, theta_mrad, mean_info, wavelength_nm=514, model="diff", label_prefix=label + " ", show_fwhm=False)
            print(f"a 0.085 µm, f {data_run['volume_fraction']:.3f}, l {r.l_um:.2f} µm, Theory l* {data_run["mean_free_path"]}, {data_run['transport_mean_free_path']} µm")
            print(f"error l: {r.l_um - data_run['transport_mean_free_path']:.2f} µm ({(r.l_um - data_run['transport_mean_free_path']) / data_run['transport_mean_free_path'] * 100:.1f}%)")
            rad_2_points.append((r.fwhm_mrad, r.l_um))

        if np.isclose(data_run["radius"], 0.100):
            r = plot_cbs_fit(ax3, theta_mrad, mean_info, wavelength_nm=514, model="diff", label_prefix=label + " ", show_fwhm=False)
            print(f"a 0.100 µm, f {data_run['volume_fraction']:.3f}, l {r.l_um:.2f} µm, Theory l* {data_run["mean_free_path"]}, {data_run['transport_mean_free_path']} µm")
            print(f"error l: {r.l_um - data_run['transport_mean_free_path']:.2f} µm ({(r.l_um - data_run['transport_mean_free_path']) / data_run['transport_mean_free_path'] * 100:.1f}%)")
            rad_3_points.append((r.fwhm_mrad, r.l_um))

ax1.set_title(r"$a=0.070$ µm")
ax2.set_title(r"$a=0.085$ µm")
ax3.set_title(r"$a=0.100$ µm")

ax1.set_xlabel("Scattering angle (mrad)")
ax2.set_xlabel("Scattering angle (mrad)")
ax3.set_xlabel("Scattering angle (mrad)")
ax1.set_ylabel(r"$I_\mathrm{co}$ enhancement")
# ax1.legend()
# ax2.legend()
# ax3.legend()
ax1.set_xlim(0, 10)
ax2.set_xlim(0, 10)
ax3.set_xlim(0, 10)

plt.tight_layout()
plt.savefig(f"{save_path}/quantic/cbs-RGD-circular-mean-radius-comparison-fit.pdf", bbox_inches="tight")



# Plot the extracted l* vs volume fraction for each radius
fig, ax = plt.subplots(figsize=(6, 4))
for points, radius in zip([rad_1_points, rad_2_points, rad_3_points], [0.070, 0.085, 0.100]):
    fwhm_mrad = [p[0] for p in points]
    l_um = [p[1] for p in points]
    inv_l_um = [1 / l for l in l_um]
    ax.plot(inv_l_um, fwhm_mrad, marker="o", label=f"$a={radius:.3f}$ µm")
ax.set_xlabel("1/l*")
ax.set_ylabel("FWHM of CBS peak (mrad)")
ax.set_title("Extracted l* vs CBS peak width")
ax.legend()
plt.tight_layout()
plt.savefig(f"{save_path}/quantic/cbs-RGD-lstar-vs-fwhm.pdf", bbox_inches="tight")


plt.show()
