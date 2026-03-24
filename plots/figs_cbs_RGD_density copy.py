from scipy.sparse.linalg._isolve.lsqr import eps
from utils.loaders import load_sweep
from utils.styles import apply
import utils.figures as figures
import utils.cbs as cbs
from scipy.ndimage import uniform_filter1d

import numpy as np
import matplotlib.pyplot as plt


apply(context="paper", col="single")

save_path = "/Users/niaggar/Documents/Thesis/Results"

f_scale = 50
volume_fraction_s = [0.05, 0.06, 0.08, 0.10, 0.15]

data = {}

theta_degrees = np.linspace(0, 30, 600)
theta_mrad = theta_degrees * (1000 * np.pi / 180)
phi_degrees = np.linspace(0, 360, 60)

MAX_MRAD = 70
N_PHOTONS = 1_000_000_000

t_max = 50.0
d_time = 5.0
n_time_bins = int(t_max / d_time)



circular_raw = load_sweep(f"data/cbs/cbs_esclaed_0.100_circular")

for run_name in circular_raw.keys():
    circular_loader = circular_raw[run_name]

    radius = circular_loader.params["radius_um"]
    volume_fraction = circular_loader.params["volume_fraction"]
    n_particle = circular_loader.params["n_particle"]
    n_medium = circular_loader.params["n_medium"]
    m_relative = circular_loader.params["m_relative"]

    wavelength = circular_loader.params["wavelength_um"]
    mean_free_path = circular_loader.params["mean_free_path_ls_um"]
    transport_mean_free_path = circular_loader.params["transport_mean_free_path_lstar_um"]

    scattering_efficiency = circular_loader.params["scattering_efficiency"]
    mu_scattering = circular_loader.params["mu_scattering_um_inv"]
    mu_absortion = circular_loader.params["mu_absortion_um_inv"]
    anisotropy_factor = circular_loader.params["anisotropy_factor"]
    size_parameter = circular_loader.params["size_parameter"]
    condition_1 = circular_loader.params["condition_1"]
    condition_2 = circular_loader.params["condition_2"]

    theta_coherent = 1 / (2 * np.pi * n_medium * transport_mean_free_path / wavelength) * 180 / np.pi  # Convertir a grados
    theta_coherent_milirad = theta_coherent * (1000 * np.pi / 180)

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
        "ff_timed": [{
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
    }

    data[run_name] = data_current

    # Calculate the intensities and enhancement factors for each time bin and scattering order, and store them in the data structure. Use a small epsilon to avoid division by zero when calculating enhancement factors.
    eps = 1e-30  # pure numerical guard, no physical effect

    for t in range(n_time_bins):
        data[run_name]["ff_timed"][t]["coherent"]["Ico"] = (data[run_name]["ff_timed"][t]["coherent"]["s0"] - data[run_name]["ff_timed"][t]["coherent"]["s3"]) / 2
        data[run_name]["ff_timed"][t]["coherent"]["Icross"] = (data[run_name]["ff_timed"][t]["coherent"]["s0"] + data[run_name]["ff_timed"][t]["coherent"]["s3"]) / 2
        data[run_name]["ff_timed"][t]["coherent"]["Itotal"] = data[run_name]["ff_timed"][t]["coherent"]["Ico"] + data[run_name]["ff_timed"][t]["coherent"]["Icross"]
        data[run_name]["ff_timed"][t]["incoherent"]["Ico"] = (data[run_name]["ff_timed"][t]["incoherent"]["s0"] - data[run_name]["ff_timed"][t]["incoherent"]["s3"]) / 2
        data[run_name]["ff_timed"][t]["incoherent"]["Icross"] = (data[run_name]["ff_timed"][t]["incoherent"]["s0"] + data[run_name]["ff_timed"][t]["incoherent"]["s3"]) / 2
        data[run_name]["ff_timed"][t]["incoherent"]["Itotal"] = data[run_name]["ff_timed"][t]["incoherent"]["Ico"] + data[run_name]["ff_timed"][t]["incoherent"]["Icross"]

        data[run_name]["ff_timed"][t]["enhancement"]["Ico"] = (
            (data[run_name]["ff_timed"][t]["coherent"]["Ico"] + eps)
            / (data[run_name]["ff_timed"][t]["incoherent"]["Ico"] + eps)
        )
        data[run_name]["ff_timed"][t]["enhancement"]["Icross"] = (
            (data[run_name]["ff_timed"][t]["coherent"]["Icross"] + eps)
            / (data[run_name]["ff_timed"][t]["incoherent"]["Icross"] + eps)
        )
        data[run_name]["ff_timed"][t]["enhancement"]["total"] = (
            (data[run_name]["ff_timed"][t]["coherent"]["Itotal"] + eps)
            / (data[run_name]["ff_timed"][t]["incoherent"]["Itotal"] + eps)
        )



# plot total intensityes for each type of particle. The total is on ff_timed[0], the rest of the time bins are for the temporal evolution

fig, ax = plt.subplots(figsize=(10, 4))

for run_name in data.keys():
    print(f"Run: {run_name}")
    label = run_name
    run = data[run_name]

    theta_coherent_milirad = run["theta_coherent_mrad"]

    data_plot = np.mean(run["ff_timed"][0]["enhancement"]["Ico"], axis=1)
    ax.plot(theta_mrad, data_plot, label=label)

    ax.axvline(theta_coherent_milirad,label=f"Coherent angle {label}")

ax.set_xlabel("Scattering Angle")
ax.set_ylabel("Enhancement Factor")
ax.legend()
plt.tight_layout()

# for run_name in data.keys():
#     data_plot = data[run_name]
#     figures.plot_disk(
#         Icoh=(
#             data_plot["ff_timed"][0]["coherent"]["Itotal"],
#             data_plot["ff_timed"][0]["coherent"]["Ico"],
#             data_plot["ff_timed"][0]["coherent"]["Icross"],
#         ),
#         Iinc=(
#             data_plot["ff_timed"][0]["incoherent"]["Itotal"],
#             data_plot["ff_timed"][0]["incoherent"]["Ico"],
#             data_plot["ff_timed"][0]["incoherent"]["Icross"],
#         ),
#         labels=["Total", "Co-polarized", "Cross-polarized"],
#         theta_deg=theta_degrees,
#         phi_data=np.deg2rad(phi_degrees),
#         title=f"CBS Disk Plot (Circular) - Volume Fraction {data_plot['volume_fraction']:.3f}",
#         save_path=f"{save_path}/just_a_test_{run_name}.png"
#     )







# for run_name in run_names_circular:
#     transport_mean_free_path = data_circular[run_name]["transport_mean_free_path"]
#     theta_coherent_milirad = data_circular[run_name]["theta_coherent_mrad"]
#     radius = data_circular[run_name]["radius"]
#     volume_fraction = data_circular[run_name]["volume_fraction"]

#     label = f"$a={radius:.3f}$, $f={volume_fraction:.3f}$"

#     fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

#     ax_left.plot(theta_mrad, data_circular[run_name]["ff_timed"][0]["enhancement"]["Ico"], label=label)
#     ax_left.axvline(theta_coherent_milirad, color="red", linestyle="--", alpha=0.7)
#     ax_left.set_xlabel("Scattering angle (mrad)")
#     ax_left.set_ylabel("Enhancement factor")

#     ax_right.plot(theta_mrad, data_circular[run_name]["ff_timed"][0]["enhancement"]["Icross"], label=label)
#     ax_right.axvline(theta_coherent_milirad, color="red", linestyle="--", alpha=0.7)
#     ax_right.set_xlabel("Scattering angle (mrad)")
#     ax_right.legend()

#     plt.tight_layout()
#     plt.savefig(f"{save_path}/cbs-RGD-ff-total-enhancement-ico-icross-{run_name}.pdf")


# fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4))
# for run_name in run_names_circular:
#     transport_mean_free_path = data_circular[run_name]["transport_mean_free_path"]
#     theta_coherent_milirad = data_circular[run_name]["theta_coherent_mrad"]
#     radius = data_circular[run_name]["radius"]
#     volume_fraction = data_circular[run_name]["volume_fraction"]

#     label = f"$a={radius:.3f}$, $f={volume_fraction:.3f}$"

#     ax_left.plot(theta_mrad, data_circular[run_name]["ff_timed"][0]["enhancement"]["Ico"], label=label)
#     # ax_left.axvline(theta_coherent_milirad, color="red", linestyle="--", alpha=0.7)
#     ax_left.set_xlabel("Scattering angle (mrad)")
#     ax_left.set_ylabel("Intensity (a.u.)")
#     # ax_left.set_ylabel("Enhancement factor")
#     ax_left.set_title("Co-polarized")
#     ax_left.set_xlim(0, 40)


#     ax_right.plot(theta_mrad, data_circular[run_name]["ff_timed"][0]["enhancement"]["Icross"], label=label)
#     # ax_right.axvline(theta_coherent_milirad, color="red", linestyle="--", alpha=0.7)
#     ax_right.set_xlabel("Scattering angle (mrad)")
#     ax_right.set_ylabel("Intensity (a.u.)")
#     ax_right.set_title("Cross-polarized")
#     ax_right.set_xlim(0, 40)
#     ax_right.legend()

# plt.tight_layout()
# plt.savefig(f"{save_path}/cbs-RGD-ff-total-enhancement-ico-icross.pdf")



# for run_name in run_names_circular:
#     transport_mean_free_path = data_circular[run_name]["transport_mean_free_path"]
#     theta_coherent_milirad = data_circular[run_name]["theta_coherent_mrad"]
#     radius = data_circular[run_name]["radius"]
#     volume_fraction = data_circular[run_name]["volume_fraction"]

#     label = f"$a={radius:.3f}$, $f={volume_fraction:.3f}$"

#     fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

#     for order in scattering_order_bins:
#         ax_left.plot(theta_mrad, data_circular[run_name]["ff_order"][scattering_order_bins.index(order)]["coherent"]["Ico"], label=f"order {order}")
#         ax_right.plot(theta_mrad, data_circular[run_name]["ff_order"][scattering_order_bins.index(order)]["coherent"]["Icross"], label=f"order {order}")

#     plt.tight_layout()
#     ax_right.legend()
#     ax_left.set_xlabel("Scattering angle (mrad)")
#     ax_left.set_ylabel("Enhancement factor")
#     plt.savefig(f"{save_path}/cbs-RGD-ff-scattering-order-enhancement-ico-icross-{run_name}.pdf")




plt.show()
