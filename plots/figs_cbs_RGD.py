from utils.loaders import load_sweep
from utils.styles import apply
import utils.figures as figures
import utils.cbs as cbs
from scipy.ndimage import uniform_filter1d

import numpy as np
import matplotlib.pyplot as plt


apply(context="paper", col="single")
save_path = "/Users/niaggar/Documents/Thesis/Results"



MAX_MRAD = 70

folder_circular = "16Mar26/test"
# folder_circular = "16Mar26/sim_cbs_RGD_multiple-volumefraction_radius-1000M"
data_circular = {}
sweep_data_circular = load_sweep(folder_circular)
params_sweep = [
    {
        "radius": 0.070 / 2,
        "volume_fraction": 0.1,
    },
    {
        "radius": 0.070 / 2,
        "volume_fraction": 0.2,
    },
    {
        "radius": 0.110 / 2,
        "volume_fraction": 0.1,
    },
    {
        "radius": 0.110 / 2,
        "volume_fraction": 0.2,
    },
    {
        "radius": 0.350 / 2,
        "volume_fraction": 0.1,
    },
    {
        "radius": 0.350 / 2,
        "volume_fraction": 0.2,
    }
]

theta_degrees_circular = np.linspace(0, 45, 400)
theta_mrad = theta_degrees_circular * (1000 * np.pi / 180)

phi_degrees_circular = np.linspace(0, 360, 1)
run_names_circular = sorted(sweep_data_circular.keys())

N_PHOTONS_CIRCULAR = 1_000_000_000


t_max = 0
d_time = 10
n_time_bins = 1

scattering_order_bins = [2, 3, 4, 5, 7, 10, 15, 20, 50]



for run_name, result_loader in sweep_data_circular.items():
    print(f"---------------- Processing run: {run_name}")

    wavelength = result_loader.params["wavelength_um"]
    radius = result_loader.params["radius_um"]
    volume_fraction = result_loader.params["volume_fraction"]
    mean_free_path = result_loader.params["mean_free_path_ls_um"]
    scattering_efficiency = result_loader.params["scattering_efficiency"]

    n_particle = result_loader.params["n_particle"]
    n_medium = result_loader.params["n_medium"]

    anysotropy = result_loader.params["anisotropy_factor"]
    size_parameter = result_loader.params["size_parameter"]
    condition_1 = result_loader.params["condition_1"]
    condition_2 = result_loader.params["condition_2"]
    transport_mean_free_path = mean_free_path / (1 - anysotropy)
    theta_coherent = 1 / (2 * np.pi * n_medium * transport_mean_free_path / wavelength) * 180 / np.pi  # Convertir a grados
    theta_coherent_milirad = theta_coherent * (1000 * np.pi / 180)

    print(f"Anisotropy factor for radius {radius:.3f}: {anysotropy:.4f}")
    print(f"Condition 1 (|m-1|): {condition_1:.4f}")
    print(f"Condition 2 (size parameter * |m-1|): {condition_2:.4f}")
    print(f"Transport mean free path: {transport_mean_free_path:.4f}")
    print(f"Coherent backscattering angle (degrees): {theta_coherent:.4f}")
    print(f"Coherent backscattering angle (mrad): {theta_coherent_milirad:.4f}")

    params = result_loader.params
    data_circular[run_name] = {
        "anisotropy": anysotropy,
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
                "s0": result_loader.derived(f"farfield_cbs_timed_{t}/coherent/s0") / N_PHOTONS_CIRCULAR,
                "s1": result_loader.derived(f"farfield_cbs_timed_{t}/coherent/s1") / N_PHOTONS_CIRCULAR,
                "s2": result_loader.derived(f"farfield_cbs_timed_{t}/coherent/s2") / N_PHOTONS_CIRCULAR,
                "s3": result_loader.derived(f"farfield_cbs_timed_{t}/coherent/s3") / N_PHOTONS_CIRCULAR,
            },
            "incoherent": {
                "s0": result_loader.derived(f"farfield_cbs_timed_{t}/incoherent/s0") / N_PHOTONS_CIRCULAR,
                "s1": result_loader.derived(f"farfield_cbs_timed_{t}/incoherent/s1") / N_PHOTONS_CIRCULAR,
                "s2": result_loader.derived(f"farfield_cbs_timed_{t}/incoherent/s2") / N_PHOTONS_CIRCULAR,
                "s3": result_loader.derived(f"farfield_cbs_timed_{t}/incoherent/s3") / N_PHOTONS_CIRCULAR,
            },
            "enhancement": {},
        } for t in range(n_time_bins)],
        "ff_order": [{
            "coherent": {
                "s0": result_loader.derived(f"farfield_cbs_scattering_order_{order}/coherent/s0") / N_PHOTONS_CIRCULAR,
                "s1": result_loader.derived(f"farfield_cbs_scattering_order_{order}/coherent/s1") / N_PHOTONS_CIRCULAR,
                "s2": result_loader.derived(f"farfield_cbs_scattering_order_{order}/coherent/s2") / N_PHOTONS_CIRCULAR,
                "s3": result_loader.derived(f"farfield_cbs_scattering_order_{order}/coherent/s3") / N_PHOTONS_CIRCULAR,
            },
            "incoherent": {
                "s0": result_loader.derived(f"farfield_cbs_scattering_order_{order}/incoherent/s0") / N_PHOTONS_CIRCULAR,
                "s1": result_loader.derived(f"farfield_cbs_scattering_order_{order}/incoherent/s1") / N_PHOTONS_CIRCULAR,
                "s2": result_loader.derived(f"farfield_cbs_scattering_order_{order}/incoherent/s2") / N_PHOTONS_CIRCULAR,
                "s3": result_loader.derived(f"farfield_cbs_scattering_order_{order}/incoherent/s3") / N_PHOTONS_CIRCULAR,
            },
            "enhancement": {},
        } for order in scattering_order_bins]
    }

    print(len(data_circular[run_name]["ff_timed"]))

    eps = 1e-30  # pure numerical guard, no physical effect

    for t in range(n_time_bins):
        data_circular[run_name]["ff_timed"][t]["coherent"]["Ico"] = (data_circular[run_name]["ff_timed"][t]["coherent"]["s0"] - data_circular[run_name]["ff_timed"][t]["coherent"]["s3"]) / 2
        data_circular[run_name]["ff_timed"][t]["coherent"]["Icross"] = (data_circular[run_name]["ff_timed"][t]["coherent"]["s0"] + data_circular[run_name]["ff_timed"][t]["coherent"]["s3"]) / 2
        data_circular[run_name]["ff_timed"][t]["coherent"]["Itotal"] = data_circular[run_name]["ff_timed"][t]["coherent"]["Ico"] + data_circular[run_name]["ff_timed"][t]["coherent"]["Icross"]
        data_circular[run_name]["ff_timed"][t]["incoherent"]["Ico"] = (data_circular[run_name]["ff_timed"][t]["incoherent"]["s0"] - data_circular[run_name]["ff_timed"][t]["incoherent"]["s3"]) / 2
        data_circular[run_name]["ff_timed"][t]["incoherent"]["Icross"] = (data_circular[run_name]["ff_timed"][t]["incoherent"]["s0"] + data_circular[run_name]["ff_timed"][t]["incoherent"]["s3"]) / 2
        data_circular[run_name]["ff_timed"][t]["incoherent"]["Itotal"] = data_circular[run_name]["ff_timed"][t]["incoherent"]["Ico"] + data_circular[run_name]["ff_timed"][t]["incoherent"]["Icross"]

        data_circular[run_name]["ff_timed"][t]["enhancement"]["Ico"] = (
            (data_circular[run_name]["ff_timed"][t]["coherent"]["Ico"] + eps)
            / (data_circular[run_name]["ff_timed"][t]["incoherent"]["Ico"] + eps)
        )
        data_circular[run_name]["ff_timed"][t]["enhancement"]["Icross"] = (
            (data_circular[run_name]["ff_timed"][t]["coherent"]["Icross"] + eps)
            / (data_circular[run_name]["ff_timed"][t]["incoherent"]["Icross"] + eps)
        )
        data_circular[run_name]["ff_timed"][t]["enhancement"]["total"] = (
            (data_circular[run_name]["ff_timed"][t]["coherent"]["Itotal"] + eps)
            / (data_circular[run_name]["ff_timed"][t]["incoherent"]["Itotal"] + eps)
        )

    for i_order in range(len(scattering_order_bins)):
        data_circular[run_name]["ff_order"][i_order]["coherent"]["Ico"] = (data_circular[run_name]["ff_order"][i_order]["coherent"]["s0"] - data_circular[run_name]["ff_order"][i_order]["coherent"]["s3"]) / 2
        data_circular[run_name]["ff_order"][i_order]["coherent"]["Icross"] = (data_circular[run_name]["ff_order"][i_order]["coherent"]["s0"] + data_circular[run_name]["ff_order"][i_order]["coherent"]["s3"]) / 2
        data_circular[run_name]["ff_order"][i_order]["coherent"]["Itotal"] = data_circular[run_name]["ff_order"][i_order]["coherent"]["Ico"] + data_circular[run_name]["ff_order"][i_order]["coherent"]["Icross"]
        data_circular[run_name]["ff_order"][i_order]["incoherent"]["Ico"] = (data_circular[run_name]["ff_order"][i_order]["incoherent"]["s0"] - data_circular[run_name]["ff_order"][i_order]["incoherent"]["s3"]) / 2
        data_circular[run_name]["ff_order"][i_order]["incoherent"]["Icross"] = (data_circular[run_name]["ff_order"][i_order]["incoherent"]["s0"] + data_circular[run_name]["ff_order"][i_order]["incoherent"]["s3"]) / 2
        data_circular[run_name]["ff_order"][i_order]["incoherent"]["Itotal"] = data_circular[run_name]["ff_order"][i_order]["incoherent"]["Ico"] + data_circular[run_name]["ff_order"][i_order]["incoherent"]["Icross"]

        data_circular[run_name]["ff_order"][i_order]["enhancement"]["Ico"] = (
            (data_circular[run_name]["ff_order"][i_order]["coherent"]["Ico"] + eps)
            / (data_circular[run_name]["ff_order"][i_order]["incoherent"]["Ico"] + eps)
        )
        data_circular[run_name]["ff_order"][i_order]["enhancement"]["Icross"] = (
            (data_circular[run_name]["ff_order"][i_order]["coherent"]["Icross"] + eps)
            / (data_circular[run_name]["ff_order"][i_order]["incoherent"]["Icross"] + eps)
        )
        data_circular[run_name]["ff_order"][i_order]["enhancement"]["total"] = (
            (data_circular[run_name]["ff_order"][i_order]["coherent"]["Itotal"] + eps)
            / (data_circular[run_name]["ff_order"][i_order]["incoherent"]["Itotal"] + eps)
        )



# # plot total intensityes for each type of particle. The total is on ff_timed[0], the rest of the time bins are for the temporal evolution
# fig, ax = plt.subplots(figsize=(10, 4))
# for run_name in run_names_circular:
#     transport_mean_free_path = data_circular[run_name]["transport_mean_free_path"]
#     ax.plot(theta_mrad, data_circular[run_name]["ff_timed"][0]["enhancement"]["Ico"], label=f"{transport_mean_free_path}", alpha=0.7)
# ax.set_xlabel("Scattering Angle")
# ax.set_ylabel("Enhancement Factor")
# ax.legend()
# plt.tight_layout()


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



for run_name in run_names_circular:
    transport_mean_free_path = data_circular[run_name]["transport_mean_free_path"]
    theta_coherent_milirad = data_circular[run_name]["theta_coherent_mrad"]
    radius = data_circular[run_name]["radius"]
    volume_fraction = data_circular[run_name]["volume_fraction"]

    label = f"$a={radius:.3f}$, $f={volume_fraction:.3f}$"

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for order in scattering_order_bins:
        ax_left.plot(theta_mrad, data_circular[run_name]["ff_order"][scattering_order_bins.index(order)]["coherent"]["Ico"], label=f"order {order}")
        ax_right.plot(theta_mrad, data_circular[run_name]["ff_order"][scattering_order_bins.index(order)]["coherent"]["Icross"], label=f"order {order}")

    plt.tight_layout()
    ax_right.legend()
    ax_left.set_xlabel("Scattering angle (mrad)")
    ax_left.set_ylabel("Enhancement factor")
    plt.savefig(f"{save_path}/cbs-RGD-ff-scattering-order-enhancement-ico-icross-{run_name}.pdf")




plt.show()