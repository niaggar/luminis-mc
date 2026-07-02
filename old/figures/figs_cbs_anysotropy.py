from utils.loaders import load_sweep
from utils.styles import apply
import utils.figures as figures
import utils.cbs as cbs

from luminis_mc import MiePhaseFunction

import numpy as np
import matplotlib.pyplot as plt

apply(context="paper", col="single")
save_path = "/Users/niaggar/Documents/Thesis/Results"




MAX_MRAD = 70

folder_linear = "23Feb26/RESULT-sim_cbs-linear-res"
data_linear = {}
sweep_data_linear = load_sweep(folder_linear)
radius_linear = [0.05, 0.3, 0.5, 1.0]

theta_degrees_linear = np.linspace(0, 40, 500)
phi_degrees_linear = np.linspace(0, 360, 100)
run_names_linear = sorted(sweep_data_linear.keys())

N_PHOTONS_LINEAR = 1_000_000_000



folder_circular = "23Feb26/RESULT-sim_cbs-circular"
data_circular = {}
sweep_data_circular = load_sweep(folder_circular)
radius_circular = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0]

theta_degrees_circular = np.linspace(0, 40, 200)
phi_degrees_circular = np.linspace(0, 360, 60)
run_names_circular = sorted(sweep_data_circular.keys())

N_PHOTONS_CIRCULAR = 500_000_000



for run_name, result_loader in sweep_data_circular.items():
    wavelength_real = result_loader.params["wavelength_real"]
    radius_real = result_loader.params["radius_real"]
    n_particle_real = result_loader.params["n_particle_real"]
    n_medium_real = result_loader.params["n_medium_real"]
    phasef_theta_min = result_loader.params["phasef_theta_min"]
    phasef_theta_max = result_loader.params["phasef_theta_max"]
    phasef_ndiv = result_loader.params["phasef_ndiv"]

    phase = MiePhaseFunction(wavelength_real, radius_real, n_particle_real, n_medium_real, phasef_ndiv, phasef_theta_min, phasef_theta_max)
    anysotropy = phase.get_anisotropy_factor()

    transport_mean_free_path = result_loader.params["mean_free_path_sim"] / (1 - anysotropy[0])
    theta_coherent = wavelength_real / (2 * np.pi * transport_mean_free_path) * 180 / np.pi  # Convertir a grados
    size_parameter = 2 * np.pi * radius_real / wavelength_real

    theta_coherent_milirad = theta_coherent * (1000 * np.pi / 180)

    params = result_loader.params
    data_circular[run_name] = {
        "anisotropy": anysotropy[0],
        "mean_free_path": params["mean_free_path_sim"],
        "radius": radius_real,
        "size_parameter": size_parameter,
        "theta_coherent": theta_coherent,
        "theta_coherent_mrad": theta_coherent_milirad,
        "ff": {
            "coherent": {
                "s0": result_loader.derived("farfield_cbs/S0_coh") / N_PHOTONS_CIRCULAR,
                "s1": result_loader.derived("farfield_cbs/S1_coh") / N_PHOTONS_CIRCULAR,
                "s2": result_loader.derived("farfield_cbs/S2_coh") / N_PHOTONS_CIRCULAR,
                "s3": result_loader.derived("farfield_cbs/S3_coh") / N_PHOTONS_CIRCULAR,
            },
            "incoherent": {
                "s0": result_loader.derived("farfield_cbs/S0_inc") / N_PHOTONS_CIRCULAR,
                "s1": result_loader.derived("farfield_cbs/S1_inc") / N_PHOTONS_CIRCULAR,
                "s2": result_loader.derived("farfield_cbs/S2_inc") / N_PHOTONS_CIRCULAR,
                "s3": result_loader.derived("farfield_cbs/S3_inc") / N_PHOTONS_CIRCULAR,
            },
            "enhancement": {},
        },
    }

    data_circular[run_name]["ff"]["coherent"]["Ico"] = (data_circular[run_name]["ff"]["coherent"]["s0"] - data_circular[run_name]["ff"]["coherent"]["s3"]) / 2
    data_circular[run_name]["ff"]["coherent"]["Icross"] = (data_circular[run_name]["ff"]["coherent"]["s0"] + data_circular[run_name]["ff"]["coherent"]["s3"]) / 2
    data_circular[run_name]["ff"]["coherent"]["Itotal"] = data_circular[run_name]["ff"]["coherent"]["Ico"] + data_circular[run_name]["ff"]["coherent"]["Icross"]
    data_circular[run_name]["ff"]["incoherent"]["Ico"] = (data_circular[run_name]["ff"]["incoherent"]["s0"] - data_circular[run_name]["ff"]["incoherent"]["s3"]) / 2
    data_circular[run_name]["ff"]["incoherent"]["Icross"] = (data_circular[run_name]["ff"]["incoherent"]["s0"] + data_circular[run_name]["ff"]["incoherent"]["s3"]) / 2
    data_circular[run_name]["ff"]["incoherent"]["Itotal"] = data_circular[run_name]["ff"]["incoherent"]["Ico"] + data_circular[run_name]["ff"]["incoherent"]["Icross"]
    eps = 1e-30  # pure numerical guard, no physical effect

    data_circular[run_name]["ff"]["enhancement"]["Ico"] = (
        (data_circular[run_name]["ff"]["coherent"]["Ico"] + eps)
        / (data_circular[run_name]["ff"]["incoherent"]["Ico"] + eps)
    )
    data_circular[run_name]["ff"]["enhancement"]["Icross"] = (
        (data_circular[run_name]["ff"]["coherent"]["Icross"] + eps)
        / (data_circular[run_name]["ff"]["incoherent"]["Icross"] + eps)
    )
    data_circular[run_name]["ff"]["enhancement"]["total"] = (
        (data_circular[run_name]["ff"]["coherent"]["Itotal"] + eps)
        / (data_circular[run_name]["ff"]["incoherent"]["Itotal"] + eps)
    )


for run_name, result_loader in sweep_data_linear.items():
    wavelength_real = result_loader.params["wavelength_real"]
    radius_real = result_loader.params["radius_real"]
    n_particle_real = result_loader.params["n_particle_real"]
    n_medium_real = result_loader.params["n_medium_real"]
    phasef_theta_min = result_loader.params["phasef_theta_min"]
    phasef_theta_max = result_loader.params["phasef_theta_max"]
    phasef_ndiv = result_loader.params["phasef_ndiv"]

    phase = MiePhaseFunction(wavelength_real, radius_real, n_particle_real, n_medium_real, phasef_ndiv, phasef_theta_min, phasef_theta_max)
    anysotropy = phase.get_anisotropy_factor()

    transport_mean_free_path = result_loader.params["mean_free_path_sim"] / (1 - anysotropy[0])
    theta_coherent = wavelength_real / (2 * np.pi * transport_mean_free_path) * 180 / np.pi  # Convertir a grados
    size_parameter = 2 * np.pi * radius_real / wavelength_real

    theta_coherent_milirad = theta_coherent * (1000 * np.pi / 180)

    params = result_loader.params
    data_linear[run_name] = {
        "anisotropy": anysotropy[0],
        "mean_free_path": params["mean_free_path_sim"],
        "radius": radius_real,
        "size_parameter": size_parameter,
        "theta_coherent": theta_coherent,
        "theta_coherent_mrad": theta_coherent_milirad,
        "ff": {
            "coherent": {
                "s0": result_loader.derived("farfield_cbs/S0_coh") / N_PHOTONS_LINEAR,
                "s1": result_loader.derived("farfield_cbs/S1_coh") / N_PHOTONS_LINEAR,
                "s2": result_loader.derived("farfield_cbs/S2_coh") / N_PHOTONS_LINEAR,
                "s3": result_loader.derived("farfield_cbs/S3_coh") / N_PHOTONS_LINEAR,
            },
            "incoherent": {
                "s0": result_loader.derived("farfield_cbs/S0_inc") / N_PHOTONS_LINEAR,
                "s1": result_loader.derived("farfield_cbs/S1_inc") / N_PHOTONS_LINEAR,
                "s2": result_loader.derived("farfield_cbs/S2_inc") / N_PHOTONS_LINEAR,
                "s3": result_loader.derived("farfield_cbs/S3_inc") / N_PHOTONS_LINEAR,
            },
            "enhancement": {},
        },
    }

    data_linear[run_name]["ff"]["coherent"]["Ico"] = (data_linear[run_name]["ff"]["coherent"]["s0"] - data_linear[run_name]["ff"]["coherent"]["s3"]) / 2
    data_linear[run_name]["ff"]["coherent"]["Icross"] = (data_linear[run_name]["ff"]["coherent"]["s0"] + data_linear[run_name]["ff"]["coherent"]["s3"]) / 2

    data_linear[run_name]["ff"]["coherent"]["Ix"] = (data_linear[run_name]["ff"]["coherent"]["s0"] + data_linear[run_name]["ff"]["coherent"]["s1"]) / 2
    data_linear[run_name]["ff"]["coherent"]["Iy"] = (data_linear[run_name]["ff"]["coherent"]["s0"] - data_linear[run_name]["ff"]["coherent"]["s1"]) / 2 
    data_linear[run_name]["ff"]["coherent"]["Itotal"] = data_linear[run_name]["ff"]["coherent"]["Ix"] + data_linear[run_name]["ff"]["coherent"]["Iy"]

    data_linear[run_name]["ff"]["incoherent"]["Ico"] = (data_linear[run_name]["ff"]["incoherent"]["s0"] - data_linear[run_name]["ff"]["incoherent"]["s3"]) / 2
    data_linear[run_name]["ff"]["incoherent"]["Icross"] = (data_linear[run_name]["ff"]["incoherent"]["s0"] + data_linear[run_name]["ff"]["incoherent"]["s3"]) / 2

    data_linear[run_name]["ff"]["incoherent"]["Ix"] = (data_linear[run_name]["ff"]["incoherent"]["s0"] + data_linear[run_name]["ff"]["incoherent"]["s1"]) / 2
    data_linear[run_name]["ff"]["incoherent"]["Iy"] = (data_linear[run_name]["ff"]["incoherent"]["s0"] - data_linear[run_name]["ff"]["incoherent"]["s1"]) / 2 
    data_linear[run_name]["ff"]["incoherent"]["Itotal"] = data_linear[run_name]["ff"]["incoherent"]["Ix"] + data_linear[run_name]["ff"]["incoherent"]["Iy"]


    eps = 1e-30  # pure numerical guard, no physical effect

    data_linear[run_name]["ff"]["enhancement"]["Ico"] = (
        (data_linear[run_name]["ff"]["coherent"]["Ico"] + eps)
        / (data_linear[run_name]["ff"]["incoherent"]["Ico"] + eps)
    )
    data_linear[run_name]["ff"]["enhancement"]["Icross"] = (
        (data_linear[run_name]["ff"]["coherent"]["Icross"] + eps)
        / (data_linear[run_name]["ff"]["incoherent"]["Icross"] + eps)
    )
    data_linear[run_name]["ff"]["enhancement"]["total"] = (
        (data_linear[run_name]["ff"]["coherent"]["Itotal"] + eps)
        / (data_linear[run_name]["ff"]["incoherent"]["Itotal"] + eps)
    )
    data_linear[run_name]["ff"]["enhancement"]["Ix"] = (
        (data_linear[run_name]["ff"]["coherent"]["Ix"] + eps)
        / (data_linear[run_name]["ff"]["incoherent"]["Ix"] + eps)
    )
    data_linear[run_name]["ff"]["enhancement"]["Iy"] = (
        (data_linear[run_name]["ff"]["coherent"]["Iy"] + eps)
        / (data_linear[run_name]["ff"]["incoherent"]["Iy"] + eps)
    )




# ------ Diks
# for run_name in run_names_circular:
#     figures.plot_disk(
#         Icoh=(
#             data_circular[run_name]["ff"]["coherent"]["Itotal"],
#             data_circular[run_name]["ff"]["coherent"]["Ico"],
#             data_circular[run_name]["ff"]["coherent"]["Icross"],
#         ),
#         Iinc=(
#             data_circular[run_name]["ff"]["incoherent"]["Itotal"],
#             data_circular[run_name]["ff"]["incoherent"]["Ico"],
#             data_circular[run_name]["ff"]["incoherent"]["Icross"],
#         ),
#         labels=["Total", "Co-polarized", "Cross-polarized"],
#         theta_deg=theta_degrees_circular,
#         phi_data=np.deg2rad(phi_degrees_circular),
#         title=f"CBS Disk Plot (Circular) - Radius {data_circular[run_name]['radius']:.3f}",
#         save_path=f"{save_path}/cbs-mie-onelayer-circular_{run_name}.png"
#     )
# for run_name in run_names_linear:
#     figures.plot_disk(
#         Icoh=(
#             data_linear[run_name]["ff"]["coherent"]["Itotal"],
#             data_linear[run_name]["ff"]["coherent"]["Ix"],
#             data_linear[run_name]["ff"]["coherent"]["Iy"],
#         ),
#         Iinc=(
#             data_linear[run_name]["ff"]["incoherent"]["Itotal"],
#             data_linear[run_name]["ff"]["incoherent"]["Ix"],
#             data_linear[run_name]["ff"]["incoherent"]["Iy"],
#         ),
#         labels=["Total", "Co-polarized", "Cross-polarized"],
#         theta_deg=theta_degrees_linear,
#         phi_data=np.deg2rad(phi_degrees_linear),
#         title=f"CBS Disk Plot (Linear) - Radius {data_linear[run_name]['radius']:.3f}",
#         save_path=f"{save_path}/cbs-mie-onelayer-linear_{run_name}.png"
#     )






# ------ Radial profiles linear

# MAX_MRAD = 200
# run_names = [
#     "0000_radius_0.050",
#     "0001_radius_0.300",
#     "0003_radius_1.000",
# ]

# I_left = []
# I_right = []
# theta_mrad = None
# for i, rn in enumerate(run_names):
#     theta_max = data_linear[rn]["theta_coherent_mrad"]

#     theta_mrad, I_co = cbs.get_sym_slice(data_linear[rn]["ff"]["enhancement"]["Ix"], theta_degrees_linear, phi_degrees_linear, phi_cut=0.0, smooth_size=2)
#     _, I_cross = cbs.get_sym_slice(data_linear[rn]["ff"]["enhancement"]["Iy"], theta_degrees_linear, phi_degrees_linear, phi_cut=0.0, smooth_size=2)

#     I_left.append(I_co)
#     I_right.append(I_cross)

# figures.plot_profiles_left_right(
#     left_series=I_left,
#     right_series=I_right,
#     labels_left=[figures.get_label(data_linear, rn) for rn in run_names],
#     labels_right=[figures.get_label(data_linear, rn) for rn in run_names],
#     x_axis=theta_mrad,
#     title="CBS intensity distributions (Linear)",
#     title_left="Co-polarized ($I_x$)",
#     title_right="Cross-polarized ($I_y$)",
#     xlabel="Reduced scattering angle (mrad)",
#     ylabel="Normalized intensity",
#     save_path=f"{save_path}/cbs-mie-onelayer-intensity-profiles-linear-enhancement_co_vs_cross.png",
#     max_x=MAX_MRAD
# )

# I_left = []
# I_right = []
# theta_mrad = None
# for i, rn in enumerate(run_names):
#     theta_max = data_linear[rn]["theta_coherent_mrad"]

#     theta_mrad, I_co = cbs.get_sym_slice(data_linear[rn]["ff"]["enhancement"]["Ix"], theta_degrees_linear, phi_degrees_linear, phi_cut=0.0, smooth_size=2)
#     _, I_cross = cbs.get_sym_slice(data_linear[rn]["ff"]["enhancement"]["Ix"], theta_degrees_linear, phi_degrees_linear, phi_cut=90.0, smooth_size=2)

#     I_left.append(I_co)
#     I_right.append(I_cross)

# figures.plot_profiles_left_right(
#     left_series=I_left,
#     right_series=I_right,
#     labels_left=[figures.get_label(data_linear, rn) for rn in run_names],
#     labels_right=[figures.get_label(data_linear, rn) for rn in run_names],
#     x_axis=theta_mrad,
#     title="Copolarized intensity profiles (Linear)",
#     title_left=r"X-scan ($\phi = 0^\circ$)",
#     title_right=r"Y-scan ($\phi = 90^\circ$)",
#     xlabel="Reduced scattering angle (mrad)",
#     ylabel="Normalized intensity",
#     save_path=f"{save_path}/cbs-mie-onelayer-intensity-profiles-linear-enhancement_coherent_x_vs_y.png",
#     max_x=MAX_MRAD
# )





# ------ Radial profiles Circular
# Spatial anisotropy here is 0, so no difference between X and Y scans, it can be integrated all the phi angles without losing information. The plot would be the same for any phi_cut.

# I_left = []
# I_right = []
# theta_mrad = np.deg2rad(theta_degrees_circular) * (1000 * np.pi / 180)

# run_names = [
#     "0000_radius_0.050",
#     "0002_radius_0.200",
#     "0003_radius_0.300",
#     "0007_radius_1.000",
# ]

# for i, rn in enumerate(run_names):
#     theta_max = data_circular[rn]["theta_coherent_mrad"]

#     matrix_co = data_circular[rn]["ff"]["enhancement"]["Ico"]
#     matrix_cross = data_circular[rn]["ff"]["enhancement"]["Icross"]

#     I_co = np.mean(matrix_co, axis=1)
#     I_cross = np.mean(matrix_cross, axis=1)

#     I_left.append(I_co)
#     I_right.append(I_cross)

# figures.plot_profiles_left_right(
#     left_series=I_left,
#     right_series=I_right,
#     labels_left=[figures.get_label(data_circular, rn) for rn in run_names],
#     labels_right=[figures.get_label(data_circular, rn) for rn in run_names],
#     x_axis=theta_mrad,
#     title="CBS intensity distributions (Circular)",
#     title_left="Co-polarized ($I_{co}$)",
#     title_right="Cross-polarized ($I_{cross}$)",
#     xlabel="Reduced scattering angle (mrad)",
#     ylabel="Normalized intensity",
#     save_path=f"{save_path}/cbs-mie-onelayer-intensity-profiles-circular-enhancement_co_vs_cross.png",
#     max_x=np.deg2rad(30) * (1000 * np.pi / 180),
#     min_x=0
# )






# ------ Anysotropy analysis
# int_anisotropy_circular = []
# radii_nm_circular = []

# int_anisotropy_linear = []
# radii_nm_linear = []

# for rn in run_names_circular:
#     theta_mrad, I_X = cbs.get_sym_slice(data_circular[rn]["ff"]["enhancement"]["Ico"], theta_degrees_circular, phi_degrees_circular, phi_cut=0.0)
#     _, I_Y = cbs.get_sym_slice(data_circular[rn]["ff"]["enhancement"]["Ico"], theta_degrees_circular, phi_degrees_circular, phi_cut=90.0)

#     Ai = cbs.profile_anisotropy(theta_mrad, I_X, I_Y, angle_limit=MAX_MRAD)
#     rad = data_circular[rn]["radius"]
#     if not np.isclose(rad, radius_linear).any():
#         continue
#     int_anisotropy_circular.append(Ai)
#     radii_nm_circular.append(data_circular[rn]["radius"] * 1000)

# for rn in run_names_linear:
#     theta_mrad, I_X = cbs.get_sym_slice(data_linear[rn]["ff"]["enhancement"]["Ico"], theta_degrees_linear, phi_degrees_linear, phi_cut=0.0)
#     _, I_Y = cbs.get_sym_slice(data_linear[rn]["ff"]["enhancement"]["Ico"], theta_degrees_linear, phi_degrees_linear, phi_cut=90.0)

#     Ai = cbs.profile_anisotropy(theta_mrad, I_X, I_Y, angle_limit=MAX_MRAD)
#     int_anisotropy_linear.append(Ai)
#     radii_nm_linear.append(data_linear[rn]["radius"] * 1000)

# fig4, ax4 = plt.subplots(1, 1, figsize=(6, 4))
# ax4.plot(radii_nm_circular, int_anisotropy_circular, "o-", lw=1.5, ms=6)
# ax4.plot(radii_nm_linear, int_anisotropy_linear, "s-", lw=1.5, ms=6)
# ax4.axhline(0.0, color="gray", lw=0.8, ls="--", label="Isotropic ($A_i = 0$)")
# ax4.set_xlabel("Particle radius $a$ (nm)")
# ax4.set_ylabel(r"Integrated anisotropy $A_i$")
# ax4.set_title("Fig. 4 — Integrated spatial anisotropy (co-polarized, circular)")
# ax4.legend()
# fig4.tight_layout()
# plt.savefig(f"{save_path}/cbs-mie-onelayer-anisotropy-linear-vs-circular.png", dpi=300)




# ------ Show all figures
plt.show()