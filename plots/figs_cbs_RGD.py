from utils.loaders import load_sweep
from utils.styles import apply
import utils.figures as figures
import utils.cbs as cbs
from scipy.ndimage import uniform_filter1d

import numpy as np
import matplotlib.pyplot as plt

apply(context="paper", col="single")

folder = "09Mar26/sim_cbs_RGD_multiple"

data = {}
sweep_data = load_sweep(folder)
scattering_order_bins = [2, 3, 4, 5, 7, 10, 15, 20, 50]

theta_degrees = np.linspace(0, 5, 400)  # De 0 a 5 grados con paso de 0.5 grados
phi_degrees = np.linspace(0, 360, 360)    # De

for run_name, result_loader in sweep_data.items():
    transport_mean_free_path = result_loader.params["mean_free_path_real"] / (1 - result_loader.params["anisotropy_factor"])
    theta_coherent = result_loader.params["wavelength_real"] / (2 * np.pi * transport_mean_free_path) * 180 / np.pi  # Convertir a grados

    meta = result_loader.sensor_meta("farfield_cbs")
    print(meta["hits"])

    params = result_loader.params
    data[run_name] = {
        "anisotropy": params["anisotropy_factor"],
        "mean_free_path": params["mean_free_path_real"],
        "radius": params["radius_real"],
        "size_parameter": params["size_parameter"],
        "condition_1": params["condition_1"],
        "condition_2": params["condition_2"],
        "theta_coherent": theta_coherent,
        "ff": {
            "coherent": {
                "s0": result_loader.derived("farfield_cbs/coherent/s0"),
                "s1": result_loader.derived("farfield_cbs/coherent/s1"),
                "s2": result_loader.derived("farfield_cbs/coherent/s2"),
                "s3": result_loader.derived("farfield_cbs/coherent/s3"),
            },
            "incoherent": {
                "s0": result_loader.derived("farfield_cbs/incoherent/s0"),
                "s1": result_loader.derived("farfield_cbs/incoherent/s1"),
                "s2": result_loader.derived("farfield_cbs/incoherent/s2"),
                "s3": result_loader.derived("farfield_cbs/incoherent/s3"),
            },
            "enhancement": {},
        },
        "scattering_order": [{
            "order": order,
            "coherent": {
                "s0": result_loader.derived(f"farfield_cbs_scattering_order_{order}/coherent/s0"),
                "s1": result_loader.derived(f"farfield_cbs_scattering_order_{order}/coherent/s1"),
                "s2": result_loader.derived(f"farfield_cbs_scattering_order_{order}/coherent/s2"),
                "s3": result_loader.derived(f"farfield_cbs_scattering_order_{order}/coherent/s3"),
            },
            "incoherent": {
                "s0": result_loader.derived(f"farfield_cbs_scattering_order_{order}/incoherent/s0"),
                "s1": result_loader.derived(f"farfield_cbs_scattering_order_{order}/incoherent/s1"),
                "s2": result_loader.derived(f"farfield_cbs_scattering_order_{order}/incoherent/s2"),
                "s3": result_loader.derived(f"farfield_cbs_scattering_order_{order}/incoherent/s3"),
            },
            "enhancement": {},
        } for order in scattering_order_bins]
    }

    data[run_name]["ff"]["coherent"]["Ix"] = (data[run_name]["ff"]["coherent"]["s0"] + data[run_name]["ff"]["coherent"]["s1"]) / 2
    data[run_name]["ff"]["coherent"]["Iy"] = (data[run_name]["ff"]["coherent"]["s0"] - data[run_name]["ff"]["coherent"]["s1"]) / 2
    data[run_name]["ff"]["coherent"]["Itotal"] = data[run_name]["ff"]["coherent"]["Ix"] + data[run_name]["ff"]["coherent"]["Iy"]
    data[run_name]["ff"]["incoherent"]["Ix"] = (data[run_name]["ff"]["incoherent"]["s0"] + data[run_name]["ff"]["incoherent"]["s1"]) / 2
    data[run_name]["ff"]["incoherent"]["Iy"] = (data[run_name]["ff"]["incoherent"]["s0"] - data[run_name]["ff"]["incoherent"]["s1"]) / 2
    data[run_name]["ff"]["incoherent"]["Itotal"] = data[run_name]["ff"]["incoherent"]["Ix"] + data[run_name]["ff"]["incoherent"]["Iy"]
    eps = 1e-30  # pure numerical guard, no physical effect

    data[run_name]["ff"]["enhancement"]["Ix"] = (
        (data[run_name]["ff"]["coherent"]["Ix"] + eps)
        / (data[run_name]["ff"]["incoherent"]["Ix"] + eps)
    )
    data[run_name]["ff"]["enhancement"]["Iy"] = (
        (data[run_name]["ff"]["coherent"]["Iy"] + eps)
        / (data[run_name]["ff"]["incoherent"]["Iy"] + eps)
    )
    data[run_name]["ff"]["enhancement"]["total"] = (
        (data[run_name]["ff"]["coherent"]["Itotal"] + eps)
        / (data[run_name]["ff"]["incoherent"]["Itotal"] + eps)
    )
    for order_data in data[run_name]["scattering_order"]:
        order_data["coherent"]["Ix"] = (order_data["coherent"]["s0"] + order_data["coherent"]["s1"]) / 2
        order_data["coherent"]["Iy"] = (order_data["coherent"]["s0"] - order_data["coherent"]["s1"]) / 2
        order_data["coherent"]["Itotal"] = order_data["coherent"]["Ix"] + order_data["coherent"]["Iy"]
        order_data["incoherent"]["Ix"] = (order_data["incoherent"]["s0"] + order_data["incoherent"]["s1"]) / 2
        order_data["incoherent"]["Iy"] = (order_data["incoherent"]["s0"] - order_data["incoherent"]["s1"]) / 2
        order_data["incoherent"]["Itotal"] = order_data["incoherent"]["Ix"] + order_data["incoherent"]["Iy"]
        order_data["enhancement"]["total"] = (order_data["coherent"]["Itotal"] + eps) / (order_data["incoherent"]["Itotal"] + eps)   
        order_data["enhancement"]["Ix"] = (order_data["coherent"]["Ix"] + eps) / (order_data["incoherent"]["Ix"] + eps)
        order_data["enhancement"]["Iy"] = (order_data["coherent"]["Iy"] + eps) / (order_data["incoherent"]["Iy"] + eps)


print("Data loaded for runs:", list(data.keys()))


# figures.plot_cbs_disk(
#     I_coherent=coherent_test,
#     I_incoherent=incoherent_test,
#     theta_degrees=theta_degrees,
#     phi_degrees=phi_degrees,
#     col_titles    = [r'$Ix + Iy$', r'$Ix$', r'$Iy$'],
#     title         = 'CBS - Linearly Polarised Light',
#     cmap          = 'inferno',
#     dpi=100
# )

# plt.show()


# fig = plt.figure(figsize=(6, 4))
# for run_name, run_data in data.items():
#     theta_sym, I_sym = cbs.get_slice(run_data["ff"]["enhancement"]["Ix"], theta_degrees, phi_degrees, phi_cut=0.0)
#     I_sym_smooth = uniform_filter1d(I_sym, size=5)  # Suavizar con un filtro de media móvil
#     plt.plot(np.deg2rad(theta_sym), I_sym_smooth, label=run_name, markersize=4)

#     # vertical line at the coherent theta
#     plt.axvline(np.deg2rad(run_data["theta_coherent"]), color='gray', linestyle='--', linewidth=0.8, label=f'Coherent Theta ({run_name})')

# plt.xlabel("Theta (rad)")
# plt.ylabel("Intensity")
# plt.title("CBS Slice at phi=0°")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.xlim(-0.02, 0.02)
# plt.show()




# ─── helpers ────────────────────────────────────────────────────────────────

def deg_to_mrad(deg):
    return deg * (1000 * np.pi / 180)

def get_sym_slice(enhancement_2d, theta_deg, phi_deg, phi_cut, smooth_size=5):
    """
    Returns (theta_mrad_sym, I_sym) — negative and positive angles,
    X or Y scan, smoothed with 5-point moving average.
    """
    theta_sym, I = cbs.get_slice(enhancement_2d, theta_deg, phi_deg, phi_cut=phi_cut)
    I_smooth = uniform_filter1d(I, size=smooth_size)
    return deg_to_mrad(theta_sym), I_smooth

# sorted run list for consistent ordering
run_names = sorted(data.keys())
labels = {rn: data[rn].get("radius", rn) for rn in run_names}  # use radius as label if available

# Use a label based on radius or run name
def get_label(run_name):
    r = data[run_name].get("radius", None)
    if r is None or r == 0:
        return "Rayleigh"
    return rf"$a={r*1000:.0f}$ nm"

linestyles = ["-", "--", ":", "-."]

# ─── Fig 2: Co-pol and cross-pol, all runs, X-scan (phi=0) ──────────────────

fig2, (ax_co, ax_cross) = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

for i, rn in enumerate(run_names):
    ls = linestyles[i % len(linestyles)]
    label = get_label(rn)

    theta_mrad, I_co = get_sym_slice(data[rn]["ff"]["enhancement"]["Ix"], theta_degrees, phi_degrees, phi_cut=0.0)
    theta_mrad, I_cross = get_sym_slice(data[rn]["ff"]["enhancement"]["Iy"], theta_degrees, phi_degrees, phi_cut=0.0)

    ax_co.plot(theta_mrad,    I_co,    ls, label=label)
    ax_cross.plot(theta_mrad, I_cross, ls, label=label)

for ax, title in [(ax_co, "(a) Co-polarized"), (ax_cross, "(b) Cross-polarized")]:
    ax.axhline(1.0, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("Reduced scattering angle (mrad)")
    ax.set_ylabel("Normalized intensity")
    ax.set_title(title)
    ax.set_xlim(-20, 20)
    ax.legend(fontsize=8)

fig2.suptitle("Fig. 2 — CBS intensity distributions")
fig2.tight_layout()


# ─── Fig 3: Spatial anisotropy (X-scan vs Y-scan, co-pol), one panel/run ───

n_runs = len(run_names)
fig3, axes3 = plt.subplots(1, n_runs, figsize=(3.5 * n_runs, 4), sharey=True)
if n_runs == 1:
    axes3 = [axes3]

for ax, rn in zip(axes3, run_names):
    theta_X, I_X = get_sym_slice(
        data[rn]["ff"]["enhancement"]["Ix"], theta_degrees, phi_degrees, phi_cut=0.0)
    theta_Y, I_Y = get_sym_slice(
        data[rn]["ff"]["enhancement"]["Ix"], theta_degrees, phi_degrees, phi_cut=90.0)

    ax.plot(theta_X, I_X, "-",  lw=1.5, label="X-scan")
    ax.plot(theta_Y, I_Y, "--", lw=1.0, label="Y-scan")
    ax.axhline(1.0, color="gray", lw=0.6, ls=":")
    ax.set_title(get_label(rn))
    ax.set_xlabel("Reduced scattering angle (mrad)")
    ax.set_xlim(-20, 20)
    ax.legend(fontsize=7)

axes3[0].set_ylabel("Normalized intensity")
fig3.suptitle("Fig. 3 — Spatial anisotropy of co-polarized component")
fig3.tight_layout()


# # ─── Fig 4: Per-order decomposition, one figure per run ─────────────────────

for rn in run_names:
    fig4, ax4 = plt.subplots(figsize=(6, 4))

    for order_data in data[rn]["scattering_order"]:
        n = order_data["order"]
        theta_X, I_X = get_sym_slice(order_data["enhancement"]["Ix"], theta_degrees, phi_degrees, phi_cut=0.0)
        theta_Y, I_Y = get_sym_slice(order_data["enhancement"]["Ix"], theta_degrees, phi_degrees, phi_cut=90.0)

        ax4.plot(theta_X, I_X, "-",  lw=1.5, label=f"n={n}")
        # ax4.plot(theta_Y, I_Y, "--", lw=0.8, color=color)  # fine line = Y-scan

    ax4.axhline(1.0, color="gray", lw=0.6, ls=":")
    ax4.set_xlabel("Reduced scattering angle (mrad)")
    ax4.set_ylabel("Normalized intensity")
    ax4.set_title(f"Fig. 4 — {get_label(rn)}  (bold=X-scan, dashed=Y-scan)")
    ax4.set_xlim(-20, 20)
    ax4.legend(fontsize=7, ncol=2)
    fig4.tight_layout()


plt.show()