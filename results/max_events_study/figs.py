from ..utils.loaders import load_sweep
from ..utils.styles import apply

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple



apply(context="paper", col="single")

save_path = "/home/niaggar/Developer/luminis-mc/temporal_results"
folder = "max_events_study"

eps = 1e-30

sweep_data = load_sweep(folder)
run_names = sorted(sweep_data.keys())

max_events = [5, 10, 20, 50, 100, 200, 1000, 2000, 5000, 10000]


def azimuthal_average(mat):
    """Promedio sobre phi -> perfil 1D en theta. Acepta 1D o 2D."""
    arr = np.asarray(mat)
    if arr.ndim == 1:
        return arr
    return arr.mean(axis=1)


class Profiles:
    Coh: Tuple[np.ndarray, np.ndarray, np.ndarray]
    Inc: Tuple[np.ndarray, np.ndarray, np.ndarray]
    Enh: Tuple[np.ndarray, np.ndarray, np.ndarray]

def load_profiles(loader):
    coh_s0 = azimuthal_average(loader.derived(f"farfield_cbs/coherent/s0"))
    coh_s3 = azimuthal_average(loader.derived(f"farfield_cbs/coherent/s3"))
    inc_s0 = azimuthal_average(loader.derived(f"farfield_cbs/incoherent/s0"))
    inc_s3 = azimuthal_average(loader.derived(f"farfield_cbs/incoherent/s3"))

    # Promedio azimutal → perfil 1D en theta
    coh_s0 = azimuthal_average(coh_s0)
    coh_s3 = azimuthal_average(coh_s3)
    inc_s0 = azimuthal_average(inc_s0)
    inc_s3 = azimuthal_average(inc_s3)

    # Descomposición en canales circulares
    coh_co    = (coh_s0 - coh_s3) / 2.0   # helicidad preservada
    coh_cross = (coh_s0 + coh_s3) / 2.0   # helicidad revertida
    coh_tot   = coh_co + coh_cross         # = coh_s0

    inc_co    = (inc_s0 - inc_s3) / 2.0
    inc_cross = (inc_s0 + inc_s3) / 2.0
    inc_tot   = inc_co + inc_cross

    # Enhancement (evitar NaN en bins vacíos)
    enh_co    = (coh_co    + eps) / (inc_co    + eps)
    enh_cross = (coh_cross + eps) / (inc_cross + eps)
    enh_tot   = (coh_tot   + eps) / (inc_tot   + eps)

    out = Profiles()
    out.Coh   = (coh_co, coh_cross, coh_tot)
    out.Inc   = (inc_co, inc_cross, inc_tot)
    out.Enh = (enh_co, enh_cross, enh_tot)
    return out


data_profiles = []

p = sweep_data["0000_maxevents_5"].params
radius = p["radius_um"]
volume_fraction = p["volume_fraction"]
theta_coherent = p.get("theta_coherent_rad", None)
n_photons_estimator = p["n_photons"]

theta = sweep_data["0000_maxevents_5"].derived("axes/theta_rad")          # rad
theta_mrad = theta * 1e3


for max_ev in max_events:
    run_name = f"0000_maxevents_{max_ev}"
    loader = sweep_data[run_name]
    profiles = load_profiles(loader)
    data_profiles.append((max_ev, profiles))




# ===================================================================
# Figura 1: intensidades coherente vs incoherente (total)
# ===================================================================
fig, ax = plt.subplots(figsize=(6, 4))

for max_ev, profiles in data_profiles:
    coh_co, coh_cross, coh_tot = profiles.Coh
    inc_co, inc_cross, inc_tot = profiles.Inc
    enh_co, enh_cross, enh_tot = profiles.Enh

    label = f"max_events={max_ev}"

    ax.plot(theta_mrad, coh_tot, label=label+" (coh)")
    # ax.plot(theta_mrad, inc_tot, label=label+" (inc)")

if theta_coherent is not None:
    ax.axvline(theta_coherent * 1e3, color="red", ls="--", alpha=0.6,
               label=r"$1/(k\,\ell^*)$")
ax.set_xlabel("Angulo de dispersion (mrad)")
ax.set_ylabel("Intensidad (u.a.)")
ax.set_title(f"CBS forward+reverse  $a={radius:.3f}\\,\\mu m$, $f={volume_fraction:.2f}$")
ax.legend()
fig.tight_layout()
fig.savefig(f"{save_path}/cbs_intensity_max_events.pdf")


fig, ax = plt.subplots(figsize=(6, 4))

for max_ev, profiles in data_profiles:
    coh_co, coh_cross, coh_tot = profiles.Coh
    inc_co, inc_cross, inc_tot = profiles.Inc
    enh_co, enh_cross, enh_tot = profiles.Enh

    label = f"max_events={max_ev}"

    ax.plot(theta_mrad, enh_co, label=label+" (coh)")
    # ax.plot(theta_mrad, enh_cross, label=label+" (cross)")
    # ax.plot(theta_mrad, enh_tot, label=label+" (total)")

if theta_coherent is not None:
    ax.axvline(theta_coherent * 1e3, color="red", ls="--", alpha=0.6,
               label=r"$1/(k\,\ell^*)$")
ax.set_xlabel("Angulo de dispersion (mrad)")
ax.set_ylabel("Enhancement (u.a.)")
ax.set_title(f"Enhancement CBS forward+reverse  $a={radius:.3f}\\,\\mu m$, $f={volume_fraction:.2f}$")
ax.legend()
fig.tight_layout()
fig.savefig(f"{save_path}/cbs_enhancement_max_events.pdf")
