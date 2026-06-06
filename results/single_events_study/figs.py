from ..utils.loaders import load_sweep
from ..utils.styles import apply

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple



apply(context="paper", col="single")

save_path = "/home/niaggar/Developer/luminis-mc/temporal_results"
folder = "single_events_study"

eps = 1e-30

sweep_data = load_sweep(folder)
run_names = sorted(sweep_data.keys())


def azimuthal_average(mat):
    """Promedio sobre phi -> perfil 1D en theta. Acepta 1D o 2D."""
    arr = np.asarray(mat)
    if arr.ndim == 1:
        return arr
    return arr.mean(axis=1)



run_name = "0000_radius_0.110_volumefraction_1.000_estimator"
loader = sweep_data[run_name]

p = loader.params
radius = p["radius_um"]
volume_fraction = p["volume_fraction"]
theta_coherent = p.get("theta_coherent_rad", None)
n_photons_estimator = p["n_photons"]

theta = loader.derived("axes/theta_rad")          # rad
theta_mrad = theta * 1e3

class Profiles:
    Coh: Tuple[np.ndarray, np.ndarray, np.ndarray]
    Inc: Tuple[np.ndarray, np.ndarray, np.ndarray]
    Enh: Tuple[np.ndarray, np.ndarray, np.ndarray]

def load_profiles(loader, event=0):
    coh_s0 = azimuthal_average(loader.derived(f"farfield_cbs_{event}/coherent/s0"))
    coh_s3 = azimuthal_average(loader.derived(f"farfield_cbs_{event}/coherent/s3"))
    inc_s0 = azimuthal_average(loader.derived(f"farfield_cbs_{event}/incoherent/s0"))
    inc_s3 = azimuthal_average(loader.derived(f"farfield_cbs_{event}/incoherent/s3"))

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

events = [2, 3, 4, 5, 10, 15, 20, 30, 50, 100, 150, 200, 300, 500, 1000]



profiles = load_profiles(loader, 2)

coh_co, coh_cross, coh_tot = profiles.Coh
inc_co, inc_cross, inc_tot = profiles.Inc
enh_co, enh_cross, enh_tot = profiles.Enh


# ===================================================================
# Figura 1: intensidades coherente vs incoherente (total)
# ===================================================================
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(theta_mrad, coh_tot, label="coherente")
ax.plot(theta_mrad, inc_tot, label="incoherente")
if theta_coherent is not None:
    ax.axvline(theta_coherent * 1e3, color="red", ls="--", alpha=0.6,
               label=r"$1/(k\,\ell^*)$")
ax.set_xlabel("Angulo de dispersion (mrad)")
ax.set_ylabel("Intensidad (u.a.)")
ax.set_title(f"CBS forward+reverse  $a={radius:.3f}\\,\\mu m$, $f={volume_fraction:.2f}$")
ax.legend()
fig.tight_layout()
fig.savefig(f"{save_path}/cbs_intensity_events.pdf")


fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(theta_mrad, enh_co, label="helicidad preservada")
ax.plot(theta_mrad, enh_cross, label="helicidad cruzada")
ax.plot(theta_mrad, enh_tot, label="helicidad total")
if theta_coherent is not None:
    ax.axvline(theta_coherent * 1e3, color="red", ls="--", alpha=0.6,
               label=r"$1/(k\,\ell^*)$")
ax.set_xlabel("Angulo de dispersion (mrad)")
ax.set_ylabel("Enhancement (u.a.)")
ax.set_title(f"Enhancement CBS forward+reverse  $a={radius:.3f}\\,\\mu m$, $f={volume_fraction:.2f}$")
ax.legend()
fig.tight_layout()
fig.savefig(f"{save_path}/cbs_enhancement_events.pdf")
