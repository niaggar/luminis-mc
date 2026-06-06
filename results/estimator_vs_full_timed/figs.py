from results.utils.loaders import load_sweep
from results.utils.styles import apply

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple



apply(context="paper", col="single")

save_path = "/home/niaggar/Developer/luminis-mc/temporal_results"
folder = "estimator_vs_full_timed"

eps = 1e-30

sweep_data = load_sweep(folder)
run_names = sorted(sweep_data.keys())


def azimuthal_average(mat):
    """Promedio sobre phi -> perfil 1D en theta. Acepta 1D o 2D."""
    arr = np.asarray(mat)
    if arr.ndim == 1:
        return arr
    return arr.mean(axis=1)



run_name_estimator = "0000_radius_0.110_volumefraction_1.000_estimator"
run_name_full = "0000_radius_0.110_volumefraction_1.000_full"



loader_est = sweep_data[run_name_estimator]
loader_full = sweep_data[run_name_full]

p = loader_est.params
radius = p["radius_um"]
volume_fraction = p["volume_fraction"]
theta_coherent = p.get("theta_coherent_rad", None)
n_photons_estimator = p["n_photons"]

theta = loader_est.derived("axes/theta_rad")          # rad
theta_mrad = theta * 1e3

class Profiles:
    Coh: Tuple[np.ndarray, np.ndarray, np.ndarray]
    Inc: Tuple[np.ndarray, np.ndarray, np.ndarray]
    Enh: Tuple[np.ndarray, np.ndarray, np.ndarray]

def load_profiles(loader, time_index=0):
    coh_s0 = azimuthal_average(loader.derived(f"farfield_cbs/coherent/s0_t{time_index}"))
    coh_s3 = azimuthal_average(loader.derived(f"farfield_cbs/coherent/s3_t{time_index}"))
    inc_s0 = azimuthal_average(loader.derived(f"farfield_cbs/incoherent/s0_t{time_index}"))
    inc_s3 = azimuthal_average(loader.derived(f"farfield_cbs/incoherent/s3_t{time_index}"))

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


profiles_est = load_profiles(loader_est, 10)
profiles_full = load_profiles(loader_full, 10)

coh_co_est, coh_cross_est, coh_tot_est = profiles_est.Coh
inc_co_est, inc_cross_est, inc_tot_est = profiles_est.Inc
enh_co_est, enh_cross_est, enh_tot_est = profiles_est.Enh

coh_co_full, coh_cross_full, coh_tot_full = profiles_full.Coh
inc_co_full, inc_cross_full, inc_tot_full = profiles_full.Inc
enh_co_full, enh_cross_full, enh_tot_full = profiles_full.Enh


# ===================================================================
# Figura 1: intensidades coherente vs incoherente (total)
# ===================================================================
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(theta_mrad, coh_tot_est, label="coherente (estimator)")
ax.plot(theta_mrad, inc_tot_est, label="incoherente (estimator)")
ax.plot(theta_mrad, coh_tot_full, label="coherente (full)", ls="--")
ax.plot(theta_mrad, inc_tot_full, label="incoherente (full)", ls="--")
if theta_coherent is not None:
    ax.axvline(theta_coherent * 1e3, color="red", ls="--", alpha=0.6,
               label=r"$1/(k\,\ell^*)$")
ax.set_xlabel("Angulo de dispersion (mrad)")
ax.set_ylabel("Intensidad (u.a.)")
ax.set_title(f"CBS forward+reverse  $a={radius:.3f}\\,\\mu m$, $f={volume_fraction:.2f}$")
ax.legend()
fig.tight_layout()
fig.savefig(f"{save_path}/cbs-test-intensity-comparison-{run_name_estimator}.pdf")

# ===================================================================
# Figura 2: factor de realce por canal
# ===================================================================
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(theta_mrad, enh_tot_est, label="total (estimator)")
ax.plot(theta_mrad, enh_co_est, label="co (helicidad conservada) (estimator)")
ax.plot(theta_mrad, enh_cross_est, label="cross (helicidad invertida) (estimator)")
ax.plot(theta_mrad, enh_tot_full, label="total (full)", ls="--")
ax.plot(theta_mrad, enh_co_full, label="co (helicidad conservada) (full)", ls="--")
ax.plot(theta_mrad, enh_cross_full, label="cross (helicidad invertida) (full)", ls="--")
ax.axhline(1.0, color="gray", ls=":", alpha=0.7)
if theta_coherent is not None:
    ax.axvline(theta_coherent * 1e3, color="red", ls="--", alpha=0.6,
               label=r"$1/(k\,\ell^*)$")
ax.set_xlabel("Angulo de dispersion (mrad)")
ax.set_ylabel("Factor de realce")
ax.set_title(f"Enhancement  $a={radius:.3f}\\,\\mu m$, $f={volume_fraction:.2f}$")
ax.legend()
fig.tight_layout()
fig.savefig(f"{save_path}/cbs-test-enhancement-comparison-{run_name_estimator}.pdf")

