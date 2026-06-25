from ..utils.loaders import load_sweep
from ..utils.styles import apply
from ..utils.analysis import cbs_profiles, circular

import numpy as np
import matplotlib.pyplot as plt



apply(context="paper", col="single")

save_path = "/home/niaggar/Developer/luminis-mc/temporal_results"
folder = "max_events_study"

eps = 1e-30

sweep_data = load_sweep(folder)
run_names = sorted(sweep_data.keys())

max_events = [5, 10, 20, 50, 100, 200, 1000, 2000, 5000, 10000]


p = sweep_data["0000_maxevents_5"].params           # SimParams tipado
radius = p.layers[0].medium.radius
volume_fraction = p.extra["volume_fraction"]
theta_coherent = p.extra.get("theta_coherent")
n_photons_estimator = p.run.n_photons


# Perfiles CBS por cada max_events: base circular, promedio azimutal, ventana unica (0)
data_profiles = []
for max_ev in max_events:
    loader = sweep_data[f"0000_maxevents_{max_ev}"]
    prof = cbs_profiles(loader.processed_cbs("farfield_cbs"), basis=circular, time_index=0)
    data_profiles.append((max_ev, prof))

theta_mrad = data_profiles[0][1].theta * 1e3


# ===================================================================
# Figura 1: intensidades coherente vs incoherente (total)
# ===================================================================
fig, ax = plt.subplots(figsize=(6, 4))

for max_ev, prof in data_profiles:
    ax.plot(theta_mrad, prof.coherent["total"], label=f"max_events={max_ev} (coh)")

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

for max_ev, prof in data_profiles:
    ax.plot(theta_mrad, prof.enhancement["co"], label=f"max_events={max_ev} (coh)")

if theta_coherent is not None:
    ax.axvline(theta_coherent * 1e3, color="red", ls="--", alpha=0.6,
               label=r"$1/(k\,\ell^*)$")
ax.set_xlabel("Angulo de dispersion (mrad)")
ax.set_ylabel("Enhancement (u.a.)")
ax.set_title(f"Enhancement CBS forward+reverse  $a={radius:.3f}\\,\\mu m$, $f={volume_fraction:.2f}$")
ax.legend()
fig.tight_layout()
fig.savefig(f"{save_path}/cbs_enhancement_max_events.pdf")
