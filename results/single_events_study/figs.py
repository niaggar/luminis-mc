from ..utils.loaders import load_sweep
from ..utils.styles import apply
from ..utils.analysis import cbs_profiles, circular

import numpy as np
import matplotlib.pyplot as plt



apply(context="paper", col="single")

save_path = "/home/niaggar/Developer/luminis-mc/temporal_results"
folder = "single_events_study"

eps = 1e-30

sweep_data = load_sweep(folder)
run_names = sorted(sweep_data.keys())


run_name = "0000_radius_0.110_volumefraction_1.000_estimator"
loader = sweep_data[run_name]

p = loader.params                       # SimParams tipado
radius = p.layers[0].medium.radius
volume_fraction = p.extra["volume_fraction"]
theta_coherent = p.extra.get("theta_coherent")
n_photons_estimator = p.run.n_photons

events = [2, 3, 4, 5, 10, 15, 20, 30, 50, 100, 150, 200, 300, 500, 1000]

# Perfil CBS para un orden de scattering concreto (sensor farfield_cbs_{event})
prof = cbs_profiles(loader.processed_cbs("farfield_cbs_2"), basis=circular, time_index=0)
theta_mrad = prof.theta * 1e3


# ===================================================================
# Figura 1: intensidades coherente vs incoherente (total)
# ===================================================================
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(theta_mrad, prof.coherent["total"], label="coherente")
ax.plot(theta_mrad, prof.incoherent["total"], label="incoherente")
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
ax.plot(theta_mrad, prof.enhancement["co"], label="helicidad preservada")
ax.plot(theta_mrad, prof.enhancement["cross"], label="helicidad cruzada")
ax.plot(theta_mrad, prof.enhancement["total"], label="helicidad total")
if theta_coherent is not None:
    ax.axvline(theta_coherent * 1e3, color="red", ls="--", alpha=0.6,
               label=r"$1/(k\,\ell^*)$")
ax.set_xlabel("Angulo de dispersion (mrad)")
ax.set_ylabel("Enhancement (u.a.)")
ax.set_title(f"Enhancement CBS forward+reverse  $a={radius:.3f}\\,\\mu m$, $f={volume_fraction:.2f}$")
ax.legend()
fig.tight_layout()
fig.savefig(f"{save_path}/cbs_enhancement_events.pdf")
