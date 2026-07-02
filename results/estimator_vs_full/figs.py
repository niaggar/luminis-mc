from utils.loaders import load_sweep
from utils.styles import apply
from utils.analysis import cbs_profiles, circular

import numpy as np
import matplotlib.pyplot as plt



apply(context="paper", col="single")

save_path = "/home/niaggar/Developer/luminis-mc/temporal_results"
folder = "estimator_vs_full"

eps = 1e-30

sweep_data = load_sweep(folder)
run_names = sorted(sweep_data.keys())


run_name_estimator = "0000_radius_0.110_volumefraction_1.000_estimator"
run_name_full = "0000_radius_0.110_volumefraction_1.000_full"


loader_est = sweep_data[run_name_estimator]
loader_full = sweep_data[run_name_full]

p = loader_est.params                       # SimParams tipado
radius = p.layers[0].medium.radius
volume_fraction = p.extra["volume_fraction"]
theta_coherent = p.extra.get("theta_coherent")
n_photons_estimator = p.run.n_photons

# Perfiles CBS: base circular (helicidad), promedio azimutal, ventana unica (0)
prof_est = cbs_profiles(loader_est.processed_cbs("farfield_cbs"), basis=circular, time_index=0)
prof_full = cbs_profiles(loader_full.processed_cbs("farfield_cbs"), basis=circular, time_index=0)

theta_mrad = prof_est.theta * 1e3


# ===================================================================
# Figura 1: intensidades coherente vs incoherente (total)
# ===================================================================
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(theta_mrad, prof_est.coherent["total"], label="coherente (estimator)")
ax.plot(theta_mrad, prof_est.incoherent["total"], label="incoherente (estimator)")
ax.plot(theta_mrad, prof_full.coherent["total"], label="coherente (full)", ls="--")
ax.plot(theta_mrad, prof_full.incoherent["total"], label="incoherente (full)", ls="--")
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
ax.plot(theta_mrad, prof_est.enhancement["total"], label="total (estimator)")
ax.plot(theta_mrad, prof_est.enhancement["co"], label="co (helicidad conservada) (estimator)")
ax.plot(theta_mrad, prof_est.enhancement["cross"], label="cross (helicidad invertida) (estimator)")
ax.plot(theta_mrad, prof_full.enhancement["total"], label="total (full)", ls="--")
ax.plot(theta_mrad, prof_full.enhancement["co"], label="co (helicidad conservada) (full)", ls="--")
ax.plot(theta_mrad, prof_full.enhancement["cross"], label="cross (helicidad invertida) (full)", ls="--")
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
