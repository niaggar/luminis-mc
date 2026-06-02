"""
Grafica la prueba de deteccion CBS (forward + reverse).

Lee la corrida generada por `results/sim_cbs_test.py` y dibuja, para el cono de
retrodispersion:

    - Intensidad coherente vs incoherente (canales co/cross y total)
    - Factor de realce (enhancement) = coherente / incoherente

Convencion de polarizacion circular (helicidad):
    I_co    (helicidad conservada) = (S0 - S3) / 2
    I_cross (helicidad invertida)  = (S0 + S3) / 2

Los mapas guardados son 2D (theta, phi); aqui se promedian azimutalmente
(sobre phi) para obtener el perfil 1D del cono CBS.
"""

from utils.loaders import load_sweep
from utils.styles import apply

import numpy as np
import matplotlib.pyplot as plt



apply(context="paper", col="single")

save_path = "/Users/niaggar/Documents/Thesis/tests/"
folder = "cbs_test/"  # carpeta dentro de `base_dir` donde se guardaron los resultados de la simulacion

eps = 1e-30

sweep_data = load_sweep(folder)
run_names = sorted(sweep_data.keys())


def azimuthal_average(mat):
    """Promedio sobre phi -> perfil 1D en theta. Acepta 1D o 2D."""
    arr = np.asarray(mat)
    if arr.ndim == 1:
        return arr
    return arr.mean(axis=1)


for run_name in run_names:
    print(f"---------------- {run_name}")
    loader = sweep_data[run_name]
    p = loader.params

    radius = p["radius_um"]
    volume_fraction = p["volume_fraction"]
    theta_coherent = p.get("theta_coherent_rad", None)
    n_photons = p["n_photons"]

    theta = loader.derived("axes/theta_rad")          # rad
    theta_mrad = theta * 1e3

    # --- mapas 2D -> perfil radial (promedio en phi) ---
    coh_s0 = azimuthal_average(loader.derived("farfield_cbs/coherent/s0"))
    coh_s3 = azimuthal_average(loader.derived("farfield_cbs/coherent/s3"))
    inc_s0 = azimuthal_average(loader.derived("farfield_cbs/incoherent/s0"))
    inc_s3 = azimuthal_average(loader.derived("farfield_cbs/incoherent/s3"))

    # --- canales de helicidad ---
    coh_co = (coh_s0 - coh_s3) / 2.0
    coh_cross = (coh_s0 + coh_s3) / 2.0
    coh_tot = coh_co + coh_cross

    inc_co = (inc_s0 - inc_s3) / 2.0
    inc_cross = (inc_s0 + inc_s3) / 2.0
    inc_tot = inc_co + inc_cross

    enh_co = (coh_co + eps) / (inc_co + eps)
    enh_cross = (coh_cross + eps) / (inc_cross + eps)
    enh_tot = (coh_tot + eps) / (inc_tot + eps)

    print(f"  enhancement total en theta=0: {enh_tot[0]:.3f}")
    print(f"  enhancement co    en theta=0: {enh_co[0]:.3f}")
    print(f"  enhancement cross en theta=0: {enh_cross[0]:.3f}")

    label = f"$a={radius:.3f}\\,\\mu m$, $f={volume_fraction:.2f}$"

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
    ax.set_title(f"CBS forward+reverse  {label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{save_path}/cbs-test-intensity-{run_name}.pdf")

    # ===================================================================
    # Figura 2: factor de realce por canal
    # ===================================================================
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(theta_mrad, enh_tot, label="total")
    ax.plot(theta_mrad, enh_co, label="co (helicidad conservada)")
    ax.plot(theta_mrad, enh_cross, label="cross (helicidad invertida)")
    ax.axhline(1.0, color="gray", ls=":", alpha=0.7)
    if theta_coherent is not None:
        ax.axvline(theta_coherent * 1e3, color="red", ls="--", alpha=0.6,
                   label=r"$1/(k\,\ell^*)$")
    ax.set_xlabel("Angulo de dispersion (mrad)")
    ax.set_ylabel("Factor de realce")
    ax.set_title(f"Enhancement  {label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{save_path}/cbs-test-enhancement-{run_name}.pdf")


plt.show()
