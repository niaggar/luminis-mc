"""
Figuras de validacion: HOMOGENEA vs MEZCLA (dos especies identicas).

Carga los dos datasets generados por sim_homogeneous.py y sim_mixture.py y
comprueba la equivalencia:

    HomogeneousLayer(n)   ==   MixtureLayer([n/2, n/2])

Genera:
  A) Overlay co-polarizado (homogenea vs mezcla) sobre el eje theta COMUN.
  B) Residual (mezcla - homogenea) del canal co-pol, con banda de tolerancia MC.
  C) Los tres canales (co / cross / total) superpuestos para ambos datasets.

Resumen numerico por consola: E(0) de cada uno, |dE(0)|, RMS del residual y un
veredicto PASS/FAIL. La igualdad es ESTADISTICA (la mezcla consume un draw extra
por evento en la seleccion de especie), asi que se compara dentro del ruido MC.

NOTA (igual que figs.py): theta_coherent se recalcula desde l* con
k = 2*pi*n_medium/lambda, NO desde p.extra["theta_coherent"] (arrastra un factor
180/pi por el bug de derived_quantities). Los datos SI son comparables porque
ambas corridas usan la MISMA grilla (mismo theta_max/d_theta/d_phi).

SUPUESTO DE API: se reutiliza la base 'linear' de results.utils.analysis, igual
que figs.py (co = paralela ∥, cross = perpendicular ⊥).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from utils.loaders import load_sweep
from utils.styles import apply
from utils.analysis import cbs_profiles, linear


apply(context="paper", col="single")

# ===========================================================================
# Config
# ===========================================================================
BASE_DIR = "/Users/niaggar/Documents/Thesis/tests"          # <-- mismo BASE_DIR de las sims
SAVE_PATH = "/Users/niaggar/Documents/Thesis/tests"  # <-- ajustar

EXP_HOMO = "cbs_validation_homogeneous"
EXP_MIX = "cbs_validation_mixture"

# Constantes del estudio (host acuoso, linea de Iwai)
N_MEDIUM = 1.33
WAVELENGTH = 0.514
K_MEDIUM = 2.0 * np.pi * N_MEDIUM / WAVELENGTH   # ~16.26 um^-1

# Tolerancias del gate de equivalencia (del orden del ruido MC a N_PHOTONS)
TOL_E0 = 0.05             # |E0_mix - E0_homo|
TOL_CURVE_RMS = 0.05      # RMS del residual co-pol


# ===========================================================================
# Carga robusta (load_sweep por nombre; fallback a ruta absoluta)
# ===========================================================================
def load_one(exp_name):
    """Carga un sweep de UNA corrida y devuelve (loader, params, profile)."""
    try:
        sweep_data = load_sweep(exp_name)
    except Exception:
        sweep_data = load_sweep(os.path.join(BASE_DIR, exp_name))

    names = sorted(sweep_data.keys())
    if not names:
        raise RuntimeError(f"El sweep '{exp_name}' no tiene corridas.")
    loader = sweep_data[names[0]]
    p = loader.params
    prof = cbs_profiles(loader.processed_cbs("farfield_cbs"),
                        basis=linear, time_index=0)
    return loader, p, prof


loader_h, p_h, prof_h = load_one(EXP_HOMO)
loader_m, p_m, prof_m = load_one(EXP_MIX)

# Metadatos (leidos de extra; claves forzadas por ambas sims -> siempre existen)
lstar_h = p_h.extra["transport_mean_free_path"]
lstar_m = p_m.extra["transport_mean_free_path"]
radius = p_h.extra.get("radius", float("nan"))
theta_coh = 1.0 / (K_MEDIUM * lstar_h)   # rad, calculo correcto (no el de extra)

# Ejes y canales
th_h = np.asarray(prof_h.theta)
th_m = np.asarray(prof_m.theta)
co_h = np.asarray(prof_h.enhancement["co"])
co_m = np.asarray(prof_m.enhancement["co"])

# La grilla debe ser comun: mismo tamano y mismo eje. Se recorta por seguridad.
n = min(co_h.size, co_m.size)
if not np.allclose(th_h[:n], th_m[:n], rtol=0, atol=1e-9):
    print("[warn] los ejes theta NO son identicos: revisa que las grillas de "
          "sim_homogeneous.py y sim_mixture.py coincidan (N_THETA/N_PHI/factor).")

theta_mrad = th_h[:n] * 1e3
residual = co_m[:n] - co_h[:n]

# ===========================================================================
# Resumen numerico -> PASS/FAIL
# ===========================================================================
E0_h = float(co_h[0])
E0_m = float(co_m[0])
dE0 = abs(E0_m - E0_h)
rms = float(np.sqrt(np.nanmean(residual ** 2)))
max_abs = float(np.nanmax(np.abs(residual)))
nan_h = bool(np.isnan(co_h).any())
nan_m = bool(np.isnan(co_m).any())

print("\n===== EQUIVALENCIA HOMOGENEA vs MEZCLA (co-pol) =====")
print(f"  radio               = {radius:.3f} um")
print(f"  l*  homo={lstar_h:.4f}  mix={lstar_m:.4f}  (dif rel "
      f"{abs(lstar_m - lstar_h) / lstar_h:.2e})")
print(f"  E(0) homo           = {E0_h:.4f}")
print(f"  E(0) mix            = {E0_m:.4f}")
print(f"  |dE(0)|             = {dE0:.4f}   (tol {TOL_E0})")
print(f"  RMS residual        = {rms:.4f}   (tol {TOL_CURVE_RMS})")
print(f"  max|residual|       = {max_abs:.4f}")
print(f"  NaN grids  homo={nan_h}  mix={nan_m}")

ok = (not nan_h) and (not nan_m) and dE0 < TOL_E0 and rms < TOL_CURVE_RMS
print("\n  ==>", "PASS \u2705" if ok else "FAIL \u274c")

# ===========================================================================
# Figura A: overlay co-polarizado
# ===========================================================================
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(theta_mrad, co_h[:n], label="homogenea (n)", lw=1.8)
ax.plot(theta_mrad, co_m[:n], label="mezcla (n/2 + n/2)",
        ls="--", lw=1.4)
ax.axhline(2.0, color="gray", ls=":", alpha=0.5)
ax.axhline(1.0, color="gray", ls=":", alpha=0.7)
ax.axvline(theta_coh * 1e3, color="red", ls="--", alpha=0.5,
           label=r"$1/(k\,\ell^{*})$")
ax.set_xlim(0, theta_mrad.max())
ax.set_ylim(0.9, 2.05)
ax.set_xlabel("Angulo reducido (mrad)")
ax.set_ylabel("Intensidad normalizada (co-pol)")
ax.set_title(fr"Equivalencia CBS -- $a={radius:.3f}\,\mu m$  "
             f"({'PASS' if ok else 'FAIL'})")
ax.legend()
fig.tight_layout()
fig.savefig(f"{SAVE_PATH}/validation-copol-overlay.pdf")

# ===========================================================================
# Figura B: residual con banda de tolerancia
# ===========================================================================
fig, ax = plt.subplots(figsize=(6, 4))
ax.axhspan(-TOL_CURVE_RMS, TOL_CURVE_RMS, color="green", alpha=0.10,
           label=f"banda +/-{TOL_CURVE_RMS}")
ax.axhline(0.0, color="gray", ls=":", alpha=0.7)
ax.plot(theta_mrad, residual, lw=1.2, color="C3")
ax.set_xlim(0, theta_mrad.max())
ax.set_xlabel("Angulo reducido (mrad)")
ax.set_ylabel("Residual co-pol (mezcla - homogenea)")
ax.set_title(f"Residual de equivalencia  (RMS={rms:.4f})")
ax.legend()
fig.tight_layout()
fig.savefig(f"{SAVE_PATH}/validation-copol-residual.pdf")

# ===========================================================================
# Figura C: los tres canales para ambos datasets
# ===========================================================================
fig, ax = plt.subplots(figsize=(6, 4))
channels = [("total", "C0"), ("co", "C1"), ("cross", "C2")]
labels = {"total": "total", "co": r"co-pol ($\parallel$)",
          "cross": r"cross-pol ($\perp$)"}
for ch, color in channels:
    yh = np.asarray(prof_h.enhancement[ch])[:n]
    ym = np.asarray(prof_m.enhancement[ch])[:n]
    ax.plot(theta_mrad, yh, color=color, lw=1.8, label=f"{labels[ch]} homo")
    ax.plot(theta_mrad, ym, color=color, ls="--", lw=1.2,
            label=f"{labels[ch]} mix")
ax.axhline(1.0, color="gray", ls=":", alpha=0.7)
ax.set_xlim(0, theta_mrad.max())
ax.set_xlabel("Angulo reducido (mrad)")
ax.set_ylabel("Factor de realce")
ax.set_title(fr"Canales CBS homo vs mezcla -- $a={radius:.3f}\,\mu m$")
ax.legend(ncol=2, fontsize=8)
fig.tight_layout()
fig.savefig(f"{SAVE_PATH}/validation-channels.pdf")

print(f"\nFiguras guardadas en: {SAVE_PATH}")