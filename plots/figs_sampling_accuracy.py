"""
Estudio de precisión del sampleo de theta por CDF inversa (RGD-EMC y Mie).

El muestreo de theta (src/table.cpp) construye la CDF de la función de fase con una
suma de Riemann sobre `nDiv` intervalos en [0, pi] y luego invierte por interpolación
lineal. Cuando la función de fase oscila (parámetro de tamaño x grande, contraste m
grande, lambda pequeña), un `nDiv` bajo no resuelve las oscilaciones y la distribución
sampleada se aleja de la PDF real.

Este script cuantifica ese error de discretización frente a una referencia de alta
resolución y deriva una guía práctica para elegir `nDiv`. Es 100% Python: solo llama a
`.pdf()` y `.sample_theta()` ya expuestos por el módulo.

Ejecutar desde el directorio `plots/`:
    python figs_sampling_accuracy.py
"""

from utils.styles import apply, COLORS
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

from luminis_mc import RayleighDebyeEMCPhaseFunction, MiePhaseFunction

apply(context="paper", col="single")

save_path = "/Users/niaggar/Documents/Thesis/tests"

# ---------------------------------------------------------------------------
# Parámetros base (coherentes con figs_RGD_valid.py). Unidades en micrómetros.
# ---------------------------------------------------------------------------
N_MEDIUM = 1.33
N_PARTICLE = 1.59
WAVELENGTH = 0.514
K_MEDIUM = 2 * np.pi * N_MEDIUM / WAVELENGTH

THETA_MIN = 0.0
THETA_MAX = np.pi

# Malla de referencia (independiente de nDiv): mucho más fina que el nDiv máximo.
N_FINE = 200_000

# Barrido de divisiones (log-espaciado).
NDIV_GRID = np.array([10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000])

# Radios: de partícula pequeña (suave) a grande (oscilante).
RADII = np.array([0.035, 0.055, 0.085, 0.110, 0.150, 0.200, 0.300, 0.450])

# Tolerancias para la guía práctica.
TOL_KS = 1e-3       # distancia KS objetivo
TOL_G_REL = 1e-2    # error relativo en g objetivo

# Medición de tiempos: nº de muestras por punto y repeticiones de la creación.
N_SAMPLE_TIMING = 500_000
N_CREATE_REPEATS = 30
TIMING_RADIUS = 0.110


# ---------------------------------------------------------------------------
# Constructores de funciones de fase
# ---------------------------------------------------------------------------
MODELS = ("RGD-EMC", "Mie")


def make_phase(model, radius, nDiv, n_particle=N_PARTICLE,
               n_medium=N_MEDIUM, wavelength=WAVELENGTH):
    if model == "RGD-EMC":
        return RayleighDebyeEMCPhaseFunction(
            wavelength, radius, n_particle, n_medium, nDiv, THETA_MIN, THETA_MAX)
    if model == "Mie":
        return MiePhaseFunction(
            wavelength, radius, n_particle, n_medium, nDiv, THETA_MIN, THETA_MAX)
    raise ValueError(f"modelo desconocido: {model}")


def size_parameter(radius, n_medium=N_MEDIUM, wavelength=WAVELENGTH):
    return 2 * np.pi * n_medium * radius / wavelength


# ---------------------------------------------------------------------------
# Referencia "verdadera" y métricas
# ---------------------------------------------------------------------------
def pdf_on_grid(phase, theta):
    """Evalúa .pdf() sobre un array de theta (la PDF ya incluye el sin(theta))."""
    return np.array([phase.pdf(float(t)) for t in theta])


def reference_cdf(phase, n_fine=N_FINE):
    """Malla fina -> densidad normalizada y CDF de referencia (trapecio acumulado)."""
    theta = np.linspace(THETA_MIN, THETA_MAX, n_fine)
    p = pdf_on_grid(phase, theta)
    norm = np.trapezoid(p, theta)
    p_norm = p / norm
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p_norm[1:] + p_norm[:-1]) * np.diff(theta))])
    cdf /= cdf[-1]
    return theta, p_norm, cdf


def g_from_density(theta, p_norm):
    """Factor de anisotropía g = <cos theta> a partir de una densidad normalizada."""
    return np.trapezoid(np.cos(theta) * p_norm, theta)


def riemann_table(phase, nDiv):
    """
    Espejo de SamplingTable::initialize (src/table.cpp): regla del rectángulo sobre
    nDiv intervalos y CDF acumulada. Devuelve (values, cdf_norm, pdf_norm, step).
    Reproduce el comportamiento *intencionado* (incluye bien el último nodo, a
    diferencia del acceso fuera de rango señalado en table.cpp:23).
    """
    step = (THETA_MAX - THETA_MIN) / nDiv
    values = THETA_MIN + np.arange(nDiv + 1) * step
    pdf = pdf_on_grid(phase, values)
    pdf_norm = pdf / (pdf.sum() * step)
    cdf = np.cumsum(pdf_norm * step)
    cdf /= cdf[-1]
    return values, cdf, pdf_norm, step


def table_metrics(phase, nDiv, theta_fine, cdf_ref, g_ref):
    """Métricas de la vía A (determinista) frente a la referencia."""
    values, cdf, pdf_norm, step = riemann_table(phase, nDiv)
    # CDF del método interpolada en la malla fina (es lineal entre nodos).
    cdf_table = np.interp(theta_fine, values, cdf)
    ks = np.max(np.abs(cdf_table - cdf_ref))
    l1 = np.trapezoid(np.abs(cdf_table - cdf_ref), theta_fine)
    g_table = np.sum(np.cos(values) * pdf_norm * step)
    g_err_rel = abs(g_table - g_ref) / abs(g_ref)
    return {"ks": ks, "l1": l1, "g_table": g_table, "g_err_rel": g_err_rel}


def required_ndiv(ndiv_grid, errors, tol):
    """Menor nDiv (interpolado en log-log) que lleva el error por debajo de tol."""
    errors = np.asarray(errors, dtype=float)
    below = errors <= tol
    if below.all():
        return ndiv_grid[0]
    if not below.any():
        return np.nan
    j = np.argmax(below)              # primer índice que cumple
    if j == 0:
        return ndiv_grid[0]
    x0, x1 = np.log(ndiv_grid[j - 1]), np.log(ndiv_grid[j])
    y0, y1 = np.log(errors[j - 1]), np.log(errors[j])
    xt = x0 + (np.log(tol) - y0) * (x1 - x0) / (y1 - y0)
    return np.exp(xt)


def count_oscillations(theta_fine, p_norm):
    """Nº de máximos locales de la densidad: proxy de la estructura/oscilación."""
    maxima = argrelextrema(p_norm, np.greater)[0]
    return len(maxima)


# ---------------------------------------------------------------------------
# Precómputo: referencias por (modelo, radio) y métricas en todo el barrido
# ---------------------------------------------------------------------------
def build_results():
    res = {}
    for model in MODELS:
        res[model] = {"x": [], "n_osc": [], "g_ref": [],
                      "ks": [], "g_err_rel": [], "l1": []}
        for r in RADII:
            ref_phase = make_phase(model, r, nDiv=2_000)  # nDiv solo para construir; usamos .pdf()
            theta_f, p_norm, cdf_ref = reference_cdf(ref_phase)
            g_ref = g_from_density(theta_f, p_norm)
            res[model]["x"].append(size_parameter(r))
            res[model]["n_osc"].append(count_oscillations(theta_f, p_norm))
            res[model]["g_ref"].append(g_ref)

            ks_row, gerr_row, l1_row = [], [], []
            for nd in NDIV_GRID:
                m = table_metrics(make_phase(model, r, int(nd)), int(nd),
                                  theta_f, cdf_ref, g_ref)
                ks_row.append(m["ks"])
                gerr_row.append(m["g_err_rel"])
                l1_row.append(m["l1"])
            res[model]["ks"].append(ks_row)
            res[model]["g_err_rel"].append(gerr_row)
            res[model]["l1"].append(l1_row)
        for key in ("x", "n_osc", "g_ref"):
            res[model][key] = np.array(res[model][key])
        for key in ("ks", "g_err_rel", "l1"):
            res[model][key] = np.array(res[model][key])  # shape (n_radii, n_ndiv)
    return res


# ---------------------------------------------------------------------------
# Figura 1 — Galería de formas de la función de fase
# ---------------------------------------------------------------------------
def fig_shapes(save=True):
    radii_show = [0.035, 0.110, 0.200, 0.450]
    theta = np.linspace(THETA_MIN, THETA_MAX, 4_000)
    deg = np.degrees(theta)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, model in zip(axes, MODELS):
        for c, r in zip(COLORS, radii_show):
            phase = make_phase(model, r, nDiv=2_000)
            p = pdf_on_grid(phase, theta)
            p = p / np.trapezoid(p, theta)
            ax.semilogy(deg, p + 1e-12,
                        label=fr"$r={r:.3f}\,\mu m,\ x={size_parameter(r):.1f}$",
                        color=c)
        ax.set_title(model)
        ax.set_xlabel(r"$\theta$ (deg)")
        ax.set_xlim(0, 180)
    axes[0].set_ylabel(r"$p(\theta)$ (normalizada)")
    axes[1].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    if save:
        fig.savefig(f"{save_path}/sampling_fig1_shapes.pdf")
    return fig


# ---------------------------------------------------------------------------
# Figura 2 — Convergencia del error vs nDiv (una curva por x)
# ---------------------------------------------------------------------------
def fig_convergence(res, save=True):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=True)
    for col, model in enumerate(MODELS):
        x = res[model]["x"]
        for i in range(len(RADII)):
            c = plt.cm.viridis(i / (len(RADII) - 1))
            lab = fr"$x={x[i]:.1f}$"
            axes[0, col].loglog(NDIV_GRID, res[model]["ks"][i], "o-", color=c, label=lab, ms=3)
            axes[1, col].loglog(NDIV_GRID, res[model]["g_err_rel"][i], "o-", color=c, ms=3)
        axes[0, col].set_title(model)
        axes[0, col].axhline(TOL_KS, ls="--", color="0.5", lw=0.8)
        axes[1, col].axhline(TOL_G_REL, ls="--", color="0.5", lw=0.8)
        axes[1, col].set_xlabel(r"$n_{\mathrm{Div}}$")
    axes[0, 0].set_ylabel(r"$D_{\mathrm{KS}}$")
    axes[1, 0].set_ylabel(r"error relativo en $g$")
    axes[0, 1].legend(frameon=False, fontsize=6, ncol=2)
    fig.tight_layout()
    if save:
        fig.savefig(f"{save_path}/sampling_fig2_convergence.pdf")
    return fig


# ---------------------------------------------------------------------------
# Figura 3 — Fallo visual: PDF real vs histograma sampleado (vía B, C++ real)
# ---------------------------------------------------------------------------
def fig_visual_failure(model="Mie", radius=0.300, n_samples=2_000_000,
                       ndiv_show=(30, 300, 100_000), save=True):
    rng = np.random.default_rng(0)
    u = rng.random(n_samples)
    theta = np.linspace(THETA_MIN, THETA_MAX, 4_000)
    ref_phase = make_phase(model, radius, nDiv=2_000)
    p_ref = pdf_on_grid(ref_phase, theta)
    p_ref = p_ref / np.trapezoid(p_ref, theta)

    fig, axes = plt.subplots(1, len(ndiv_show), figsize=(7.0, 2.4), sharey=True)
    bins = np.linspace(THETA_MIN, THETA_MAX, 200)
    for ax, nd in zip(axes, ndiv_show):
        phase = make_phase(model, radius, int(nd))
        samples = np.array([phase.sample_theta(float(x)) for x in u])
        ax.hist(samples, bins=bins, density=True, color=COLORS[4],
                alpha=0.6, label="sampleado")
        ax.plot(theta, p_ref, color="k", lw=1.0, label="PDF real")
        ax.set_title(fr"$n_{{\mathrm{{Div}}}}={nd}$")
        ax.set_xlabel(r"$\theta$ (rad)")
        ax.set_xlim(0, np.pi)
    axes[0].set_ylabel(r"densidad")
    axes[-1].legend(frameon=False, fontsize=6)
    fig.suptitle(fr"{model}, $r={radius}\,\mu m$ ($x={size_parameter(radius):.1f}$)",
                 fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(f"{save_path}/sampling_fig3_visual_failure.pdf")
    return fig


# ---------------------------------------------------------------------------
# Figura 4 — Mapa de calor del error en el plano (nDiv, x)
# ---------------------------------------------------------------------------
def fig_heatmap(res, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, model in zip(axes, MODELS):
        Z = np.log10(res[model]["ks"] + 1e-12)  # shape (n_radii, n_ndiv)
        im = ax.pcolormesh(NDIV_GRID, res[model]["x"], Z,
                           shading="nearest", cmap="viridis")
        ax.set_xscale("log")
        ax.set_xlabel(r"$n_{\mathrm{Div}}$")
        ax.set_title(model)
        fig.colorbar(im, ax=ax, label=r"$\log_{10} D_{\mathrm{KS}}$")
    axes[0].set_ylabel(r"parámetro de tamaño $x$")
    fig.tight_layout()
    if save:
        fig.savefig(f"{save_path}/sampling_fig4_heatmap.pdf")
    return fig


# ---------------------------------------------------------------------------
# Figura 5 — Guía práctica: nDiv mínimo requerido vs x (y vs nº de oscilaciones)
# ---------------------------------------------------------------------------
def fig_guideline(res, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for model, marker in zip(MODELS, ("o", "s")):
        x = res[model]["x"]
        n_osc = res[model]["n_osc"]
        nd_ks = np.array([required_ndiv(NDIV_GRID, res[model]["ks"][i], TOL_KS)
                          for i in range(len(RADII))])
        nd_g = np.array([required_ndiv(NDIV_GRID, res[model]["g_err_rel"][i], TOL_G_REL)
                         for i in range(len(RADII))])
        axes[0].loglog(x, nd_ks, marker + "-", label=fr"{model} ($D_{{KS}}<10^{{-3}}$)")
        axes[0].loglog(x, nd_g, marker + "--", alpha=0.6,
                       label=fr"{model} ($\Delta g/g<10^{{-2}}$)")
        axes[1].plot(n_osc, nd_ks, marker, label=model)
    axes[0].set_xlabel(r"parámetro de tamaño $x$")
    axes[0].set_ylabel(r"$n_{\mathrm{Div}}$ mínimo")
    axes[0].legend(frameon=False, fontsize=6)
    axes[1].set_xlabel(r"nº de máximos locales de $p(\theta)$")
    axes[1].set_ylabel(r"$n_{\mathrm{Div}}$ mínimo ($D_{KS}$)")
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    if save:
        fig.savefig(f"{save_path}/sampling_fig5_guideline.pdf")
    return fig


# ---------------------------------------------------------------------------
# Figura 6 — Tiempo de ejecución vs nDiv (creación, sampleo y total)
# ---------------------------------------------------------------------------
def measure_timing(model, ndiv_grid=None, radius=TIMING_RADIUS,
                   n_samples=N_SAMPLE_TIMING, n_create=N_CREATE_REPEATS, seed=0):
    """
    Mide, por nDiv, el tiempo de:
      - creación de la tabla (construir la función de fase, que ejecuta
        SamplingTable::initialize), promediado sobre n_create repeticiones;
      - sampleo de n_samples ángulos vía .sample_theta();
      - total = creación + sampleo.
    El mismo array de u se reutiliza en todos los nDiv para una comparación justa.
    Devuelve (t_create, t_sample, t_total) en segundos, shape (len(ndiv_grid),).
    """
    import time
    if ndiv_grid is None:
        ndiv_grid = NDIV_GRID
    rng = np.random.default_rng(seed)
    u = rng.random(n_samples)
    t_create, t_sample = [], []
    for nd in ndiv_grid:
        nd = int(nd)
        t0 = time.perf_counter()
        for _ in range(n_create):
            phase = make_phase(model, radius, nd)
        t_create.append((time.perf_counter() - t0) / n_create)

        phase = make_phase(model, radius, nd)
        t0 = time.perf_counter()
        for x in u:
            phase.sample_theta(float(x))
        t_sample.append(time.perf_counter() - t0)
    t_create = np.array(t_create)
    t_sample = np.array(t_sample)
    return t_create, t_sample, t_create + t_sample


def fig_timing(save=True):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, model in zip(axes, MODELS):
        t_create, t_sample, t_total = measure_timing(model)
        ax.loglog(NDIV_GRID, t_total * 1e3, "o-", color=COLORS[0],
                  label="total (creación + sampleo)")
        ax.loglog(NDIV_GRID, t_sample * 1e3, "s-", color=COLORS[1],
                  label=f"sampleo ({N_SAMPLE_TIMING:.0e} muestras)")
        ax.loglog(NDIV_GRID, t_create * 1e3, "^-", color=COLORS[2],
                  label="creación de la tabla")
        ax.set_title(model)
        ax.set_xlabel(r"$n_{\mathrm{Div}}$")
    axes[0].set_ylabel("tiempo (ms)")
    axes[1].legend(frameon=False, fontsize=6)
    fig.suptitle(fr"$r={TIMING_RADIUS}\,\mu m$", fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(f"{save_path}/sampling_fig6_timing.pdf")
    return fig


# ---------------------------------------------------------------------------
def main():
    print("Construyendo métricas (referencia de alta resolución + barrido nDiv)...")
    res = build_results()

    # Validación cruzada: g_ref vs estimación Monte Carlo de C++.
    print("\nValidación cruzada g_ref vs get_anisotropy_factor() (MC, C++):")
    for model in MODELS:
        for i, r in enumerate(RADII):
            g_mc = make_phase(model, r, nDiv=100_000).get_anisotropy_factor()[0]
            print(f"  {model:8s} r={r:.3f} x={res[model]['x'][i]:5.1f}  "
                  f"g_ref={res[model]['g_ref'][i]:.4f}  g_MC={g_mc:.4f}  "
                  f"n_osc={res[model]['n_osc'][i]}")

    fig_shapes()
    fig_convergence(res)
    fig_visual_failure()
    fig_heatmap(res)
    fig_guideline(res)
    fig_timing()
    print(f"\nFiguras guardadas en {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
