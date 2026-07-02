"""
cbs_fit.py — Ajuste de perfiles CBS (Coherent BackScattering)

Modelos implementados (Müller & Delande, 2016):
  - "diff"  : Ec.(134) — perfil difusivo simple
  - "bound" : Ec.(138) — difusivo con condición de borde (z₀ = 2/3)
  - "deco"  : Ec.(146) — difusivo con decoherencia fenomenológica

El camino libre medio se extrae directamente del ancho del perfil:
    FWHM_q ≈ 0.73   (q = kl·θ, unidades reducidas)
    l = kl / k = kl · λ / (2π)

Uso básico
----------
    from cbs_fit import fit_cbs, plot_cbs_fit

    result = fit_cbs(theta_mrad, enhancement, wavelength_nm=532)
    print(result)

    # integrado en un subplot existente:
    plot_cbs_fit(ax, theta_mrad, enhancement, wavelength_nm=532)
"""

import numpy as np
from scipy.optimize import minimize_scalar, minimize
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dataclasses import dataclass
from typing import Literal


# ─────────────────────────────────────────────
#  Modelos CBS (variable reducida q = kl·θ_rad)
# ─────────────────────────────────────────────

def _model_diff(q: np.ndarray) -> np.ndarray:
    """Ec.(134): I_C/I_L = 1 / (1 + |q|)²"""
    return 1.0 / (1.0 + np.abs(q)) ** 2


def _model_bound(q: np.ndarray) -> np.ndarray:
    """Ec.(138): difusivo con condición de borde (z₀ = 2/3)."""
    z0 = 2.0 / 3.0
    aq = np.abs(q)
    base = 1.0 / (1.0 + aq) ** 2
    # límite q→0: bc → (1 + 2z₀)
    bc = np.where(aq < 1e-10,
                  1.0 + 2.0 * z0,
                  1.0 + (1.0 - np.exp(-2.0 * z0 * aq)) / aq)
    return base / (1.0 + 2.0 * z0) * bc


def _model_deco(q: np.ndarray, eps: float) -> np.ndarray:
    """Ec.(146): difusivo con decoherencia — eps = l/Lφ."""
    return 1.0 / (1.0 + np.sqrt(q ** 2 + eps ** 2)) ** 2


_MODELS = {
    "diff":  ("Ec.(134) difusivo",           lambda q, _: _model_diff(q)),
    "bound": ("Ec.(138) cond. de borde",     lambda q, _: _model_bound(q)),
    "deco":  ("Ec.(146) con decoherencia",   _model_deco),
}

_EQ_LABELS = {
    "diff":  r"$\frac{I_C}{I_L} = \frac{1}{(1+kl|\theta|)^2}$",
    "bound": r"bound_eq",
    "deco":  r"$\frac{I_C}{I_L} = \frac{1}{\left[1+\sqrt{(kl\theta)^2+(l/L_\varphi)^2}\right]^2}$",
}


# ─────────────────────────────────────────────
#  Resultado del ajuste
# ─────────────────────────────────────────────

@dataclass
class CBSFitResult:
    model:        str          # "diff" | "bound" | "deco"
    kl:           float        # parámetro adimensional kl
    l_um:         float        # camino libre medio [µm]
    l_mm:         float        # camino libre medio [mm]
    A:            float        # amplitud CBS
    B:            float        # fondo incoherente
    enhancement:  float        # I_C(0)/I_L = A·f(0)/B  (pico/fondo)
    fwhm_mrad:    float        # FWHM del perfil en mrad
    fwhm_q:       float        # FWHM en unidades reducidas (≈ 0.73)
    chi2_per_n:   float        # χ²/N del ajuste
    eps:          float        # l/Lφ  (0 si modelo != "deco")
    Lphi_um:      float        # longitud de coherencia [µm]  (inf si eps=0)
    wavelength_nm: float
    n_points:     int

    def __str__(self):
        name, _ = _MODELS[self.model]
        lines = [
            f"── CBS Fit Result ─────────────────────────",
            f"  Modelo    : {name}",
            f"  λ         : {self.wavelength_nm:.1f} nm",
            f"  kl        : {self.kl:.2f}",
            f"  l         : {self.l_um:.3f} µm  ({self.l_mm:.4f} mm)",
            f"  A (CBS)   : {self.A:.4f}",
            f"  B (fondo) : {self.B:.4f}",
            f"  Pico/fondo: {self.enhancement:.4f}",
            f"  FWHM      : {self.fwhm_mrad:.4f} mrad",
            f"  Δq FWHM   : {self.fwhm_q:.3f} (unidades reducidas)",
            f"  χ²/N      : {self.chi2_per_n:.3e}  (N={self.n_points})",
        ]
        if self.model == "deco":
            lph = f"{self.Lphi_um:.2f} µm" if np.isfinite(self.Lphi_um) else "∞"
            lines += [
                f"  l/Lφ      : {self.eps:.4f}",
                f"  Lφ        : {lph}",
            ]
        lines.append("─" * 44)
        return "\n".join(lines)


# ─────────────────────────────────────────────
#  Motor de ajuste
# ─────────────────────────────────────────────

def _linfit_AB(f_vals: np.ndarray, y: np.ndarray):
    """Mínimos cuadrados lineales: y ≈ A·f + B.  Devuelve (A, B, residual²)."""
    n = len(y)
    sf2 = f_vals @ f_vals
    sf  = f_vals.sum()
    sfy = f_vals @ y
    sy  = y.sum()
    det = sf2 * n - sf * sf
    if abs(det) < 1e-30:
        return None
    A = (n * sfy - sf * sy) / det
    B = (sf2 * sy - sf * sfy) / det
    if A <= 0 or B < 0:
        return None
    res = np.sum((y - (A * f_vals + B)) ** 2)
    return A, B, res


def fit_cbs(
    theta_mrad:     np.ndarray,
    enhancement:    np.ndarray,
    wavelength_nm:  float = 532.0,
    model:          Literal["diff", "bound", "deco"] = "diff",
    center_mrad:    float | None = None,
    kl_bounds:      tuple[float, float] = (1.0, 5e5),
) -> CBSFitResult:
    """
    Ajusta un perfil CBS a los modelos de Müller & Delande (2016).

    Parámetros
    ----------
    theta_mrad    : array de ángulos [mrad], no necesita estar centrado.
    enhancement   : array de intensidad / fondo (I_C/I_L), misma longitud.
    wavelength_nm : longitud de onda del láser [nm].
    model         : "diff" | "bound" | "deco".
    center_mrad   : centro θ=0 en mrad; si None se usa el argmax.
    kl_bounds     : (kl_min, kl_max) para la búsqueda.

    Devuelve
    --------
    CBSFitResult con kl, l [µm/mm], A, B, enhancement, FWHM, χ², etc.
    """
    theta = np.asarray(theta_mrad, dtype=float)
    y     = np.asarray(enhancement, dtype=float)

    if len(theta) < 5:
        raise ValueError("Se necesitan al menos 5 puntos para el ajuste.")
    if len(theta) != len(y):
        raise ValueError("theta_mrad e enhancement deben tener la misma longitud.")

    # Centrar en θ=0
    if center_mrad is None:
        center_mrad = float(theta[np.argmax(y)])
    theta_c = (theta - center_mrad) * 1e-3   # → radianes, centrado

    lam_m = wavelength_nm * 1e-9
    k     = 2.0 * np.pi / lam_m

    model_fn = _MODELS[model][1]

    # ── Grilla gruesa logarítmica en kl ──────────────────────────────────────
    kl_min, kl_max = kl_bounds
    N_coarse = 500
    kl_grid  = np.exp(np.linspace(np.log(kl_min), np.log(kl_max), N_coarse))

    # Para "deco" también barremos eps = l/Lφ
    eps_grid = (np.array([0.0, 0.005, 0.01, 0.02, 0.05, 0.08, 0.12,
                           0.18, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5])
                if model == "deco" else np.array([0.0]))

    best_kl, best_eps = kl_grid[0], 0.0
    best_A,  best_B   = 1.0, 0.0
    best_res          = np.inf

    for eps in eps_grid:
        for kl in kl_grid:
            q    = kl * theta_c
            fval = model_fn(q, eps)
            ab   = _linfit_AB(fval, y)
            if ab and ab[2] < best_res:
                best_res = ab[2]
                best_kl, best_eps = kl, eps
                best_A, best_B = ab[0], ab[1]

    # ── Refinamiento fino con scipy ──────────────────────────────────────────
    def cost_kl(log_kl, eps):
        kl   = np.exp(log_kl)
        fval = model_fn(kl * theta_c, eps)
        ab   = _linfit_AB(fval, y)
        return ab[2] if ab else 1e30

    opt = minimize_scalar(
        cost_kl,
        bounds=(np.log(best_kl / 3.0), np.log(best_kl * 3.0)),
        method="bounded",
        args=(best_eps,),
        options={"xatol": 1e-6},
    )
    best_kl = np.exp(opt.x)

    if model == "deco":
        def cost_2d(params):
            log_kl, eps = params
            fval = model_fn(np.exp(log_kl) * theta_c, max(eps, 0.0))
            ab   = _linfit_AB(fval, y)
            return ab[2] if ab else 1e30

        res2 = minimize(
            cost_2d,
            x0=[np.log(best_kl), best_eps],
            method="Nelder-Mead",
            options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 4000},
        )
        best_kl  = np.exp(res2.x[0])
        best_eps = max(res2.x[1], 0.0)

    # ── Resultado final ───────────────────────────────────────────────────────
    q_final  = best_kl * theta_c
    f_final  = model_fn(q_final, best_eps)
    ab_final = _linfit_AB(f_final, y)
    if ab_final is None:
        raise RuntimeError("El ajuste no convergió. Comprueba los datos.")
    best_A, best_B, best_res = ab_final

    l_m       = best_kl / k
    l_um      = l_m * 1e6
    f0        = float(model_fn(np.array([0.0]), best_eps)[0])
    enh       = (best_A * f0 + best_B) / best_B if best_B > 0 else float("nan")
    fwhm_q    = 0.73                           # Müller & Delande (2016)
    fwhm_mrad = fwhm_q / best_kl * 1e3        # rad → mrad
    Lphi_um   = (l_um / best_eps) if best_eps > 1e-6 else float("inf")

    return CBSFitResult(
        model         = model,
        kl            = best_kl,
        l_um          = l_um,
        l_mm          = l_um / 1e3,
        A             = best_A,
        B             = best_B,
        enhancement   = enh,
        fwhm_mrad     = fwhm_mrad,
        fwhm_q        = fwhm_q,
        chi2_per_n    = best_res / len(y),
        eps           = best_eps,
        Lphi_um       = Lphi_um,
        wavelength_nm = wavelength_nm,
        n_points      = len(y),
    )


# ─────────────────────────────────────────────
#  Visualización
# ─────────────────────────────────────────────

def plot_cbs_fit(
    ax:             plt.Axes,
    theta_mrad:     np.ndarray,
    enhancement:    np.ndarray,
    result:         CBSFitResult | None = None,
    wavelength_nm:  float = 532.0,
    model:          Literal["diff", "bound", "deco"] = "diff",
    center_mrad:    float | None = None,
    data_kw:        dict | None = None,
    fit_kw:         dict | None = None,
    show_fwhm:      bool = True,
    show_eq:        bool = True,
    label_prefix:   str = "",
) -> CBSFitResult:
    """
    Dibuja los datos CBS + curva ajustada en un Axes existente.

    Si `result` es None, realiza el ajuste internamente.
    Devuelve el CBSFitResult para que puedas usar kl, l, etc.

    Ejemplo
    -------
        fig, ax = plt.subplots()
        r = plot_cbs_fit(ax, theta_mrad, mean_info, wavelength_nm=532)
        print(r)
    """
    if result is None:
        result = fit_cbs(
            theta_mrad, enhancement,
            wavelength_nm=wavelength_nm,
            model=model,
            center_mrad=center_mrad,
        )

    theta  = np.asarray(theta_mrad, dtype=float)
    y      = np.asarray(enhancement, dtype=float)
    center = (center_mrad if center_mrad is not None
              else float(theta[np.argmax(y)]))

    # Datos
    _dk = dict(fmt=".", ms=3, alpha=0.55, color="steelblue", zorder=2)
    if data_kw:
        _dk.update(data_kw)
    ax.errorbar(theta, y, **_dk,
                label=f"{label_prefix}datos" if label_prefix else "_nolegend_")

    # Curva ajustada (alta resolución)
    theta_fine  = np.linspace(theta.min(), theta.max(), 1000)
    theta_c_rad = (theta_fine - center) * 1e-3
    q_fine      = result.kl * theta_c_rad
    model_fn    = _MODELS[result.model][1]
    y_fit       = result.A * model_fn(q_fine, result.eps) + result.B

    _fk = dict(color="tomato", lw=1.6, zorder=3)
    if fit_kw:
        _fk.update(fit_kw)
    fit_label = (f"{label_prefix}fit — "
                 f"$l={result.l_um:.1f}\\,\\mu$m, $kl={result.kl:.0f}$")
    ax.plot(theta_fine, y_fit, **_fk, label=fit_label)

    # Línea del FWHM
    if show_fwhm:
        half_width = result.fwhm_mrad / 2.0
        y_half     = result.B + result.A * float(
            model_fn(np.array([result.kl * half_width * 1e-3]), result.eps)[0])
        ax.axhline(y_half, color="gray", lw=0.7, ls="--", alpha=0.6)
        ax.annotate(
            f"FWHM = {result.fwhm_mrad:.3f} mrad",
            xy=(center + half_width, y_half),
            xytext=(0, 6), textcoords="offset points",
            fontsize=7, color="gray", ha="left",
        )
        ax.axvspan(center - half_width, center + half_width,
                   alpha=0.06, color="tomato", zorder=0)

    # Ecuación en el gráfico
    if show_eq:
        ax.text(
            0.97, 0.95, _EQ_LABELS[result.model],
            transform=ax.transAxes, fontsize=7,
            ha="right", va="top",
            color=plt.rcParams.get("text.color", "black"),
            alpha=0.7,
        )

    return result


# ─────────────────────────────────────────────
#  Helper: figura de resumen de barrido
# ─────────────────────────────────────────────

def plot_sweep_cbs(
    axes_data:      list[tuple[plt.Axes, np.ndarray, np.ndarray, dict]],
    wavelength_nm:  float = 532.0,
    model:          Literal["diff", "bound", "deco"] = "diff",
    suptitle:       str = "",
) -> list[CBSFitResult]:
    """
    Ajusta y grafica múltiples perfiles CBS en una lista de Axes.

    axes_data : lista de (ax, theta_mrad, enhancement, kw_extra)
        kw_extra puede tener: label_prefix, color, center_mrad, etc.

    Ejemplo
    -------
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        results = plot_sweep_cbs([
            (axes[0], theta, enh_a070, dict(label_prefix="a=0.07 ")),
            (axes[1], theta, enh_a085, dict(label_prefix="a=0.085 ")),
            (axes[2], theta, enh_a100, dict(label_prefix="a=0.10 ")),
        ], wavelength_nm=532)
    """
    results = []
    for ax, theta, enh, kw in axes_data:
        prefix    = kw.pop("label_prefix", "")
        center    = kw.pop("center_mrad", None)
        color     = kw.pop("color", None)
        data_kw   = dict(color=color) if color else {}
        fit_kw    = dict(color=color) if color else {}
        r = plot_cbs_fit(
            ax, theta, enh,
            wavelength_nm=wavelength_nm,
            model=model,
            center_mrad=center,
            label_prefix=prefix,
            data_kw=data_kw or None,
            fit_kw=fit_kw or None,
        )
        results.append(r)
        ax.legend(fontsize=7)
        ax.set_xlabel("Scattering angle (mrad)")
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    if suptitle:
        plt.suptitle(suptitle, fontsize=11)

    return results


# ─────────────────────────────────────────────
#  Demo / test rápido
# ─────────────────────────────────────────────

if __name__ == "__main__":

    rng = np.random.default_rng(42)
    lam_nm = 532.0
    k      = 2 * np.pi / (lam_nm * 1e-9)
    kl_true = 250.0

    theta_mrad = np.linspace(-40, 40, 201)
    theta_rad  = theta_mrad * 1e-3
    q_true     = kl_true * theta_rad
    y_clean    = 1.0 / (1.0 + np.abs(q_true)) ** 2 + 1.0
    y_noisy    = y_clean + rng.normal(0, 0.025, size=y_clean.shape)

    # ── Fit ──────────────────────────────────────────────────────────────────
    result = fit_cbs(theta_mrad, y_noisy, wavelength_nm=lam_nm, model="diff")
    print(result)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, m in zip(axes, ["diff", "bound", "deco"]):
        r = plot_cbs_fit(
            ax, theta_mrad, y_noisy,
            wavelength_nm=lam_nm,
            model=m,
            show_fwhm=True,
            show_eq=True,
        )
        name = _MODELS[m][0]
        ax.set_title(f"{name}\n$kl={r.kl:.1f}$,  $l={r.l_um:.2f}\\,\\mu$m",
                     fontsize=9)
        ax.set_xlabel("Scattering angle (mrad)")
        ax.legend(fontsize=7)

    axes[0].set_ylabel(r"$I_\mathrm{co}$ enhancement")
    plt.tight_layout()
    plt.savefig("cbs_fit_demo.png", dpi=150)
    plt.show()
    print(f"\nTrue kl = {kl_true:.1f},  fitted kl = {result.kl:.2f}")
    print(f"True l  = {kl_true/k*1e6:.3f} µm,  fitted l = {result.l_um:.3f} µm")