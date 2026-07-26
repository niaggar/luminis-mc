"""
cbs_fit.py — Ajuste de perfiles de retrodispersión coherente (CBS)
==================================================================

Implementa el procedimiento de particle sizing de:

  T. Iwai, K. Ishii, T. Asakura, "Particle sizing based on enhanced
  backscatterings of light from dense disordered media",
  Proc. SPIE 3729, 293-297 (1999).

Dos funciones modelo para la parte coherente I_C(theta):

  (Ec. 1)  Lorentziana empírica     I_C(th) = 1 / (1 + |th|/P2)
           (Ec. 2)                  l* = lambda_med / (4 pi P2)  <=>  P2 = 1/(2 k l*)

  (Ec. 3)  Perfil de Akkermans et al. [PRL 56, 1471 (1986)]:
           I_C(th) = 3/(7 (1+x)^2) * [ 1 + (1 - e^{-4x/3}) / x ],  x = P1 |th|
           (Ec. 4)                  l* = lambda_med P1 / (2 pi)  <=>  P1 = k l*

Ambas están normalizadas a I_C(0) = 1 (para la Ec. 3: lim_{x->0}
(1-e^{-4x/3})/x = 4/3 y (3/7)(1+4/3) = 1).

El observable de la simulación es el factor de realce
    eta(theta) = S0_coh / S0_incoh,
que incluye el fondo difuso. El modelo ajustado es por tanto

    eta_model(theta) = B + A * I_C(theta),

con B ~ 1 (fondo) y A <= 1 (reducción del ápice por dispersión simple y
canal de polarización). Por defecto B se fija en 1, que es exacto por
construcción del estimador (eta -> 1 en las alas); puede liberarse.

Notas de rigor:
  * P1 * theta debe ser adimensional: P1 = k l*, con k = 2 pi n_host / lambda_0
    el número de onda EN EL MEDIO. Por eso el ajuste recibe `k` explícito.
  * La Ec. (1) del paper está impresa sin cuadrado en el denominador
    ("1 + theta/P2"); se implementa tal cual (`model="lorentz_iwai"`) y,
    como control, la Lorentziana estándar con cuadrado (`model="lorentz_sq"`).
  * El perfil de Akkermans tiene un cúspide triangular en theta = 0; con
    bins angulares finitos el dato es el PROMEDIO del modelo sobre el bin.
    `bin_average=True` promedia el modelo con cuadratura de Gauss-Legendre.
  * El ajuste usa Levenberg-Marquardt (scipy method='lm'), como en el paper
    (Numerical Recipes). Pesos: sigma del ensamble si se provee.

Uso mínimo (con tus objetos):

    theta, q, m, s = profile_stats(sweep_data_lineal, g_, 0, "co")
    res = fit_cbs_profile(theta, m, sigma=s, model="akkermans", k=K_MED)
    print(res.summary())
    # res.ell_star, res.ell_star_err, res.A, res.chi2_red, res.eval(theta)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.optimize import curve_fit

__all__ = [
    "akkermans_shape",
    "lorentz_iwai_shape",
    "lorentz_sq_shape",
    "fit_cbs_profile",
    "fit_sweep",
    "CBSFitResult",
]

# ----------------------------------------------------------------------------
# Formas normalizadas I_C(theta) con I_C(0) = 1
# ----------------------------------------------------------------------------

def akkermans_shape(theta: np.ndarray, P1: float) -> np.ndarray:
    """Perfil de Akkermans-Wolf-Maynard, Ec. (3) de Iwai et al. (1999).

    x = P1 * |theta| = k l* |theta|.  Usa z0 = 2 l*/3 (aprox. de difusión,
    de ahí el 4/3 = 2 z0/l* en la exponencial).  Serie de Taylor cerca de
    x=0 para evitar 0/0:  (1 - e^{-4x/3})/x = 4/3 - (8/9) x + O(x^2).
    """
    x = P1 * np.abs(np.asarray(theta, dtype=float))
    out = np.empty_like(x)
    small = x < 1e-8
    xs = x[~small]
    out[~small] = 3.0 / (7.0 * (1.0 + xs) ** 2) * (1.0 + (1.0 - np.exp(-4.0 * xs / 3.0)) / xs)
    xt = x[small]
    out[small] = 3.0 / (7.0 * (1.0 + xt) ** 2) * (1.0 + 4.0 / 3.0 - (8.0 / 9.0) * xt)
    return out


def lorentz_iwai_shape(theta: np.ndarray, P2: float) -> np.ndarray:
    """Ec. (1) de Iwai et al. (1999) tal como está impresa: 1/(1 + |theta|/P2).

    Es empírica (los autores lo advierten: "not based on the theoretical
    background"); su ala ~ 1/theta reproduce el decaimiento de largo alcance
    del cono mejor que una Lorentziana con cuadrado.
    """
    return 1.0 / (1.0 + np.abs(np.asarray(theta, dtype=float)) / P2)


def lorentz_sq_shape(theta: np.ndarray, P2: float) -> np.ndarray:
    """Lorentziana estándar 1/(1 + (theta/P2)^2), como variante de control."""
    t = np.asarray(theta, dtype=float) / P2
    return 1.0 / (1.0 + t * t)


_SHAPES: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "akkermans": akkermans_shape,
    "lorentz_iwai": lorentz_iwai_shape,
    "lorentz_sq": lorentz_sq_shape,
}

# Relación P -> l*  (l* en las mismas unidades que 1/k y que theta^-1 * long.)
#   akkermans:    l* = P1 / k          (Ec. 4 con lambda_med = 2 pi / k)
#   lorentz_*:    l* = 1 / (2 k P2)    (Ec. 2 con lambda_med = 2 pi / k)
def _ell_star_from_param(model: str, P: float, P_err: float, k: float):
    if model == "akkermans":
        return P / k, P_err / k
    # Lorentzianas: l* = 1/(2 k P2); propagación lineal: dl* = dP2/(2 k P2^2)
    return 1.0 / (2.0 * k * P), P_err / (2.0 * k * P * P)


# ----------------------------------------------------------------------------
# Resultado
# ----------------------------------------------------------------------------

@dataclass
class CBSFitResult:
    model: str
    A: float
    A_err: float
    P: float                 # P1 (akkermans) o P2 (lorentzianas), en rad^-1 / rad
    P_err: float
    B: float                 # fondo (1.0 si se fijó)
    B_err: float
    k: Optional[float]       # número de onda usado (None si no se dio)
    ell_star: Optional[float]
    ell_star_err: Optional[float]
    chi2_red: float
    ndof: int
    theta_range: tuple
    label: str = ""
    pcov: np.ndarray = field(default=None, repr=False)

    def eval(self, theta: np.ndarray) -> np.ndarray:
        """Evalúa el modelo ajustado eta(theta)."""
        return self.B + self.A * _SHAPES[self.model](theta, self.P)

    @property
    def fwhm(self) -> float:
        """Ancho a media altura del pico coherente (numérico, en rad)."""
        f = lambda t: _SHAPES[self.model](t, self.P) - 0.5
        lo, hi = 0.0, 1.0 / self.P if self.model == "akkermans" else self.P
        while f(hi) > 0:
            hi *= 2.0
        for _ in range(80):  # bisección
            mid = 0.5 * (lo + hi)
            (lo, hi) = (mid, hi) if f(mid) > 0 else (lo, mid)
        return 2.0 * lo

    def summary(self) -> str:
        lines = [
            f"[{self.label or self.model}]  modelo = {self.model}",
            f"  A  = {self.A:.4f} ± {self.A_err:.4f}   (apice eta(0) = {self.B + self.A:.4f})",
            f"  P  = {self.P:.6g} ± {self.P_err:.2g}  [1/rad]" if self.model == "akkermans"
            else f"  P  = {self.P:.6g} ± {self.P_err:.2g}  [rad]",
            f"  B  = {self.B:.4f} ± {self.B_err:.4f}",
            f"  FWHM = {self.fwhm*1e3:.3f} mrad",
            f"  chi2_red = {self.chi2_red:.3f}  (ndof = {self.ndof})",
        ]
        if self.ell_star is not None:
            lines.append(f"  l* = {self.ell_star:.6g} ± {self.ell_star_err:.2g}  [unid. de 1/k]")
        return "\n".join(lines)


# ----------------------------------------------------------------------------
# Ajuste
# ----------------------------------------------------------------------------

def _bin_averaged(shape, theta, P, dtheta, ngl=5):
    """Promedia el modelo sobre cada bin [theta - d/2, theta + d/2] (G-Legendre).

    Necesario porque el cúspide de Akkermans en theta=0 hace que el valor
    puntual en el centro del bin sobreestime el promedio que mide el sensor.
    """
    xg, wg = np.polynomial.legendre.leggauss(ngl)   # nodos en [-1, 1]
    th = theta[:, None] + 0.5 * dtheta * xg[None, :]
    vals = shape(th, P)
    return 0.5 * (vals @ wg)


def fit_cbs_profile(
    theta: np.ndarray,
    eta: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    model: str = "akkermans",
    k: Optional[float] = None,
    theta_max: Optional[float] = None,
    fit_baseline: bool = False,
    bin_average: bool = True,
    label: str = "",
) -> CBSFitResult:
    """Ajusta eta(theta) = B + A * I_C(theta; P) por Levenberg-Marquardt.

    Parámetros
    ----------
    theta : ángulos de retrodispersión en radianes (>= 0; si vienen con
        signo se usa |theta| — los modelos son pares).
    eta : factor de realce medido (media del ensamble).
    sigma : desviación estándar del ensamble (pesos 1/sigma^2,
        absolute_sigma=True). Si None, ajuste sin pesos.
    model : "akkermans" (Ec. 3), "lorentz_iwai" (Ec. 1 tal como está
        impresa) o "lorentz_sq" (Lorentziana con cuadrado, control).
    k : número de onda en el medio, k = 2 pi n_host / lambda_0, en las
        unidades inversas de las de l* deseadas. Si se da, se reporta
        l* con su error (Ecs. 2 y 4).
    theta_max : recorte del rango de ajuste (rad). Útil para excluir alas
        donde el modelo semi-infinito escalar deja de ser válido.
    fit_baseline : si True, B es libre; si False, B = 1 (exacto por
        construcción de eta en el estimador).
    bin_average : promediar el modelo sobre el ancho de bin (recomendado).
    """
    if model not in _SHAPES:
        raise ValueError(f"model debe ser uno de {list(_SHAPES)}")
    shape = _SHAPES[model]

    theta = np.abs(np.asarray(theta, dtype=float))
    eta = np.asarray(eta, dtype=float)
    order = np.argsort(theta)
    theta, eta = theta[order], eta[order]
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)[order]

    mask = np.isfinite(theta) & np.isfinite(eta)
    if theta_max is not None:
        mask &= theta <= theta_max
    theta, eta = theta[mask], eta[mask]
    if sigma is not None:
        sigma = sigma[mask]
        # sigma = 0 (p.ej. bin sin varianza) rompe los pesos: usar la mediana
        pos = sigma > 0
        if not pos.all():
            fill = np.median(sigma[pos]) if pos.any() else 1.0
            sigma = np.where(pos, sigma, fill)

    if theta.size < 4:
        raise ValueError("Muy pocos puntos para ajustar (>= 4 requeridos).")

    dtheta = np.median(np.diff(np.unique(theta))) if bin_average else 0.0

    # --- Valores iniciales físicamente motivados -------------------------
    B0 = 1.0
    A0 = max(eta[np.argmin(theta)] - B0, 0.05)
    # HWHM empírico: primer theta donde eta - B0 cae a A0/2
    half = eta - B0 <= 0.5 * A0
    th_h = theta[half][0] if half.any() and theta[half][0] > 0 else theta[max(1, theta.size // 4)]
    if model == "akkermans":
        P0 = 0.34 / th_h          # I_C(x)=0.5 en x ~ 0.34  =>  P1 ~ 0.34/HWHM
    else:
        P0 = th_h                 # HWHM ~ P2 en ambas lorentzianas

    def model_eta(th, A, P, B=1.0):
        if bin_average and dtheta > 0:
            core = _bin_averaged(shape, th, P, dtheta)
        else:
            core = shape(th, P)
        return B + A * core

    if fit_baseline:
        f = lambda th, A, P, B: model_eta(th, A, P, B)
        p0 = [A0, P0, B0]
    else:
        f = lambda th, A, P: model_eta(th, A, P, 1.0)
        p0 = [A0, P0]

    popt, pcov = curve_fit(
        f, theta, eta, p0=p0,
        sigma=sigma, absolute_sigma=sigma is not None,
        method="lm", maxfev=20000,
    )
    perr = np.sqrt(np.diag(pcov))

    A, P = popt[0], abs(popt[1])           # el modelo es par en P -> signo espurio
    A_err, P_err = perr[0], perr[1]
    if fit_baseline:
        B, B_err = popt[2], perr[2]
    else:
        B, B_err = 1.0, 0.0

    resid = eta - f(theta, *popt)
    if sigma is not None:
        resid = resid / sigma
    ndof = theta.size - len(popt)
    chi2_red = float(resid @ resid) / ndof

    if k is not None:
        ell, ell_err = _ell_star_from_param(model, P, P_err, k)
    else:
        ell = ell_err = None

    return CBSFitResult(
        model=model, A=A, A_err=A_err, P=P, P_err=P_err, B=B, B_err=B_err,
        k=k, ell_star=ell, ell_star_err=ell_err,
        chi2_red=chi2_red, ndof=ndof,
        theta_range=(float(theta.min()), float(theta.max())),
        label=label, pcov=pcov,
    )


# ----------------------------------------------------------------------------
# Barrido sobre familias (radios) y canales, siguiendo tu bucle de ploteo
# ----------------------------------------------------------------------------

def fit_sweep(
    sweep_data, grouped_data, profile_stats, k,
    basis,
    channels=("co", "cross"), phi=0,
    models=("akkermans", "lorentz_iwai"),
    **fit_kwargs,
):
    """Ajusta todos los perfiles del barrido y devuelve una tabla de resultados.

    Replica tu acceso a datos:  theta, q, m, s = profile_stats(sweep, g, phi, ch)
    Devuelve un pandas.DataFrame con una fila por (grupo, canal, modelo).
    """
    import pandas as pd

    rows = []
    results = {}
    for g in grouped_data:
        for ch in channels:
            theta, q, m, s = profile_stats(sweep_data, g, phi, ch, basis=basis)
            for mod in models:
                res = fit_cbs_profile(
                    theta, m, sigma=s, model=mod, k=k,
                    label=f"{g.name}/{ch}", **fit_kwargs,
                )
                results[(g.name, ch, mod)] = res
                rows.append({
                    "group": g.name, "channel": ch, "model": mod,
                    "A": res.A, "A_err": res.A_err,
                    "P": res.P, "P_err": res.P_err,
                    "eta0": res.B + res.A,
                    "FWHM_mrad": res.fwhm * 1e3,
                    "ell_star": res.ell_star, "ell_star_err": res.ell_star_err,
                    "chi2_red": res.chi2_red,
                })
    return pd.DataFrame(rows), results









#TESTS


# 1

# N_HOST = 1.33
# LAMBDA0 = 514.5e-9                      # [m] — usa las unidades de tu simulación
# K_MED = 2 * np.pi * N_HOST / LAMBDA0    # k en el medio; l* saldrá en metros

# # Tabla completa del barrido (una fila por radio × canal × modelo)
# df, fits = fit_sweep(sweep_data_lineal, grouped_data, profile_stats, k=K_MED)
# print(df.to_string(float_format=lambda x: f"{x:.4g}"))

# # Superponer un ajuste en tu figura:
# for c, g_ in zip(COL, grouped_data):
#     theta, q, m, s = profile_stats(sweep_data_lineal, g_, 0, "co")
#     res = fits[(g_.name, "co", "akkermans")]
#     th_fine = np.linspace(0, theta.max(), 400)
#     q_fine = np.interp(th_fine, theta, q)          # mismo mapeo theta -> q de tus datos
#     ax1.plot(np.r_[-q_fine[::-1], q_fine],
#              np.r_[res.eval(th_fine)[::-1], res.eval(th_fine)],
#              color=c, ls=":", lw=0.8)








# 2

# RADIOS = ["$35$ nm", "$75$ nm", "$175$ nm"]   # <-- elige aquí los tres radios
# CANAL = "co"

# N_HOST = 1.33
# LAMBDA0 = 514.5e-9                      # [m] — usa las unidades de tu simulación
# K_MED = 2 * np.pi * N_HOST / LAMBDA0    # k en el medio; l* saldrá en metros

# # Tabla completa del barrido (una fila por radio × canal × modelo)
# df, fits = fit_sweep(sweep_data_lineal, grouped_data, profile_stats, k=K_MED)
# print(df.to_string(float_format=lambda x: f"{x:.4g}"))

# # Superponer un ajuste en tu figura:
# for c, g_ in zip(COL, grouped_data):
#     theta, q, m, s = profile_stats(sweep_data_lineal, g_, 0, "co")
#     res = fits[(g_.name, "co", "akkermans")]
#     th_fine = np.linspace(0, theta.max(), 400)
#     q_fine = np.interp(th_fine, theta, q)          # mismo mapeo theta -> q de tus datos
#     ax1.plot(np.r_[-q_fine[::-1], q_fine],
#              np.r_[res.eval(th_fine)[::-1], res.eval(th_fine)],
#              color=c, ls=":", lw=0.8)

# print([g.name for g in grouped_data])

# fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 0.4*TEXTWIDTH_IN), sharey=True)

# for ax, name in zip(axes, RADIOS):
#     g_ = next(g for g in grouped_data if g.name == name)

#     theta, q, m, s = profile_stats(sweep_data_lineal, g_, 0, CANAL)
#     qs, ms, ss = mirror(q, m, s)
#     ax.plot(qs, ms, lw=0.8, label="MC")
#     ax.fill_between(qs, ms - ss, ms + ss, alpha=0.2, lw=0)

#     res = fits[(name, CANAL, "akkermans")]
#     th_fine = np.linspace(0, theta.max(), 400)
#     q_fine = np.interp(th_fine, theta, q)
#     eta_fine = res.eval(th_fine)
#     ax.plot(np.r_[-q_fine[::-1], q_fine], np.r_[eta_fine[::-1], eta_fine],
#             "k--", lw=0.8, label="Akkermans")

#     ax.set_title(name, loc="left")
#     ax.set_xlabel(r"$q$")
#     ax.set_xlim(-10, 10)
#     ax.text(0.03, 0.95,
#             rf"$\ell^*={res.ell_star*1e6:.1f}\pm{res.ell_star_err*1e6:.1f}\,\mu$m"
#             "\n"
#             rf"$A={res.A:.2f}$,  $\chi^2_\nu={res.chi2_red:.1f}$",
#             transform=ax.transAxes, va="top", fontsize=7)

# axes[0].set_ylabel(r"$\eta$")
# axes[0].legend(frameon=False, fontsize=7, loc="upper right")
# fig.savefig(FIGDIR / "cbs_fit_tres_radios.pdf")