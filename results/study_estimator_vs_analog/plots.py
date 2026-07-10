from utils.loaders import load_sweep
from utils.styles import apply
from utils.analysis import cbs_profiles, circular, keep, azimuthal_average

import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

apply(context="paper", col="double")

save_path = "/home/niaggar/Developer/luminis-mc/temporal_results"
folder = "study_estimator_vs_analog"

# Canal helicidad-preservada: en convencion de lab frame el estado circular
# ortogonal preserva la helicidad al invertirse la propagacion -> "cross".
CHANNEL = "cross"

# Metrica de pico: amplitud en ventana angular FIJA, sumando coherente e
# incoherente sobre (theta, phi) ANTES del cociente. Insesgada a todo N
# (los bins hambrientos del analogico se agregan, no se promedian).
W_MRAD = 2.5        # ~ HWHM del cono; declarar en el caption de F15
MIN_REPS = 2        # con R=1 corre en modo piloto (sin sigma)

RUN_RE = re.compile(
    r"radius_(?P<radius>[\d.]+)_(?P<mode>estimator|analog)_N(?P<n>\d+)_rep(?P<rep>\d+)"
)

sweep_data = load_sweep(folder, base_path=Path(save_path))
print(f"Loaded {len(sweep_data)} datasets")


# ===========================================================================
# Carga por corrida: eta(theta) para F16/F15b + pico de ventana para F15
# ===========================================================================
runs = []
for key, loader in sweep_data.items():
    m = RUN_RE.search(key)
    if m is None:
        print(f"  skip (nombre no reconocido): {key}")
        continue

    processed = loader.processed_cbs("farfield_cbs")

    # perfil 1D para F16 / F15b
    prof1d = cbs_profiles(processed, basis=circular, time_index=0,
                          reduce=azimuthal_average)
    eta = np.asarray(prof1d.enhancement[CHANNEL])

    # pico de ventana: mapas 2D coherente/incoherente, suma-luego-cociente
    prof2d = cbs_profiles(processed, basis=circular, time_index=0, reduce=keep)
    coh = np.asarray(prof2d.coherent[CHANNEL])         # (n_theta, n_phi)
    inc = np.asarray(prof2d.incoherent[CHANNEL])
    mask = np.asarray(prof2d.theta) * 1e3 <= W_MRAD
    peak = float(coh[mask].sum() / inc[mask].sum())

    runs.append({
        "key": key,
        "mode": m["mode"],
        "n": int(m["n"]),
        "rep": int(m["rep"]),
        "theta": np.asarray(prof1d.theta),
        "eta": eta,
        "eta2d": np.asarray(prof2d.enhancement[CHANNEL]),
        "peak": peak,
        "T": float(loader.params_flat["runtime_s"]),
    })

print(f"{len(runs)} corridas reconocidas")
th = runs[0]["theta"] * 1e3   # mrad
n_window = int((runs[0]["theta"] * 1e3 <= W_MRAD).sum())
print(f"ventana de pico: theta <= {W_MRAD} mrad ({n_window} bins)")


# ===========================================================================
# Chequeo de simetria azimutal -- sobre el ESTIMADOR de mayor N
# (el analogico en el pico es ruido de Poisson y no diagnostica nada)
# ===========================================================================
big = max(runs, key=lambda r: (r["mode"] == "estimator", r["n"]))
mask = big["theta"] * 1e3 <= W_MRAD
per_phi = big["eta2d"][mask].mean(axis=0)
spread = (per_phi.max() - per_phi.min()) / per_phi.mean()
print(f"\nSimetria azimutal ({big['mode']} N={big['n']:.0e} rep{big['rep']}):")
print(f"  eta ventana por phi = {np.array2string(per_phi, precision=5)}")
print(f"  spread relativo     = {spread:.3%}  (esperado ~nivel de sigma_rel del run)")


# ===========================================================================
# Tabla (mode, N) -> replicas; estadisticas F15
# ===========================================================================
table = {}
for r in runs:
    d = table.setdefault((r["mode"], r["n"]), {"etas": [], "peaks": [], "T": []})
    d["etas"].append(r["eta"])
    d["peaks"].append(r["peak"])
    d["T"].append(r["T"])

stats = {}   # (mode, n) -> (mean_peak, sigma_rel, T_mean, R, eps)
print(f"\n{'modo':<10s} {'N':>12s} {'R':>3s} {'pico_vent':>10s} {'sigma_rel':>10s} {'T[s]':>8s} {'eps[1/s]':>10s}")
for (mode, n), d in sorted(table.items()):
    R = len(d["peaks"])
    p = np.array(d["peaks"])
    T = float(np.mean(d["T"]))
    if R < MIN_REPS:
        print(f"{mode:<10s} {n:>12,d} {R:>3d} {p.mean():>10.4f} {'--':>10s} {T:>8.1f} {'--':>10s}  (piloto)")
        continue
    sig_rel = p.std(ddof=1) / p.mean()
    eps = 1.0 / (sig_rel**2 * T)
    stats[(mode, n)] = (p.mean(), sig_rel, T, R, eps)
    print(f"{mode:<10s} {n:>12,d} {R:>3d} {p.mean():>10.4f} {sig_rel:>10.3e} {T:>8.1f} {eps:>10.3e}")

# gate de insesgadez de la METRICA: la media analogica debe ser plana en N
ana_means = [v[0] for (m, _), v in sorted(stats.items()) if m == "analog"]
if len(ana_means) >= 2:
    drift = (max(ana_means) - min(ana_means)) / np.mean(ana_means)
    print(f"\nderiva de la media analogica a lo largo de N: {drift:.2%} "
          f"(si >> sigma_rel, la metrica sigue sesgada)")


# ===========================================================================
# F15 -- convergencia sigma_rel(N) + eficiencia eps(N)
# ===========================================================================
if stats:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

    for mode, color in (("estimator", "C0"), ("analog", "C1")):
        pts = sorted((n, v) for (m, n), v in stats.items() if m == mode)
        if len(pts) < 2:
            continue
        N = np.array([n for n, _ in pts], dtype=float)
        S = np.array([v[1] for _, v in pts])
        E = np.array([v[4] for _, v in pts])
        b, a = np.polyfit(np.log(N), np.log(S), 1)
        ax1.loglog(N, S, "o", color=color, label=f"{mode} (pendiente {b:+.2f})")
        ax1.loglog(N, np.exp(a) * N**b, "-", color=color, lw=1)
        ax2.loglog(N, E, "o-", color=color, label=mode)
        print(f"pendiente log-log {mode}: {b:+.3f} (esperado -0.5)")

    Ng = np.array([min(n for _, n in stats), max(n for _, n in stats)], dtype=float)
    s0 = next(iter(stats.values()))[1]
    ax1.loglog(Ng, s0 * (Ng / Ng[0]) ** -0.5, "k--", lw=0.8, alpha=0.5, label=r"$N^{-1/2}$")

    ax1.set_xlabel(r"$N_{\rm photons}$")
    ax1.set_ylabel(r"$\sigma_{\rm rel}(\eta_{\rm w})$")
    ax1.grid(alpha=0.2)
    ax1.legend(loc="lower left", fontsize=8)
    ax2.set_xlabel(r"$N_{\rm photons}$")
    ax2.set_ylabel(r"$\epsilon = 1/(\sigma_{\rm rel}^2\,T)$ [s$^{-1}$]")
    ax2.grid(alpha=0.2)
    ax2.legend(loc="center left", fontsize=8)
    fig.tight_layout()
    # fig.savefig(out, dpi=200)
    plt.show()

    eps_est = np.mean([v[4] for (m, _), v in stats.items() if m == "estimator"])
    eps_ana = np.mean([v[4] for (m, _), v in stats.items() if m == "analog"])
    print(f"\neps medio: estimator = {eps_est:.3e}, analog = {eps_ana:.3e}")
    print(f"factor de eficiencia estimador/analogico = {eps_est / eps_ana:.0f}x")
else:
    print("\nF15 omitido: ninguna combinacion (mode, N) alcanza "
          f"{MIN_REPS} replicas (modo piloto)")


# ===========================================================================
# F16 -- insesgadez: referencias fusionadas del N tope de cada modo
# ===========================================================================
EXCLUDE_FIRST_BIN = True   # bin 0 analogico: hambriento de cuentas, SEM no
                           # gaussiana subestimada -> excluir del chi2 (se
                           # declara en el caption); el overlay lo muestra igual

refs = {}
for mode in ("estimator", "analog"):
    ns = [n for (m, n) in table if m == mode]
    if not ns:
        continue
    n_top = max(ns)
    etas = np.array(table[(mode, n_top)]["etas"])      # (R, n_theta)
    R = etas.shape[0]
    refs[mode] = {
        "n": n_top, "R": R,
        "mean": etas.mean(axis=0),
        "sem": etas.std(axis=0, ddof=1) / np.sqrt(R) if R >= 2 else None,
    }

if "estimator" in refs and "analog" in refs:
    e, a = refs["estimator"], refs["analog"]
    have_bands = e["sem"] is not None and a["sem"] is not None

    if have_bands:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(5.2, 5.4), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
    else:
        fig, ax1 = plt.subplots(figsize=(5.2, 3.8))
        ax2 = None

    ax1.plot(th, a["mean"], "-", color="C1",
             label=f"analog, $N=${a['n']:.0e} ($\\times${a['R']})")
    ax1.plot(th, e["mean"], "--", color="C0",
             label=f"estimator, $N=${e['n']:.0e} ($\\times${e['R']})")
    if have_bands:
        ax1.fill_between(th, a["mean"] - a["sem"], a["mean"] + a["sem"], color="C1", alpha=0.3)
        ax1.fill_between(th, e["mean"] - e["sem"], e["mean"] + e["sem"], color="C0", alpha=0.3)
    ax1.axhline(2.0, color="k", lw=0.6, ls=":", alpha=0.6)
    ax1.set_ylabel(r"$\eta_{++}(\theta)$")
    ax1.grid(alpha=0.2)
    ax1.legend(loc="upper right", fontsize=8)

    if have_bands:
        sig_comb = np.sqrt(e["sem"]**2 + a["sem"]**2)
        resid = (e["mean"] - a["mean"]) / sig_comb
        i0 = 1 if EXCLUDE_FIRST_BIN else 0
        chi2_dof = float(np.mean(resid[i0:]**2))
        # benchmark: con R replicas, E[t^2] = nu/(nu-2), nu = R-1
        nu = min(e["R"], a["R"]) - 1
        chi2_ref = nu / (nu - 2)
        print(f"\nF16: chi2/dof = {chi2_dof:.2f} "
              f"(benchmark con sigma estimada, R={min(e['R'], a['R'])}: ~{chi2_ref:.2f}"
              f"{'; primer bin excluido' if EXCLUDE_FIRST_BIN else ''})")
        ax2.plot(th[i0:], resid[i0:], "o", ms=2.5, color="k")
        if EXCLUDE_FIRST_BIN:
            ax2.plot(th[:1], resid[:1], "o", ms=2.5, mfc="none", color="k", alpha=0.5)
        ax2.axhline(0, color="k", lw=0.6)
        for y in (-2, 2):
            ax2.axhline(y, color="k", lw=0.6, ls="--", alpha=0.4)
        ax2.set_xlabel(r"$\theta$ [mrad]")
        ax2.set_ylabel(r"$\Delta\eta/\sigma$")
        ax2.grid(alpha=0.2)
        ax2.text(0.98, 0.9, rf"$\chi^2/\mathrm{{dof}} = {chi2_dof:.2f}$",
                 transform=ax2.transAxes, ha="right", va="top", fontsize=8)
    else:
        ax1.set_xlabel(r"$\theta$ [mrad]")
        print("\nF16 (piloto, R=1): overlay sin bandas -- solo visual")

    fig.tight_layout()
    # fig.savefig(out, dpi=200)
    plt.show()


# ===========================================================================
# F15b -- sigma_rel(theta) del par (estimator, analog) a costo mas parecido
# ===========================================================================
pairs = [((me, ne), (ma, na))
         for (me, ne) in stats if me == "estimator"
         for (ma, na) in stats if ma == "analog"]
if pairs:
    best = min(pairs, key=lambda p: abs(stats[p[0]][2] - stats[p[1]][2]))
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for key, color in zip(best, ("C0", "C1")):
        mode, n = key
        etas = np.array(table[key]["etas"])
        sig_th = etas.std(axis=0, ddof=1) / etas.mean(axis=0)
        T = stats[key][2]
        ax.semilogy(th, sig_th, color=color,
                    label=f"{mode}, $N=${n:.0e} ($T\\approx${T:.0f} s)")
    ax.set_xlabel(r"$\theta$ [mrad]")
    ax.set_ylabel(r"$\sigma_{\rm rel}(\eta)$ por bin")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    # fig.savefig(out, dpi=200)
    plt.show()
    print(f"F15b: par a costo comparable = {best[0]} vs {best[1]}")