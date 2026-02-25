"""
Speckle Sweep Analysis
======================
Loads results from sweep_A (n_photons sweep) and sweep_B (anisotropy sweep)
and produces the diagnostic plots described in the thesis study.

Run AFTER sim_speckle_sweep.py has completed.
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.stats import entropy

from luminis_mc import ResultsLoader   # adjust import if needed

# ─────────────────────────────────────────────────────────────────────────────
# Helpers (same as your analysis notebook)
# ─────────────────────────────────────────────────────────────────────────────

def load_quality(sweep_dir):
    """
    Walk all run sub-directories in a sweep folder and collect the scalar
    quality metrics saved by run_speckle().

    Returns a dict of lists:
        runtime_s, anisotropy_g, contrast_x, contrast_y, kl_x, kl_y
    plus whatever param you were sweeping (n_photons or radius).
    """
    results = {k: [] for k in
               ["n_photons", "radius_real", "anisotropy_g",
                "runtime_s", "contrast_x", "contrast_y",
                "kl_x", "kl_y"]}

    run_dirs = sorted(glob.glob(f"{sweep_dir}/runs/*/"))
    for rdir in run_dirs:
        try:
            data = ResultsLoader(rdir)
            params = data.params()

            for key in ["n_photons", "radius_real", "anisotropy_g"]:
                results[key].append(params.get(key, np.nan))

            for key in ["runtime_s", "contrast_x", "contrast_y", "kl_x", "kl_y"]:
                val = data.derived(f"quality/{key}")
                results[key].append(float(val[0]) if val is not None else np.nan)

        except Exception as e:
            print(f"  Warning: could not load {rdir}: {e}")

    return {k: np.array(v) for k, v in results.items()}


def kl_from_exponential(I_inst, I_avg, n_bins=80):
    mask = I_avg > 0.01 * np.max(I_avg)
    eta  = I_inst[mask] / I_avg[mask]
    hist, edges = np.histogram(eta, bins=n_bins, range=(0, 8), density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    bin_w   = edges[1] - edges[0]
    p_sim    = np.clip(hist * bin_w,         1e-10, None)
    p_theory = np.clip(np.exp(-centers)*bin_w, 1e-10, None)
    return float(entropy(p_sim, p_theory))


# ─────────────────────────────────────────────────────────────────────────────
# Locate sweep output directories  (edit these paths)
# ─────────────────────────────────────────────────────────────────────────────

base_dir = "/Users/niaggar/Documents/Thesis/Progress/02Mar26"

# Glob picks up the timestamped folder automatically:
dirs_A = sorted(glob.glob(f"{base_dir}/*speckle_sweep_A_nphotons*/"))
dirs_B = sorted(glob.glob(f"{base_dir}/*speckle_sweep_B_anisotropy*/"))

sweep_A_dir = dirs_A[-1] if dirs_A else None   # use most recent
sweep_B_dir = dirs_B[-1] if dirs_B else None

# ─────────────────────────────────────────────────────────────────────────────
# Load scalar metrics
# ─────────────────────────────────────────────────────────────────────────────

res_A = load_quality(sweep_A_dir) if sweep_A_dir else None
res_B = load_quality(sweep_B_dir) if sweep_B_dir else None


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Sweep A: runtime and quality vs n_photons
# ─────────────────────────────────────────────────────────────────────────────

if res_A is not None and len(res_A["n_photons"]) > 0:
    n_ph = res_A["n_photons"]
    sort = np.argsort(n_ph)
    n_ph = n_ph[sort]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Sweep A — quality & runtime vs n_photons", fontsize=13)

    # (1) Runtime
    ax = axes[0]
    ax.loglog(n_ph, res_A["runtime_s"][sort], 'o-', color='steelblue')
    # Reference lines: O(N) and O(N^2)
    for exp_pow, ls, lbl in [(1, '--', r'$\propto N$'), (1.5, ':', r'$\propto N^{1.5}$')]:
        x0 = n_ph[0]; y0 = res_A["runtime_s"][sort][0]
        ax.loglog(n_ph, y0 * (n_ph/x0)**exp_pow, ls, color='gray', label=lbl)
    ax.set_xlabel("n_photons"); ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Runtime"); ax.legend(); ax.grid(True, which='both', ls='--', alpha=0.3)

    # (2) Speckle contrast  (target = 1)
    ax = axes[1]
    ax.semilogx(n_ph, res_A["contrast_x"][sort], 'o-', color='purple', label=r'$C_x$')
    ax.semilogx(n_ph, res_A["contrast_y"][sort], 's--', color='orchid',  label=r'$C_y$')
    ax.axhline(1.0, color='k', lw=1.5, ls=':', label='theory')
    ax.set_xlabel("n_photons"); ax.set_ylabel(r"Speckle contrast $C = \sigma(I)/\langle I \rangle$")
    ax.set_title("Speckle Contrast"); ax.legend(); ax.grid(True, which='both', ls='--', alpha=0.3)

    # (3) KL divergence from exponential (target = 0)
    ax = axes[2]
    ax.semilogx(n_ph, res_A["kl_x"][sort], 'o-', color='crimson', label=r'$D_{KL}^x$')
    ax.semilogx(n_ph, res_A["kl_y"][sort], 's--', color='salmon',  label=r'$D_{KL}^y$')
    ax.axhline(0.0, color='k', lw=1.5, ls=':')
    ax.set_xlabel("n_photons"); ax.set_ylabel(r"$D_{KL}(p_{sim}\,\|\,e^{-\eta})$")
    ax.set_title("KL Divergence from Theory"); ax.legend(); ax.grid(True, which='both', ls='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{base_dir}/sweep_A_nphotons.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Sweep B: runtime and quality vs anisotropy g
# ─────────────────────────────────────────────────────────────────────────────

if res_B is not None and len(res_B["anisotropy_g"]) > 0:
    g    = res_B["anisotropy_g"]
    sort = np.argsort(g)
    g    = g[sort]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Sweep B — quality & runtime vs anisotropy $g$", fontsize=13)

    # (1) Runtime vs g
    ax = axes[0]
    ax.plot(g, res_B["runtime_s"][sort], 'o-', color='darkorange')
    ax.set_xlabel(r"Anisotropy $g$"); ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Runtime"); ax.grid(True, ls='--', alpha=0.3)
    # Annotate with radius labels
    for gi, ri, ti in zip(g, res_B["radius_real"][sort], res_B["runtime_s"][sort]):
        ax.annotate(f"r={ri:.2f}", (gi, ti), textcoords="offset points",
                    xytext=(4, 4), fontsize=7)

    # (2) Speckle contrast vs g
    ax = axes[1]
    ax.plot(g, res_B["contrast_x"][sort], 'o-', color='purple', label=r'$C_x$')
    ax.plot(g, res_B["contrast_y"][sort], 's--', color='orchid',  label=r'$C_y$')
    ax.axhline(1.0, color='k', lw=1.5, ls=':', label='theory')
    ax.set_xlabel(r"Anisotropy $g$"); ax.set_ylabel(r"Speckle contrast $C$")
    ax.set_title("Speckle Contrast"); ax.legend(); ax.grid(True, ls='--', alpha=0.3)

    # (3) KL divergence vs g
    ax = axes[2]
    ax.plot(g, res_B["kl_x"][sort], 'o-', color='crimson', label=r'$D_{KL}^x$')
    ax.plot(g, res_B["kl_y"][sort], 's--', color='salmon',  label=r'$D_{KL}^y$')
    ax.axhline(0.0, color='k', lw=1.5, ls=':')
    ax.set_xlabel(r"Anisotropy $g$"); ax.set_ylabel(r"$D_{KL}$")
    ax.set_title("KL Divergence from Theory"); ax.legend(); ax.grid(True, ls='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{base_dir}/sweep_B_anisotropy.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — 2D tradeoff map: quality-per-second
#            Only available if both sweeps share a common run
# ─────────────────────────────────────────────────────────────────────────────
# (Optional: build a combined grid by re-running at a small set of
#  (n_photons, g) pairs and plotting a heatmap. See comments below.)

# The idea:
#   quality_per_second = (1 - kl_x) / runtime_s    (higher is better)
#   This tells you what combination of photons and scattering regime
#   gives you the best result for the least compute cost.


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Detailed η distribution for selected runs
# ─────────────────────────────────────────────────────────────────────────────

def plot_eta_distribution(sweep_dir, run_index=0, label=""):
    """
    Load one run from a sweep and reproduce the η-histogram comparison
    (panel (c) of your existing plot_figure_5_corrected).
    """
    run_dirs = sorted(glob.glob(f"{sweep_dir}/runs/*/"))
    if run_index >= len(run_dirs):
        return
    data = ResultsLoader(run_dirs[run_index])
    meta = data.sensor_meta("planarfluence")
    S0_t = data.sensor_data("planarfluence", "S0_t")[0]
    S1_t = data.sensor_data("planarfluence", "S1_t")[0]
    Ex   = data.sensor_data("planarfield",   "Ex")

    I_inst_x = np.abs(Ex)**2
    I_avg_x  = (S0_t + S1_t) / 2.0
    mask = I_avg_x > 0.01 * np.max(I_avg_x)
    eta  = I_inst_x[mask] / I_avg_x[mask]

    fig, ax = plt.subplots(figsize=(5, 4))
    hist, edges = np.histogram(eta, bins=80, range=(0, 8), density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.semilogy(centers, hist, 'mo', mfc='none', ms=4, label='Simulation')
    ax.semilogy(np.linspace(0, 8, 200), np.exp(-np.linspace(0, 8, 200)),
                'k-', lw=2, label=r'Theory $e^{-\eta}$')
    kl = kl_from_exponential(I_inst_x, I_avg_x)
    C  = float(np.std(I_inst_x[mask]) / np.mean(I_avg_x[mask]))
    ax.set_title(f"{label}\n$C={C:.3f}$, $D_{{KL}}={kl:.4f}$")
    ax.set_xlabel(r"$\eta = I_x / \langle I_x \rangle$")
    ax.set_ylabel("Probability density")
    ax.set_xlim(0, 8); ax.set_ylim(1e-4, 2)
    ax.legend(); ax.grid(True, which='both', ls='--', alpha=0.2)
    plt.tight_layout()
    plt.show()


# Example usage: plot the η distribution for the smallest and largest n_photons
if sweep_A_dir:
    plot_eta_distribution(sweep_A_dir, run_index=0,  label="Sweep A — smallest n_photons")
    plot_eta_distribution(sweep_A_dir, run_index=-1, label="Sweep A — largest n_photons")

if sweep_B_dir:
    plot_eta_distribution(sweep_B_dir, run_index=0,  label="Sweep B — isotropic (small r)")
    plot_eta_distribution(sweep_B_dir, run_index=-1, label="Sweep B — forward-scattering (large r)")