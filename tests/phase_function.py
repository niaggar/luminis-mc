# %%
from luminis_mc import (
    Rng,
    UniformPhaseFunction,
    RayleighPhaseFunction,
    HenyeyGreensteinPhaseFunction,
    RayleighDebyePhaseFunction,
    DrainePhaseFunction,
    RayleighDebyeEMCPhaseFunction,
)
from luminis_mc import set_log_level
from luminis_mc import LogLevel
import numpy as np
import matplotlib.pyplot as plt


set_log_level(LogLevel.debug)

cosMin = -1.0
cosMax = 1.0

thetaMin = 0.00001
thetaMax = np.pi

nDiv = 10000
nSamples = 1000000

anysotropy = -0.4
radius = np.array([0.1, 0.5, 0.7, 1.0, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 9, 15, 20])
wavelength = 1

# %%

# rayleigh_debye = RayleighDebyePhaseFunction(wavelength, radius, nDiv, thetaMin, thetaMax)
# rayleigh_debye_emc = RayleighDebyeEMCPhaseFunction(wavelength, radius, nDiv, thetaMin, thetaMax)

# %%

rng = np.random.default_rng()
mus = rng.random(nSamples)

# for r in radius:
#     rayleigh_debye = RayleighDebyePhaseFunction(wavelength, r, nDiv, thetaMin, thetaMax)
#     rayleigh_debye_emc = RayleighDebyeEMCPhaseFunction(wavelength, r, nDiv, thetaMin, thetaMax)

#     theta = np.array([rayleigh_debye.sample_theta(x) for x in mus])
#     theta_emc = np.array([rayleigh_debye_emc.sample_theta(x) for x in mus])

#     plt.hist(theta, bins=100, density=True, alpha=0.5, label=f'RD', log=True)
#     plt.hist(theta_emc, bins=100, density=True, alpha=0.5, label=f'RD Polarized', log=True)
#     plt.xlabel('θ')
#     plt.ylabel('Probability Density')
#     plt.title(f'Phase Function Sampling (a={r})')
#     plt.legend()
#     plt.grid()
#     plt.show()










def kl_divergence_from_samples(theta_P, theta_Q, nbins=720, tmin=1e-6, tmax=np.pi):
    """
    Compute D_KL(P||Q) from samples of theta for two phase functions.
    Histograms are built on the same bins over [tmin, tmax].
    """
    bins = np.linspace(tmin, tmax, nbins + 1)
    p, _ = np.histogram(theta_P, bins=bins, density=True)
    q, _ = np.histogram(theta_Q, bins=bins, density=True)

    # convert density -> discrete probs over bins (density * bin_width) and renormalize
    bw = np.diff(bins)
    p = p * bw
    q = q * bw
    p = p / p.sum()
    q = q / q.sum()

    # epsilon to avoid log(0) / division by zero
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)

    return float(np.sum(p * np.log(p / q)))

def js_divergence_from_samples(theta_P, theta_Q, nbins=720, tmin=1e-6, tmax=np.pi):
    """Optional: symmetric Jensen–Shannon divergence for reference."""
    bins = np.linspace(tmin, tmax, nbins + 1)
    bw = np.diff(bins)
    p, _ = np.histogram(theta_P, bins=bins, density=True); p = p * bw; p /= p.sum()
    q, _ = np.histogram(theta_Q, bins=bins, density=True); q = q * bw; q /= q.sum()
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))

# Reusa los mismos números uniformes para reducir la varianza
rng = np.random.default_rng(12345)
mus = rng.random(nSamples)

results = []
for r in radius:
    rd = RayleighDebyePhaseFunction(wavelength, r, nDiv, thetaMin, thetaMax)
    rdemc = RayleighDebyeEMCPhaseFunction(wavelength, r, nDiv, thetaMin, thetaMax)

    theta_rd    = np.array([rd.sample_theta(x)    for x in mus])
    theta_rdemc = np.array([rdemc.sample_theta(x) for x in mus])

    dkl = kl_divergence_from_samples(theta_rd, theta_rdemc, nbins=720,
                                     tmin=thetaMin, tmax=thetaMax)
    # opcional: también el inverso y el JS
    dkl_rev = kl_divergence_from_samples(theta_rdemc, theta_rd, nbins=720,
                                         tmin=thetaMin, tmax=thetaMax)
    djs = js_divergence_from_samples(theta_rd, theta_rdemc, nbins=720,
                                     tmin=thetaMin, tmax=thetaMax)

    results.append((r, dkl, dkl_rev, djs))
    print(f"a={r:>4.1f}:  D_KL(RD || RD-EMC) = {dkl:.6e}   "
          f"D_KL(RD-EMC || RD) = {dkl_rev:.6e}   JS = {djs:.6e}")


import matplotlib.pyplot as plt

a_vals, dkl, dkl_rev, djs = np.array(results).T
plt.semilogy(a_vals, dkl, 'o-', label=r'D$_{KL}$(RD‖RD–P)')
plt.semilogy(a_vals, dkl_rev, 's--', label=r'D$_{KL}$(RD–P‖RD)')
# plt.semilogy(a_vals, djs, 'd-.', label='Jensen–Shannon')
plt.xlabel('Parameter x')
plt.ylabel('Divergence')
# plt.title('Divergence between Rayleigh–Debye and RD–P phase functions')
plt.legend()
plt.grid(True, which='both', ls=':')
plt.show()
