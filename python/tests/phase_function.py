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

nDiv = 1000
nSamples = 100000

anysotropy = -0.4
radius = 0.1
wavelength = 0.5

# %%

uniform = UniformPhaseFunction()
rayleigh = RayleighPhaseFunction(nDiv, cosMin, cosMax)
henyey_greenstein = HenyeyGreensteinPhaseFunction(anysotropy)
rayleigh_debye = RayleighDebyePhaseFunction(wavelength, radius, nDiv, thetaMin, thetaMax)
rayleigh_debye_emc = RayleighDebyeEMCPhaseFunction(wavelength, radius, nDiv, thetaMin, thetaMax)
draine = DrainePhaseFunction(0.2, 1, nDiv, cosMin, cosMax)

# %%

rng = np.random.default_rng()
mus = rng.random(nSamples)
data = {
    "uniform": np.array([uniform.sample(mu) for mu in mus]),
    "rayleigh": np.array([rayleigh.sample(mu) for mu in mus]),
    "henyey_greenstein": np.array([henyey_greenstein.sample(mu) for mu in mus]),
    "rayleigh_debye": np.array([rayleigh_debye.sample(mu) for mu in mus]),
    "rayleigh_debye_emc": np.array([rayleigh_debye_emc.sample(mu) for mu in mus]),
    "draine": np.array([draine.sample(mu) for mu in mus]),
}

# %%

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

phase_functions = [
    "uniform",
    "rayleigh",
    "henyey_greenstein",
    "rayleigh_debye",
    "rayleigh_debye_emc",
    "draine",
]

for i, phase_func in enumerate(phase_functions):
    axes[i].hist(data[phase_func], bins=50, alpha=0.7, density=True)
    axes[i].set_title(f"{phase_func.replace('_', ' ').title()} Phase Function")
    axes[i].set_xlabel("Sampled Value")
    axes[i].set_ylabel("Density")
    axes[i].grid(True, alpha=0.3)

axes[5].remove()
plt.tight_layout()
plt.show()

# %%

rng = Rng()
g_u_c = uniform.get_anisotropy_factor(rng)
g_r_c = rayleigh.get_anisotropy_factor(rng)
g_hg_c = henyey_greenstein.get_anisotropy_factor(rng)
g_rd_c = rayleigh_debye.get_anisotropy_factor(rng)
g_rde_c = rayleigh_debye_emc.get_anisotropy_factor(rng)
g_dr_c = draine.get_anisotropy_factor(rng)
print(f"Uniform (C++): g={g_u_c}")
print(f"Rayleigh (C++): g={g_r_c}", f"(esperado ≈ 0)")
print(f"HG (C++): g={g_hg_c}", f"(esperado ≈ {anysotropy})")
print(f"Rayleigh-Debye (C++): g={g_rd_c}")
print(f"Rayleigh-Debye EMC (C++): g={g_rde_c}")
print(f"Draine (C++): g={g_dr_c}")
