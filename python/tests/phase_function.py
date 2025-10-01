# %%
from luminis_mc import (
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

anysotropy = 0.3
radius = 0.1
wavelength = 0.5

# %%

uniform = UniformPhaseFunction()
rayleigh = RayleighPhaseFunction(nDiv, cosMin, cosMax)
henyey_greenstein = HenyeyGreensteinPhaseFunction(anysotropy)
rayleigh_debye = RayleighDebyePhaseFunction(
    wavelength, radius, nDiv, thetaMin, thetaMax
)
rayleigh_debye_emc = RayleighDebyeEMCPhaseFunction(
    wavelength, radius, nDiv, thetaMin, thetaMax
)
draine = DrainePhaseFunction(0.2, 1, nDiv, cosMin, cosMax)

# %%

data = {
    "uniform": np.array([]),
    "rayleigh": np.array([]),
    "henyey_greenstein": np.array([]),
    "rayleigh_debye": np.array([]),
    "rayleigh_debye_emc": np.array([]),
    "draine": np.array([]),
}

for i in range(nSamples):
    mu = np.random.uniform(0, 1)

    xUniform = uniform.sample(mu)
    xRayleigh = rayleigh.sample(mu)
    xHenyeyGreenstein = henyey_greenstein.sample(mu)
    xRayleighDebye = rayleigh_debye.sample(mu)
    xRayleighDebyeEMC = rayleigh_debye_emc.sample(mu)
    xDraine = draine.sample(mu)

    data["uniform"] = np.append(data["uniform"], xUniform)
    data["rayleigh"] = np.append(data["rayleigh"], xRayleigh)
    data["henyey_greenstein"] = np.append(data["henyey_greenstein"], xHenyeyGreenstein)
    data["rayleigh_debye"] = np.append(data["rayleigh_debye"], xRayleighDebye)
    data["rayleigh_debye_emc"] = np.append(
        data["rayleigh_debye_emc"], xRayleighDebyeEMC
    )
    data["draine"] = np.append(data["draine"], xDraine)

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

# Remove the empty subplot
axes[5].remove()

plt.tight_layout()
plt.show()

# %%


sample_phi =
