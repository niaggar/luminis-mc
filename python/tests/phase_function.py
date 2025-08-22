# %%
from luminis_mc import UniformPhaseFunction, RayleighPhaseFunction, HenyeyGreensteinPhaseFunction, RayleighDebyePhaseFunction
from luminis_mc import set_log_level
from luminis_mc import LogLevel
import numpy as np

set_log_level(LogLevel.debug)

cosMin = -1.0
cosMax = 1.0
thetaMin = 0.00001
thetaMax = np.arccos(cosMin)
nDiv = 1000
nSamples = 100000

anysotropy = 0.4
radius = 0.1
wavelength = 0.5

print(f"cosMin: {cosMin}, cosMax: {cosMax}, thetaMin: {thetaMin}, thetaMax: {thetaMax}")
print(f"nDiv: {nDiv}, anysotropy: {anysotropy}, radius: {radius}, wavelength: {wavelength}")

# %%

uniform = UniformPhaseFunction()
rayleigh = RayleighPhaseFunction(nDiv, cosMin, cosMax)
henyey_greenstein = HenyeyGreensteinPhaseFunction(anysotropy)
# rayleigh_debye = RayleighDebyePhaseFunction(wavelength, radius, nDiv, thetaMin, thetaMax)

# %%

data = {
    "uniform": np.array([]),
    "rayleigh": np.array([]),
    "henyey_greenstein": np.array([]),
    # "rayleigh_debye": np.array([]),
}

for i in range(nSamples):
    mu = np.random.uniform(0, 1)

    xUniform = uniform.sample(mu)
    xRayleigh = rayleigh.sample(mu)
    xHenyeyGreenstein = henyey_greenstein.sample(mu)
    # xRayleighDebye = rayleigh_debye.sample(mu)

    data["uniform"] = np.append(data["uniform"], xUniform)
    data["rayleigh"] = np.append(data["rayleigh"], xRayleigh)
    data["henyey_greenstein"] = np.append(data["henyey_greenstein"], xHenyeyGreenstein)
    # data["rayleigh_debye"] = np.append(data["rayleigh_debye"], xRayleighDebye)

# %%

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(data["uniform"], bins=100, alpha=0.3, label="Uniform Phase Function", density=True)
plt.hist(data["rayleigh"], bins=100, alpha=0.3, label="Rayleigh Phase Function", density=True)
plt.hist(data["henyey_greenstein"], bins=100, alpha=0.3, label="Henyey-Greenstein Phase Function", density=True)
# plt.hist(data["rayleigh_debye"], bins=50, alpha=0.5, label="Rayleigh-Debye Phase Function", density=True)
plt.title("Phase Function Samples")
plt.xlabel("Sample Value")
plt.ylabel("Density")
plt.legend()
plt.grid()
plt.show()
