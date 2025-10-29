from luminis_mc import (
    Rng,
    RayleighDebyePhaseFunction,
    RayleighDebyeEMCPhaseFunction,
)
from luminis_mc import set_log_level
from luminis_mc import LogLevel
import numpy as np
import matplotlib.pyplot as plt

set_log_level(LogLevel.debug)


thetaMin = 0.000001
thetaMax = np.pi
nDiv = 1000
nSamples = 100000

wavelength = 1
min_radius = 0.1

def scattering_parameter(radius, wavelength):
    return (2 * np.pi * radius) / wavelength

k = 2*np.pi/wavelength

x_list_0_to_5 = np.linspace(0.01, 5, 100)
x_list_5_to_20 = np.linspace(5, 20, 10)
x_list = np.concatenate((x_list_0_to_5, x_list_5_to_20))
r_list = [x/k for x in x_list]

scattering_parameters = [scattering_parameter(r, wavelength) for r in r_list]

rng = Rng()

anysotropy = []
anysotropy_emc = []

for radius in r_list:
    distribution_rd = RayleighDebyePhaseFunction(wavelength, radius, nDiv, thetaMin, thetaMax)
    distribution_rd_emc = RayleighDebyeEMCPhaseFunction(wavelength, radius, nDiv, thetaMin, thetaMax)

    g_rd = distribution_rd.get_anisotropy_factor(rng)
    g_rd_emc = distribution_rd_emc.get_anisotropy_factor(rng)
    anysotropy.append(g_rd[0])
    anysotropy_emc.append(g_rd_emc[0])


plt.figure(figsize=(10, 6))
plt.plot(scattering_parameters, anysotropy, label='Rayleigh-Debye')
plt.plot(scattering_parameters, anysotropy_emc, label='Rayleigh-Debye Polarized')
plt.xlabel('Scattering Parameter (x)')
plt.ylabel('Anisotropy Factor (g)')
plt.title('Anisotropy Factor vs Scattering Parameter')
plt.legend()
plt.grid()
plt.show()
