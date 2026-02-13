from luminis_mc import (
    Laser,
    SimpleMedium,
    MieMedium,
    MultiDetector,
    HistogramDetector,
    SpatialDetector,
    SpatialTimeDetector,
    AbsorptionTimeDependent,
    SimConfig,
    RayleighDebyeEMCPhaseFunction,
    MiePhaseFunction,
    CVec2,
    Vec3,
    Rng,
)
from luminis_mc import LogLevel, LaserSource
from luminis_mc import run_simulation_parallel, set_log_level
import numpy as np
import matplotlib.pyplot as plt
import time

set_log_level(LogLevel.info)


start_time = time.time()

# Global frame of reference
m_global = Vec3(1, 0, 0)
n_global = Vec3(0, 1, 0)
s_global = Vec3(0, 0, 1)
light_speed = 1

# Medium parameters in micrometers
mean_free_path_real = 2.8
radius_real = 0.8
wavelength_real = 0.6328
n_particle_real = 1.34
n_medium_real = 1.33


mean_free_path_sim = 1.0
inv_mfp_sim = 1 / mean_free_path_sim
mu_absortion_sim = 0.05 * inv_mfp_sim
mu_scattering_sim = inv_mfp_sim - mu_absortion_sim



size_parameter = 2 * np.pi * radius_real * n_medium_real / wavelength_real
condition_1 = np.abs(n_particle_real / n_medium_real - 1)
condition_2 = size_parameter * condition_1


# Time parameters
t_ref = mean_free_path_sim / light_speed
dt = 0.5 * t_ref
max_time = 10 * t_ref

# Phase function parameters
thetaMin = 0.0
thetaMax = np.pi
nDiv = 1000

# Laser parameters
origin = Vec3(0, 0, 0)

polarization = CVec2(1/np.sqrt(2), -1j/np.sqrt(2))  # Circular Right polarization
# polarization = CVec2(1/np.sqrt(2), 1j/np.sqrt(2))   # Circular Left polarization
laser_radius = 0.5 * mean_free_path_sim
laser_type = LaserSource.Gaussian


rng_test = Rng()
laser_source = Laser(origin, polarization, wavelength_real, laser_radius, laser_type)
phase_function = RayleighDebyeEMCPhaseFunction(wavelength_real, radius_real, n_particle_real, n_medium_real, nDiv, thetaMin, thetaMax)
mie_phase_function = MiePhaseFunction(wavelength_real, radius_real, n_particle_real, n_medium_real, nDiv, thetaMin, thetaMax)
medium = SimpleMedium(mu_absortion_sim, mu_scattering_sim, phase_function, mean_free_path_sim, radius_real, n_particle_real, n_medium_real)
mie_medium = MieMedium(mu_absortion_sim, mu_scattering_sim, mie_phase_function, mean_free_path_sim, radius_real, n_particle_real, n_medium_real)

anysotropy = phase_function.get_anisotropy_factor(rng_test)
mie_anysotropy = mie_phase_function.get_anisotropy_factor(rng_test)





print("CONDITIONS FOR RAYLEIGH-DEBYE APPROXIMATION:")
print(f"Size parameter: {size_parameter}")
print(f"|n_particle / n_medium - 1|: {condition_1}")
print(f"Size parameter * |n_particle / n_medium - 1|: {condition_2}")

print("MEDIUM PARAMETERS:")
print(f"Mean free path: {mean_free_path_sim}")
print(f"Transport mean free path: {mean_free_path_sim / (1 - anysotropy[0])}")
print(f"Particle radius: {radius_real}")
print(f"Wavelength: {wavelength_real}")
print(f"Scattering coefficient: {mu_scattering_sim}")
print(f"Absorption coefficient: {mu_absortion_sim}")
print(f"Albedo: {mu_scattering_sim / (mu_scattering_sim + mu_absortion_sim)}")
print(f"Refractive index particle: {n_particle_real}")
print(f"Refractive index medium: {n_medium_real}")
print(f"Relative refractive index: {n_particle_real / n_medium_real}")
print(f"Anisotropy: {anysotropy}")
print(f"Mie Anisotropy: {mie_anysotropy}")


k = 2 * np.pi / wavelength_real
theta_test = np.linspace(thetaMin, thetaMax, 1000)
phase_function_values_raw = np.array([phase_function.pdf(theta) for theta in theta_test])
phase_function_values = phase_function_values_raw / np.trapezoid(phase_function_values_raw, theta_test)

time_matrix = time.time()
scattering_matrices = np.array([medium.scattering_matrix(theta, 0, k) for theta in theta_test])
d_matrix = time.time() - time_matrix
print(f"Tiempo de cálculo de las matrices de dispersión: {d_matrix:.4f} segundos")

s2_abs_values = np.abs(scattering_matrices[:, 0, 0])
s1_abs_values = np.abs(scattering_matrices[:, 1, 1])

# test_matrix = np.array(medium.scattering_matrix(0.00001, 0, k))
# print(test_matrix, "Primera matriz de dispersión")
# print(np.abs(test_matrix[0,0]), "S2 elemento de la primera matriz")
# print(test_matrix[1,1], "S1 elemento de la primera matriz")


mus = np.random.default_rng().random(5_000_000)
theta_emc = np.array([phase_function.sample_theta(x) for x in mus])

# Plotting the phase function
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.semilogy(theta_test, phase_function_values, label='Phase Function')
count, bins, ignored = plt.hist(theta_emc, bins=100, density=True, color='lightgray', label='Muestreo MC')
plt.xlabel('Scattering Angle (radians)')
plt.ylabel('Phase Function Value')
plt.title('Rayleigh-Debye Phase Function')
plt.legend()
plt.grid()

# Plotting the scattering matrix elements
plt.subplot(1, 2, 2)
plt.plot(theta_test, s2_abs_values, label='|S2|')
plt.plot(theta_test, s1_abs_values, label='|S1|')
plt.xlabel('Scattering Angle (radians)')
plt.ylabel('Scattering Matrix Element Magnitude')
plt.title('Scattering Matrix Elements')
plt.legend()
plt.grid()
plt.show()


k = 2 * np.pi / wavelength_real
theta_test = np.linspace(thetaMin, thetaMax, 1000)
phase_function_values_raw = np.array([mie_phase_function.pdf(theta) for theta in theta_test])
phase_function_values = phase_function_values_raw / np.trapezoid(phase_function_values_raw, theta_test)
time_matrix = time.time()
scattering_matrices = np.array([mie_medium.scattering_matrix(theta, 0, k) for theta in theta_test])
d_matrix = time.time() - time_matrix
print(f"Tiempo de cálculo de las matrices de dispersión (Mie): {d_matrix:.4f} segundos")

s2_abs_values = np.abs(scattering_matrices[:, 0, 0])
s1_abs_values = np.abs(scattering_matrices[:, 1, 1])

mus = np.random.default_rng().random(5_000_000)
theta_emc = np.array([mie_phase_function.sample_theta(x) for x in mus])

# Plotting the phase function
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.semilogy(theta_test, phase_function_values, label='Phase Function')
count, bins, ignored = plt.hist(theta_emc, bins=100, density=True, color='lightgray', label='Muestreo MC')
plt.xlabel('Scattering Angle (radians)')
plt.ylabel('Phase Function Value')
plt.title('Rayleigh-Debye Phase Function')
plt.legend()
plt.grid()

# Plotting the scattering matrix elements
plt.subplot(1, 2, 2)
plt.plot(theta_test, s2_abs_values, label='|S2|')
plt.plot(theta_test, s1_abs_values, label='|S1|')
plt.xlabel('Scattering Angle (radians)')
plt.ylabel('Scattering Matrix Element Magnitude')
plt.title('Scattering Matrix Elements')
plt.legend()
plt.grid()
plt.show()
