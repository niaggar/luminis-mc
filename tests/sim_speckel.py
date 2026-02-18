from hashlib import shake_128
from luminis_mc import (
    Laser,
    SimpleMedium,
    MieMedium,
    PlanarFieldSensor,
    PlanarFluenceSensor,
    SensorsGroup,
    SimConfig,
    RayleighDebyeEMCPhaseFunction,
    MiePhaseFunction,
    CVec2,
    Vec3,
    Rng,
)
from luminis_mc import LogLevel, LaserSource
from luminis_mc import run_simulation_parallel, set_log_level
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

set_log_level(LogLevel.info)


start_time = time.time()

# Global frame of reference
light_speed = 1

# Medium parameters in micrometers
mean_free_path_real = 2.8
radius_real = 0.8
wavelength_real = 0.6328
n_particle_real = 1.59
n_medium_real = 1.33

mean_free_path_sim = 1.0
inv_mfp_sim = 1 / mean_free_path_sim
mu_absortion_sim = 0.05 * inv_mfp_sim
mu_scattering_sim = inv_mfp_sim - mu_absortion_sim

size_parameter = 2 * np.pi * radius_real * n_medium_real / wavelength_real
condition_1 = np.abs(n_particle_real / n_medium_real - 1)
condition_2 = size_parameter * condition_1

# Phase function parameters
thetaMin = 0.0
thetaMax = np.pi
nDiv = 1000
n_photons = 5_000

# Laser parameters
origin = Vec3(0, 0, 0)
# polarization = CVec2(1, 0)  # Linear Vertical polarization
# polarization = CVec2(0, 1)  # Linear Horizontal polarization
# polarization = CVec2(1/np.sqrt(2), 1/np.sqrt(2))  # Linear 45 degrees polarization
# polarization = CVec2(1/np.sqrt(2), -1/np.sqrt(2))  # Linear -45 degrees polarization
polarization = CVec2(1/np.sqrt(2), -1j/np.sqrt(2))  # Circular Right polarization
# polarization = CVec2(1/np.sqrt(2), 1j/np.sqrt(2))   # Circular Left polarization
laser_radius = 5 * mean_free_path_sim
laser_type = LaserSource.Gaussian

rng_test = Rng()
laser_source = Laser(origin, polarization, wavelength_real, laser_radius, laser_type)
phase_function = RayleighDebyeEMCPhaseFunction(wavelength_real, radius_real, n_particle_real, n_medium_real, nDiv, thetaMin, thetaMax)
mie_phase_function = MiePhaseFunction(wavelength_real, radius_real, n_particle_real, n_medium_real, nDiv, thetaMin, thetaMax)
medium = SimpleMedium(mu_absortion_sim, mu_scattering_sim, phase_function, mean_free_path_sim, radius_real, n_particle_real, n_medium_real)
mie_medium = MieMedium(mu_absortion_sim, mu_scattering_sim, mie_phase_function, mean_free_path_sim, radius_real, n_particle_real, n_medium_real, wavelength_real)

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


z_sensor = 0.0
len_x_sensor = 40 * mean_free_path_sim
len_y_sensor = 40 * mean_free_path_sim
pixel_size = 0.1 * mean_free_path_sim
len_t_sensor = 0.0
pixel_t_size = 0.0

sens_group = SensorsGroup()
planar_field_sensor = sens_group.add_detector(PlanarFieldSensor(z_sensor, len_x_sensor, len_y_sensor, pixel_size, pixel_size))
planar_fluence_sensor = sens_group.add_detector(PlanarFluenceSensor(z_sensor, len_x_sensor, len_y_sensor, len_t_sensor, pixel_size, pixel_size, pixel_t_size))

config = SimConfig(
    n_photons=n_photons,
    medium=medium,
    detector=sens_group,
    laser=laser_source,
    track_reverse_paths=False,
)
config.n_threads = 8

run_simulation_parallel(config)

end_time = time.time()
print(f"---- Simulation time: {end_time - start_time:.2f} seconds")
print(f"Photons detected by planar field sensor: {planar_field_sensor.hits}")
print(f"Photons detected by planar fluence sensor: {planar_fluence_sensor.hits}")

S_0 = np.array(planar_fluence_sensor.S0_t[0], copy=False)
S_1 = np.array(planar_fluence_sensor.S1_t[0], copy=False)
S_2 = np.array(planar_fluence_sensor.S2_t[0], copy=False)
S_3 = np.array(planar_fluence_sensor.S3_t[0], copy=False)


E_x = np.array(planar_field_sensor.Ex, copy=False)
E_y = np.array(planar_field_sensor.Ey, copy=False)

I_ave_x = (S_0 + S_1) / 2
I_insta_x = np.abs(E_x)**2

I_ave_y = (S_0 - S_1) / 2
I_insta_y = np.abs(E_y)**2

def gaussian_2d_func(coords, amplitude, xo, yo, sigma, offset):
    x, y = coords
    g = offset + amplitude * np.exp( -((x-xo)**2 + (y-yo)**2) / (2*sigma**2) )
    return g.ravel()

def plot_figure_5_corrected(I_inst, I_avg_raw, NN, HW):
    # Crear grid espacial
    x = np.linspace(-HW, HW, NN)
    y = np.linspace(-HW, HW, NN)
    X, Y = np.meshgrid(x, y)

    # ... (El bloque de ajuste Gaussiano se queda igual) ...
    # Copia el bloque "try/except" y cálculo de "eta" de tu código anterior aquí
    # o usa el script completo de abajo.

    # --- REPETIMOS EL AJUSTE PARA QUE EL CÓDIGO ESTÉ COMPLETO ---
    x_flat = X.ravel()
    y_flat = Y.ravel()
    z_flat = I_avg_raw.ravel()
    initial_guess = [np.max(I_avg_raw), 0, 0, 10, 0]

    try:
        popt, pcov = curve_fit(gaussian_2d_func, (x_flat, y_flat), z_flat, p0=initial_guess)
        I_avg_smooth = gaussian_2d_func((X, Y), *popt).reshape(NN, NN)
    except:
        I_avg_smooth = I_avg_raw

    mask = I_avg_smooth > (np.max(I_avg_smooth) * 0.01)
    eta = I_inst[mask] / I_avg_smooth[mask]
    # -----------------------------------------------------------

    # --- GRAFICACIÓN ---
    fig = plt.figure(figsize=(16, 10))

    # (a) Instantaneous Ix
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(I_inst, extent=[-HW, HW, -HW, HW], cmap='gray', origin='lower')
    # CORRECCIÓN 1: Añadir r antes de las comillas si hay LaTeX (opcional aquí, pero buena práctica)
    ax1.set_title(r"(a) Instantaneous Intensity $I_x$")
    ax1.set_xlabel(r"x ($l_s$)")

    # (b) Average Ix
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(I_avg_raw, extent=[-HW, HW, -HW, HW], cmap='gray', origin='lower')
    ax2.set_title(r"(b) Average Intensity $\langle I_x \rangle$ (Raw)")
    ax2.set_xlabel(r"x ($l_s$)")

    # (c) Probability Density
    ax3 = fig.add_subplot(2, 3, 3)
    hist, bins = np.histogram(eta, bins=60, density=True)
    centers = (bins[:-1] + bins[1:]) / 2

    ax3.semilogy(centers, hist, 'mo', mfc='none', label='Simulación')

    x_theory = np.linspace(0, 10, 100)
    # CORRECCIÓN 2: Añadir r'' aquí por el \eta
    ax3.semilogy(x_theory, np.exp(-x_theory), 'k-', linewidth=2, label=r'Teoría $e^{-\eta}$')

    ax3.set_title("(c) Probability Density")
    # CORRECCIÓN 3: Añadir r'' aquí (ESTA ES LA QUE CAUSABA EL CRASH PRINCIPAL)
    ax3.set_xlabel(r"$\eta = I_x / \langle I_x \rangle$")

    ax3.set_ylim(1e-4, 1)
    ax3.set_xlim(0, 10)
    ax3.legend()
    ax3.grid(True, which="both", ls="--", alpha=0.2)

    # (d) 3D Surface
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    surf = ax4.plot_surface(X, Y, I_avg_smooth, cmap='jet', linewidth=0, antialiased=False)
    ax4.set_title("(d) Avg Intensity (Fitted Profile)")

    # (e) Cross Section X
    ax5 = fig.add_subplot(2, 3, 5)
    mid_idx = NN // 2
    ax5.plot(x, I_avg_raw[mid_idx, :], 'm-', alpha=0.4, label='Data (Raw)')
    ax5.plot(x, I_avg_smooth[mid_idx, :], 'k-', linewidth=2, label='Fit')
    ax5.set_title("(e) Cross Section X")
    ax5.legend()

    # (f) Cross Section Y
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(y, I_avg_raw[:, mid_idx], 'm-', alpha=0.4, label='Data (Raw)')
    ax6.plot(y, I_avg_smooth[:, mid_idx], 'k-', linewidth=2, label='Fit')
    ax6.set_title("(f) Cross Section Y")

    plt.tight_layout()
    plt.savefig("replicate_fig5_corrected.png", dpi=150)
    plt.show()


NN = int(len_x_sensor / pixel_size)
HW = len_x_sensor / 2
plot_figure_5_corrected(I_insta_x, I_ave_x, NN, HW)
plot_figure_5_corrected(I_insta_y, I_ave_y, NN, HW)



