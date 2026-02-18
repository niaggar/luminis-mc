from luminis_mc import (
    Laser,
    MieMedium,
    PlanarFieldSensor,
    PlanarFluenceSensor,
    FarFieldFluenceSensor,
    FarFieldCBSSensor,
    SensorsGroup,
    SimConfig,
    MiePhaseFunction,
    postprocess_farfield_cbs,
)
from luminis_mc import LogLevel, LaserSource
from luminis_mc import run_simulation_parallel, set_log_level
from luminis_mc import Experiment, ResultsLoader
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import __main__

set_log_level(LogLevel.info)

exp = Experiment("sim_cbs_new", "/Users/niaggar/Documents/Thesis/Progress/23Feb26")
exp.log_script(__main__.__file__)
start_time = time.time()


# Medium parameters in micrometers
mean_free_path_sim = 1.0
mean_free_path_real = 2.8
radius_real = 0.1
n_particle_real = 1.58984
n_medium_real = 1.33
inv_mfp_sim = 1 / mean_free_path_sim
mu_absortion_sim = 0.05 * inv_mfp_sim
mu_scattering_sim = inv_mfp_sim - mu_absortion_sim

# Laser parameters
wavelength_real = 0.5145
laser_m_polarization_state = 1/np.sqrt(2)
laser_n_polarization_state = -1j/np.sqrt(2)
laser_radius = 5 * mean_free_path_sim
laser_type = LaserSource.Gaussian

# Phase function parameters
phasef_theta_min = 0.0
phasef_theta_max = np.pi
phasef_ndiv = 1000

# Simulation parameters
n_photons = 100_000_000


laser_source = Laser(laser_m_polarization_state, laser_n_polarization_state, wavelength_real, laser_radius, laser_type)
mie_phase_function = MiePhaseFunction(wavelength_real, radius_real, n_particle_real, n_medium_real, phasef_ndiv, phasef_theta_min, phasef_theta_max)
mie_medium = MieMedium(mu_absortion_sim, mu_scattering_sim, mie_phase_function, mean_free_path_sim, radius_real, n_particle_real, n_medium_real, wavelength_real)




# Cálculo de condiciones
mie_anysotropy = mie_phase_function.get_anisotropy_factor()
size_parameter = 2 * np.pi * radius_real * n_medium_real / wavelength_real
rgd_condition_1 = np.abs(n_particle_real / n_medium_real - 1)
rgd_condition_2 = size_parameter * rgd_condition_1


print("CONDITIONS FOR RAYLEIGH-DEBYE APPROXIMATION:")
print(f"Size parameter: {size_parameter}")
print(f"|n_particle / n_medium - 1|: {rgd_condition_1}")
print(f"Size parameter * |n_particle / n_medium - 1|: {rgd_condition_2}")

print("MEDIUM PARAMETERS:")
print(f"Mean free path: {mean_free_path_sim}")
print(f"Transport mean free path: {mean_free_path_sim / (1 - mie_anysotropy[0])}")
print(f"Particle radius: {radius_real}")
print(f"Wavelength: {wavelength_real}")
print(f"Scattering coefficient: {mu_scattering_sim}")
print(f"Absorption coefficient: {mu_absortion_sim}")
print(f"Albedo: {mu_scattering_sim / (mu_scattering_sim + mu_absortion_sim)}")
print(f"Refractive index particle: {n_particle_real}")
print(f"Refractive index medium: {n_medium_real}")
print(f"Relative refractive index: {n_particle_real / n_medium_real}")
print(f"Mie Anisotropy: {mie_anysotropy}")


# Detector parameters
z_sensor = 0.0
len_x_sensor = 40 * mean_free_path_sim
len_y_sensor = 40 * mean_free_path_sim
pixel_size = 0.1 * mean_free_path_sim
len_t_sensor = 0.0
pixel_t_size = 0.0

# Far-field parameters
theta_max_far_field = np.deg2rad(75)
phi_max_far_field = 2 * np.pi
n_theta_far_field = 75
n_phi_far_field = 30


sens_group = SensorsGroup()
# planar_field_sensor = sens_group.add_detector(PlanarFieldSensor(z_sensor, len_x_sensor, len_y_sensor, pixel_size, pixel_size))
# planar_fluence_sensor = sens_group.add_detector(PlanarFluenceSensor(z_sensor, len_x_sensor, len_y_sensor, len_t_sensor, pixel_size, pixel_size, pixel_t_size))
# far_field_fluence_sensor = sens_group.add_detector(FarFieldFluenceSensor(z_sensor, theta_max_far_field, phi_max_far_field, n_theta_far_field, n_phi_far_field))
far_field_cbs_sensor = sens_group.add_detector(FarFieldCBSSensor(theta_max_far_field, phi_max_far_field, n_theta_far_field, n_phi_far_field))
far_field_cbs_sensor.set_theta_limit(0, theta_max_far_field)



exp.log_params(
    # Medium parameters
    mean_free_path_sim=mean_free_path_sim,
    mean_free_path_real=mean_free_path_real,
    radius_real=radius_real,
    n_particle_real=n_particle_real,
    n_medium_real=n_medium_real,
    mu_absortion_sim=mu_absortion_sim,
    mu_scattering_sim=mu_scattering_sim,
    # Laser parameters
    wavelength_real=wavelength_real,
    laser_m_polarization_state=laser_m_polarization_state,
    laser_n_polarization_state=laser_n_polarization_state,
    laser_radius=laser_radius,
    laser_type=laser_type,
    # Phase function parameters
    phasef_theta_min=phasef_theta_min,
    phasef_theta_max=phasef_theta_max,
    phasef_ndiv=phasef_ndiv,
    # Simulation parameters
    n_photons=n_photons,
    # Detector parameters
    z_sensor=z_sensor,
    len_x_sensor=len_x_sensor,
    len_y_sensor=len_y_sensor,
    pixel_size=pixel_size,
    len_t_sensor=len_t_sensor,
    pixel_t_size=pixel_t_size,
    # Far-field parameters
    theta_max_far_field=theta_max_far_field,
    phi_max_far_field=phi_max_far_field,
    n_theta_far_field=n_theta_far_field,
    n_phi_far_field=n_phi_far_field,
)


config = SimConfig(
    n_photons=n_photons,
    medium=mie_medium,
    detector=sens_group,
    laser=laser_source,
    track_reverse_paths=True,
)
config.n_threads = 8

run_simulation_parallel(config)

end_time = time.time()
print(f"---- Simulation time: {end_time - start_time:.2f} seconds")
# print(f"Photons detected by planar field sensor: {planar_field_sensor.hits}")
# print(f"Photons detected by planar fluence sensor: {planar_fluence_sensor.hits}")
# print(f"Photons detected by far-field fluence sensor: {far_field_fluence_sensor.hits}")
print(f"Photons detected by far-field CBS sensor: {far_field_cbs_sensor.hits}")







# S_0 = np.array(planar_fluence_sensor.S0_t[0], copy=False)
# S_1 = np.array(planar_fluence_sensor.S1_t[0], copy=False)
# S_2 = np.array(planar_fluence_sensor.S2_t[0], copy=False)
# S_3 = np.array(planar_fluence_sensor.S3_t[0], copy=False)
# E_x = np.array(planar_field_sensor.Ex, copy=False)
# E_y = np.array(planar_field_sensor.Ey, copy=False)

# I_ave_x = (S_0 + S_1) / 2
# I_insta_x = np.abs(E_x)**2

# I_ave_y = (S_0 - S_1) / 2
# I_insta_y = np.abs(E_y)**2

# def gaussian_2d_func(coords, amplitude, xo, yo, sigma, offset):
#     x, y = coords
#     g = offset + amplitude * np.exp( -((x-xo)**2 + (y-yo)**2) / (2*sigma**2) )
#     return g.ravel()

# def plot_figure_5_corrected(I_inst, I_avg_raw, NN, HW):
#     # Crear grid espacial
#     x = np.linspace(-HW, HW, NN)
#     y = np.linspace(-HW, HW, NN)
#     X, Y = np.meshgrid(x, y)

#     # ... (El bloque de ajuste Gaussiano se queda igual) ...
#     # Copia el bloque "try/except" y cálculo de "eta" de tu código anterior aquí
#     # o usa el script completo de abajo.

#     # --- REPETIMOS EL AJUSTE PARA QUE EL CÓDIGO ESTÉ COMPLETO ---
#     x_flat = X.ravel()
#     y_flat = Y.ravel()
#     z_flat = I_avg_raw.ravel()
#     initial_guess = [np.max(I_avg_raw), 0, 0, 10, 0]

#     try:
#         popt, pcov = curve_fit(gaussian_2d_func, (x_flat, y_flat), z_flat, p0=initial_guess)
#         I_avg_smooth = gaussian_2d_func((X, Y), *popt).reshape(NN, NN)
#     except:
#         I_avg_smooth = I_avg_raw

#     mask = I_avg_smooth > (np.max(I_avg_smooth) * 0.01)
#     eta = I_inst[mask] / I_avg_smooth[mask]
#     # -----------------------------------------------------------

#     # --- GRAFICACIÓN ---
#     fig = plt.figure(figsize=(16, 10))

#     # (a) Instantaneous Ix
#     ax1 = fig.add_subplot(2, 3, 1)
#     ax1.imshow(I_inst, extent=[-HW, HW, -HW, HW], cmap='gray', origin='lower')
#     # CORRECCIÓN 1: Añadir r antes de las comillas si hay LaTeX (opcional aquí, pero buena práctica)
#     ax1.set_title(r"(a) Instantaneous Intensity $I_x$")
#     ax1.set_xlabel(r"x ($l_s$)")

#     # (b) Average Ix
#     ax2 = fig.add_subplot(2, 3, 2)
#     ax2.imshow(I_avg_raw, extent=[-HW, HW, -HW, HW], cmap='gray', origin='lower')
#     ax2.set_title(r"(b) Average Intensity $\langle I_x \rangle$ (Raw)")
#     ax2.set_xlabel(r"x ($l_s$)")

#     # (c) Probability Density
#     ax3 = fig.add_subplot(2, 3, 3)
#     hist, bins = np.histogram(eta, bins=60, density=True)
#     centers = (bins[:-1] + bins[1:]) / 2

#     ax3.semilogy(centers, hist, 'mo', mfc='none', label='Simulación')

#     x_theory = np.linspace(0, 10, 100)
#     # CORRECCIÓN 2: Añadir r'' aquí por el \eta
#     ax3.semilogy(x_theory, np.exp(-x_theory), 'k-', linewidth=2, label=r'Teoría $e^{-\eta}$')

#     ax3.set_title("(c) Probability Density")
#     # CORRECCIÓN 3: Añadir r'' aquí (ESTA ES LA QUE CAUSABA EL CRASH PRINCIPAL)
#     ax3.set_xlabel(r"$\eta = I_x / \langle I_x \rangle$")

#     ax3.set_ylim(1e-4, 1)
#     ax3.set_xlim(0, 10)
#     ax3.legend()
#     ax3.grid(True, which="both", ls="--", alpha=0.2)

#     # (d) 3D Surface
#     ax4 = fig.add_subplot(2, 3, 4, projection='3d')
#     surf = ax4.plot_surface(X, Y, I_avg_smooth, cmap='jet', linewidth=0, antialiased=False)
#     ax4.set_title("(d) Avg Intensity (Fitted Profile)")

#     # (e) Cross Section X
#     ax5 = fig.add_subplot(2, 3, 5)
#     mid_idx = NN // 2
#     ax5.plot(x, I_avg_raw[mid_idx, :], 'm-', alpha=0.4, label='Data (Raw)')
#     ax5.plot(x, I_avg_smooth[mid_idx, :], 'k-', linewidth=2, label='Fit')
#     ax5.set_title("(e) Cross Section X")
#     ax5.legend()

#     # (f) Cross Section Y
#     ax6 = fig.add_subplot(2, 3, 6)
#     ax6.plot(y, I_avg_raw[:, mid_idx], 'm-', alpha=0.4, label='Data (Raw)')
#     ax6.plot(y, I_avg_smooth[:, mid_idx], 'k-', linewidth=2, label='Fit')
#     ax6.set_title("(f) Cross Section Y")

#     plt.tight_layout()
#     plt.savefig("replicate_fig5_corrected.png", dpi=150)
#     plt.show()


# NN = int(len_x_sensor / pixel_size)
# HW = len_x_sensor / 2
# plot_figure_5_corrected(I_insta_x, I_ave_x, NN, HW)
# plot_figure_5_corrected(I_insta_y, I_ave_y, NN, HW)








def plot_polar_intensity(ax, theta_mesh, phi_mesh, intensity_data, title, cmap='jet'):
    """
    Grafica intensidad en coordenadas polares (Theta = Radio, Phi = Ángulo)
    """
    mesh = ax.pcolormesh(phi_mesh, theta_mesh, intensity_data, cmap=cmap, shading='auto')
    
    # Estética similar a Sawicki et al.
    ax.set_title(title, pad=20)
    ax.grid(False) # Quitar grid para que parezca más una "imagen" óptica
    ax.set_yticklabels([]) # Quitar etiquetas radiales internas si molestan
    ax.set_xticklabels([]) # Quitar ángulos si se quiere limpiar
    
    # Ajustar límite radial
    ax.set_ylim(0, np.max(theta_mesh))
    
    return mesh

def plot_polar_stokes(ax, theta_mesh, phi_mesh, data, title, cmap='jet', vmin=None, vmax=None):
    """
    Grafica parámetros de Stokes en coordenadas polares.
    Admite vmin/vmax para centrar mapas de color divergentes.
    """
    # pcolormesh con shading='auto'
    mesh = ax.pcolormesh(phi_mesh, theta_mesh, data, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    
    # Estética
    ax.set_title(title, pad=15, fontsize=12, fontweight='bold')
    ax.grid(False) 
    ax.set_yticklabels([]) 
    ax.set_xticklabels([])
    
    # Ajustar límite radial
    ax.set_ylim(0, np.max(theta_mesh))
    
    return mesh





cbs_data = postprocess_farfield_cbs(far_field_cbs_sensor, n_photons)



S0_far_field = np.array(cbs_data.coherent.S0, copy=False)
S1_far_field = np.array(cbs_data.coherent.S1, copy=False)
S2_far_field = np.array(cbs_data.coherent.S2, copy=False)
S3_far_field = np.array(cbs_data.coherent.S3, copy=False)

Ix_far = 0.5 * (S0_far_field + S1_far_field)
Iy_far = 0.5 * (S0_far_field - S1_far_field)
I_RCP_far = 0.5 * (S0_far_field + S3_far_field)
I_LCP_far = 0.5 * (S0_far_field - S3_far_field)

theta_coords = np.linspace(0, theta_max_far_field, n_theta_far_field)
phi_coords = np.linspace(0, phi_max_far_field, n_phi_far_field)
THETA, PHI = np.meshgrid(theta_coords, phi_coords, indexing='ij')
theta_deg_max = np.degrees(theta_max_far_field)


fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': 'polar'})
axes = axes.flatten()
# Graficar Ix
c1 = plot_polar_intensity(axes[0], THETA, PHI, Ix_far, r"$I_x$ Backscattering - Coherent")
plt.colorbar(c1, ax=axes[0], fraction=0.046, pad=0.04)
# Graficar Iy
c2 = plot_polar_intensity(axes[1], THETA, PHI, Iy_far, r"$I_y$ Backscattering - Coherent")
plt.colorbar(c2, ax=axes[1], fraction=0.046, pad=0.04)
# Graficar I_RCP
c3 = plot_polar_intensity(axes[2], THETA, PHI, I_RCP_far, r"$I_{RCP}$ Backscattering - Coherent")
plt.colorbar(c3, ax=axes[2], fraction=0.046, pad=0.04)
# Graficar I_LCP
c4 = plot_polar_intensity(axes[3], THETA, PHI, I_LCP_far, r"$I_{LCP}$ Backscattering - Coherent")
plt.colorbar(c4, ax=axes[3], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()



S0_far_field = np.array(cbs_data.incoherent.S0, copy=False)
S1_far_field = np.array(cbs_data.incoherent.S1, copy=False)
S2_far_field = np.array(cbs_data.incoherent.S2, copy=False)
S3_far_field = np.array(cbs_data.incoherent.S3, copy=False)

Ix_far = 0.5 * (S0_far_field + S1_far_field)
Iy_far = 0.5 * (S0_far_field - S1_far_field)
I_RCP_far = 0.5 * (S0_far_field + S3_far_field)
I_LCP_far = 0.5 * (S0_far_field - S3_far_field)

theta_coords = np.linspace(0, theta_max_far_field, n_theta_far_field)
phi_coords = np.linspace(0, phi_max_far_field, n_phi_far_field)
THETA, PHI = np.meshgrid(theta_coords, phi_coords, indexing='ij')
theta_deg_max = np.degrees(theta_max_far_field)


fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': 'polar'})
axes = axes.flatten()
# Graficar Ix
c1 = plot_polar_intensity(axes[0], THETA, PHI, Ix_far, r"$I_x$ Backscattering - Incoherent")
plt.colorbar(c1, ax=axes[0], fraction=0.046, pad=0.04)
# Graficar Iy
c2 = plot_polar_intensity(axes[1], THETA, PHI, Iy_far, r"$I_y$ Backscattering - Incoherent")
plt.colorbar(c2, ax=axes[1], fraction=0.046, pad=0.04)
# Graficar I_RCP
c3 = plot_polar_intensity(axes[2], THETA, PHI, I_RCP_far, r"$I_{RCP}$ Backscattering - Incoherent")
plt.colorbar(c3, ax=axes[2], fraction=0.046, pad=0.04)
# Graficar I_LCP
c4 = plot_polar_intensity(axes[3], THETA, PHI, I_LCP_far, r"$I_{LCP}$ Backscattering - Incoherent")
plt.colorbar(c4, ax=axes[3], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()
