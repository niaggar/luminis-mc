from luminis_mc import (
    Laser,
    MieMedium,
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
import __main__

set_log_level(LogLevel.info)

exp = Experiment("sim_cbs_new-lin", "/Users/niaggar/Documents/Thesis/Progress/23Feb26")
exp.log_script(__main__.__file__)
start_time = time.time()


# Medium parameters in micrometers
mean_free_path_sim = 1.0
mean_free_path_real = 2.8
radius_real = 0.05
n_particle_real = 1.58984
n_medium_real = 1.33
inv_mfp_sim = 1 / mean_free_path_sim
mu_absortion_sim = 0.0
mu_scattering_sim = inv_mfp_sim - mu_absortion_sim

# Laser parameters
wavelength_real = 0.5145
# laser_m_polarization_state = 1/np.sqrt(2)
# laser_n_polarization_state = -1j/np.sqrt(2)
laser_m_polarization_state = 1
laser_n_polarization_state = 0
laser_radius = 1 * mean_free_path_sim
laser_type = LaserSource.Point

# Phase function parameters
phasef_theta_min = 0.0
phasef_theta_max = np.pi
phasef_ndiv = 1000

# Simulation parameters
n_photons = 100_000


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


# Far-field parameters
theta_max_far_field = np.deg2rad(40)
phi_max_far_field = 2 * np.pi
n_theta_far_field = 200
n_phi_far_field = 1


sens_group = SensorsGroup()
far_field_cbs_sensor = sens_group.add_detector(FarFieldCBSSensor(theta_max_far_field, phi_max_far_field, n_theta_far_field, n_phi_far_field))
far_field_cbs_sensor.set_theta_limit(0, theta_max_far_field)
far_field_cbs_sensor.use_partial_photon = True
far_field_cbs_sensor.theta_pp_max = np.deg2rad(30)
far_field_cbs_sensor.theta_stride = 1
far_field_cbs_sensor.phi_stride = 2

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
print(f"Photons detected by far-field CBS sensor: {far_field_cbs_sensor.hits}")


exp.save_sensor(far_field_cbs_sensor, "far_field_cbs_sensor")



cbs_data = postprocess_farfield_cbs(far_field_cbs_sensor, n_photons)

S0_coh = np.array(cbs_data.coherent.S0,    copy=False)   # (n_theta, n_phi)
S1_coh = np.array(cbs_data.coherent.S1,    copy=False)
S2_coh = np.array(cbs_data.coherent.S2,    copy=False)
S3_coh = np.array(cbs_data.coherent.S3,    copy=False)

S0_inc = np.array(cbs_data.incoherent.S0,  copy=False)
S1_inc = np.array(cbs_data.incoherent.S1,  copy=False)
S2_inc = np.array(cbs_data.incoherent.S2,  copy=False)
S3_inc = np.array(cbs_data.incoherent.S3,  copy=False)

theta_coords = np.linspace(0, theta_max_far_field, n_theta_far_field)
theta_deg    = np.degrees(theta_coords)

# Enhancement 2D y luego promedio en phi
S0_inc_safe = np.where(S0_inc > 0, S0_inc, np.nan)
eta_2d       = S0_coh / S0_inc_safe                        # (n_theta, n_phi)
eta_theta    = np.nanmean(eta_2d, axis=1)                  # (n_theta,)

# Intensidades promediadas en phi (para line plots)
S0_coh_avg = np.mean(S0_coh, axis=1)
S0_inc_avg = np.mean(S0_inc, axis=1)
S1_coh_avg = np.mean(S1_coh, axis=1)
S3_coh_avg = np.mean(S3_coh, axis=1)
S3_inc_avg = np.mean(S3_inc, axis=1)

def plot_test1(theta_deg, eta_theta, S0_coh_avg, S0_inc_avg):
    eta_at_0 = eta_theta[0]
    print(f"[TEST 1] η(θ=0) = {eta_at_0:.4f}  (esperado: 2.0000)")
    print(f"[TEST 1] S0_coh[0]  = {S0_coh_avg[0]:.6e}")
    print(f"[TEST 1] S0_inc[0]  = {S0_inc_avg[0]:.6e}")
    print(f"[TEST 1] Ratio raw  = {S0_coh_avg[0] / S0_inc_avg[0]:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: Intensidades coh vs incoh
    axes[0].plot(theta_deg, S0_coh_avg, 'b-', lw=2, label='Coherente')
    axes[0].plot(theta_deg, S0_inc_avg, 'r--', lw=2, label='Incoherente')
    axes[0].set_xlabel('θ (grados)')
    axes[0].set_ylabel('S0 (normalizado)')
    axes[0].set_title('Intensidades vs θ')
    axes[0].legend()
    axes[0].grid(True)

    # Panel 2: Enhancement factor — el más importante
    axes[1].plot(theta_deg, eta_theta, 'k-', lw=2)
    axes[1].axhline(2.0,  color='r',  ls='--', lw=1, label='η = 2 (teórico)')
    axes[1].axhline(1.0,  color='gray', ls=':', lw=1, label='η = 1 (fondo)')
    axes[1].set_xlabel('θ (grados)')
    axes[1].set_ylabel('η = S0_coh / S0_inc')
    axes[1].set_title(f'Enhancement Factor  |  η(0) = {eta_at_0:.3f}')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_ylim([0.8, 2.5])

    # Panel 3: Coh / (2 * Incoh) — debería ser ~1 fuera del cono y ~1 en el pico
    # Esto normaliza el fondo a 1 para ver la forma del cono más claramente
    ratio_norm = S0_coh_avg / (2.0 * S0_inc_avg)
    axes[2].plot(theta_deg, ratio_norm, 'g-', lw=2)
    axes[2].axhline(1.0, color='r', ls='--', lw=1, label='Máximo teórico')
    axes[2].axhline(0.5, color='gray', ls=':', lw=1, label='Fondo incoherente')
    axes[2].set_xlabel('θ (grados)')
    axes[2].set_ylabel('S0_coh / (2 · S0_inc)')
    axes[2].set_title('Forma del cono CBS (normalizado)')
    axes[2].legend()
    axes[2].grid(True)

    plt.suptitle('TEST 1 — Partículas pequeñas, sin absorción, luz lineal', fontweight='bold')
    plt.tight_layout()
    plt.savefig('test1_enhancement.png', dpi=150)
    plt.show()

plot_test1(theta_deg, eta_theta, S0_coh_avg, S0_inc_avg)


print("=" * 50)
print(f"Hits totales:       {far_field_cbs_sensor.hits}")
print(f"Fotones simulados:  {n_photons}")
print(f"Fracción detectada: {far_field_cbs_sensor.hits / n_photons:.4%}")
print()
print(f"S0_coh[θ=0]:  {S0_coh_avg[0]:.4e}")
print(f"S0_inc[θ=0]:  {S0_inc_avg[0]:.4e}")
print(f"η(θ=0):       {S0_coh_avg[0]/S0_inc_avg[0]:.4f}  ← debe ser ~2.0 en Test 1")
print()
print(f"S0_coh[θ=max]: {S0_coh_avg[-1]:.4e}")
print(f"S0_inc[θ=max]: {S0_inc_avg[-1]:.4e}")
print(f"η(θ=max):      {S0_coh_avg[-1]/S0_inc_avg[-1]:.4f}  ← debe ser ~1.0 (fondo)")
print("=" * 50)

