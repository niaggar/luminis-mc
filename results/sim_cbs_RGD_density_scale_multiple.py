import __main__
import time
import numpy as np

from luminis_mc import (
    SweepManager,
    Laser, RGDMedium, Sample, FarFieldCBSSensor, StatisticsSensor, SensorsGroup, SimConfig, RayleighDebyeEMCPhaseFunction,
    run_simulation_parallel, postprocess_farfield_cbs,
    set_log_level, LogLevel, LaserSource
)


set_log_level(LogLevel.info)

exp_name = "test-time-scale"
base_dir = "/Users/niaggar/Documents/Thesis/Progress/16Mar26"


sweep = SweepManager(exp_name, base_dir, timestamped=False)
sweep.snapshot_master_script(__main__.__file__)
sweep.log_readme("CBS Rayleigh-Gans-Debye Polyester particle, fixed radius, sweep over volume fraction -> mean free path")

radius = 0.100
f_scale = 50
volume_fraction_s = [0.05]
scaled_volume_fraction_s = [f * f_scale for f in volume_fraction_s]

n_particle = 1.59
n_medium = 1.33
mu_absortion_percent = 0.0
wavelength = 0.514

# Laser parameters
laser_m_polarization_state = 1/np.sqrt(2)
laser_n_polarization_state = -1j/np.sqrt(2)
laser_radius = 4.0
laser_type = LaserSource.Gaussian

# Phase function parameters
phasef_theta_min = 0.0
phasef_theta_max = np.pi
phasef_ndiv = 100_000

# Simulation parameters
n_photons = 50_000_000

# Sensor parameters
theta_max_far_field = np.deg2rad(30)
phi_max_far_field = 2 * np.pi
n_theta_far_field = 600
n_phi_far_field = 60
d_theta = theta_max_far_field / n_theta_far_field
d_phi = phi_max_far_field / n_phi_far_field

# Light speed is always 1 nm/s
t_max = 50.0
d_time = 5.0

# Statistics sensor parameters
stats_t_max = t_max
stats_d_time = d_time
max_events = 200
max_depth = 100
n_depth = 10


def run_single_simulation(exp, radius, volume_fraction):
    phase = RayleighDebyeEMCPhaseFunction(wavelength, radius, n_particle, n_medium, phasef_ndiv, phasef_theta_min, phasef_theta_max)
    medium = RGDMedium(phase, radius, n_particle, n_medium, wavelength)
    sample = Sample(n_medium)
    sample.add_layer(medium, 0.0, float('inf'))

    scattering_efficiency = medium.scattering_efficiency()
    mean_free_path = (4.0 * radius) / (3.0 * volume_fraction * scattering_efficiency)
    inv_mean_free_path = 1 / mean_free_path

    mean_free_path_real = f_scale * mean_free_path

    mu_absortion = mu_absortion_percent * inv_mean_free_path
    mu_scattering = inv_mean_free_path - mu_absortion

    medium.set_mean_free_path(mean_free_path)
    medium.set_scattering_coefficient(mu_scattering)
    medium.set_absorption_coefficient(mu_absortion)

    laser = Laser(laser_m_polarization_state, laser_n_polarization_state, wavelength, laser_radius * mean_free_path, laser_type)
    anysotropy = phase.get_anisotropy_factor()
    transport_mean_free_path = mean_free_path / (1 - anysotropy[0])
    m_relative = n_particle / n_medium
    size_parameter = 2 * np.pi * radius * n_medium / wavelength
    condition_1 = np.abs(m_relative - 1)
    condition_2 = size_parameter * np.abs(m_relative - 1)
    k_medium = 2 * np.pi * n_medium / wavelength

    max_theta_cbs = 1 / (k_medium * transport_mean_free_path)
    max_theta_cbs_real = 1 / (k_medium * (mean_free_path_real / (1 - anysotropy[0])))

    print("---- Simulation parameters -----")
    print(f"scattering_efficiency {scattering_efficiency:.3f}:")
    print(f"Calculated mean free path: {mean_free_path:.2f} micrometers, scattering coefficient: {mu_scattering:.4f} 1/micrometers")
    print(f"Anisotropy factor for radius {radius:.3f}: {anysotropy[0]:.4f}")
    print(f"Condition 1 (|m-1|): {condition_1:.4f}")
    print(f"Condition 2 (size parameter * |m-1|): {condition_2:.4f}")
    print(f"Maximum scattering angle for CBS: {np.rad2deg(max_theta_cbs):.2f} degrees")
    sens = SensorsGroup()
    det = sens.add_detector(FarFieldCBSSensor(theta_max_far_field, phi_max_far_field, t_max, d_theta, d_phi, d_time, False))
    det.set_theta_limit(0, theta_max_far_field)

    stats = sens.add_detector(StatisticsSensor(z=0, absorb=True))
    stats.set_theta_limit(0, theta_max_far_field)
    stats.set_time_resolution(stats_t_max, stats_d_time)
    stats.set_events_histogram_bins(max_events)
    stats.set_theta_histogram_bins(0, theta_max_far_field, 180)
    stats.set_phi_histogram_bins(0, 2*np.pi, 360)
    stats.set_depth_histogram_bins(max_depth, n_depth)

    config = SimConfig()
    config.n_photons = n_photons
    config.sample = sample
    config.detector = sens
    config.laser = laser
    config.track_reverse_paths = True
    config.pin_threads_to_cores = False
    config.n_threads = 6
    config.show_progress = True


    # 3) params
    exp.log_params(
        # --- 1. System & Physical Properties ---
        radius_um=radius,
        volume_fraction=volume_fraction,
        n_particle=n_particle,
        n_medium=n_medium,
        m_relative=m_relative,

        # --- 2. Calculated Optical Properties ---
        scattering_efficiency=scattering_efficiency,
        mu_scattering_um_inv=mu_scattering,
        mu_absortion_um_inv=mu_absortion,
        anisotropy_factor=anysotropy[0],
        size_parameter=size_parameter,
        condition_1=condition_1,
        condition_2=condition_2,

        # --- 3. The "Yardsticks" (CRITICAL FOR POST-PROCESSING) ---
        mean_free_path_ls_um=mean_free_path,
        mean_free_path_real_um=mean_free_path_real,
        transport_mean_free_path_lstar_um=transport_mean_free_path,
        transport_mean_free_path_real_lstar_um=mean_free_path_real / (1 - anysotropy[0]),

        max_theta_cbs_rad=max_theta_cbs,
        max_theta_cbs_real_rad=max_theta_cbs_real,

        # --- 5. Laser & Simulation Config ---
        wavelength_um=wavelength,
        laser_m_polarization_state=str(laser_m_polarization_state), # Cast complex to string to avoid JSON errors
        laser_n_polarization_state=str(laser_n_polarization_state),
        n_photons=n_photons,

        f_scale=f_scale,
    )

    # 4) run
    t0 = time.time()
    run_simulation_parallel(config)
    print("runtime_s:", time.time() - t0)

    # 5) guarda RAW del sensor
    exp.save_sensor(det, "farfield_cbs")
    exp.save_sensor(stats, "statistics")

    # 6) guarda READY-TO-PLOT (recomendado)
    cbs_total = postprocess_farfield_cbs(det, n_photons)

    print("Saving derived data...")
    print(det.hits)


    theta  = np.linspace(0, det.theta_max, det.N_theta)
    phi   = np.linspace(0, det.phi_max, det.N_phi)
    exp.save_derived("axes/theta_mrad", theta)
    exp.save_derived("axes/phi_rad", phi)

    timed_N = len(cbs_total.coherent)
    for t in range(timed_N):
        s0_total_coh = np.array(cbs_total.coherent[t].S0, copy=False)
        s1_total_coh = np.array(cbs_total.coherent[t].S1, copy=False)
        s2_total_coh = np.array(cbs_total.coherent[t].S2, copy=False)
        s3_total_coh = np.array(cbs_total.coherent[t].S3, copy=False)
        s0_total_inc = np.array(cbs_total.incoherent[t].S0, copy=False)
        s1_total_inc = np.array(cbs_total.incoherent[t].S1, copy=False)
        s2_total_inc = np.array(cbs_total.incoherent[t].S2, copy=False)
        s3_total_inc = np.array(cbs_total.incoherent[t].S3, copy=False)

        exp.save_derived(f"farfield_cbs_timed_{t}/coherent/s0", s0_total_coh)
        exp.save_derived(f"farfield_cbs_timed_{t}/coherent/s1", s1_total_coh)
        exp.save_derived(f"farfield_cbs_timed_{t}/coherent/s2", s2_total_coh)
        exp.save_derived(f"farfield_cbs_timed_{t}/coherent/s3", s3_total_coh)
        exp.save_derived(f"farfield_cbs_timed_{t}/incoherent/s0", s0_total_inc)
        exp.save_derived(f"farfield_cbs_timed_{t}/incoherent/s1", s1_total_inc)
        exp.save_derived(f"farfield_cbs_timed_{t}/incoherent/s2", s2_total_inc)
        exp.save_derived(f"farfield_cbs_timed_{t}/incoherent/s3", s3_total_inc)



for i, volume in enumerate(scaled_volume_fraction_s):
    print(f"Running simulation for volume fraction {volume:.2f} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    run_name = f"volume_fraction_{volume:.2f}"
    fun = lambda exp, r=radius, v=volume: run_single_simulation(exp, r, v)
    sweep.run(i, run_name, fun)
