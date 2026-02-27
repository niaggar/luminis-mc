import __main__
import time
import numpy as np
from datetime import datetime

from luminis_mc import (
    SweepManager, ProgressMonitor, on_progress,
    Laser, MieMedium, Sample, FarFieldCBSSensor, StatisticsSensor, SensorsGroup, SimConfig, MiePhaseFunction,
    run_simulation_parallel, postprocess_farfield_cbs,
    set_log_level, LogLevel, LaserSource
)

set_log_level(LogLevel.info)

exp_name = "sim_cbs"
base_dir = "/Users/niaggar/Documents/Thesis/Progress/02Mar26"


# Medium parameters in micrometers
mean_free_path_sim = 1.0
mean_free_path_real = 2.8
n_particle_real = 1.58984
n_medium_real = 1.33
inv_mfp_sim = 1 / mean_free_path_sim
mu_absortion_sim = 0.0
mu_scattering_sim = inv_mfp_sim - mu_absortion_sim

# Laser parameters
wavelength_real = 0.5145
laser_m_polarization_state = 1/np.sqrt(2)
laser_n_polarization_state = -1j/np.sqrt(2)
laser_radius = 1 * mean_free_path_sim
laser_type = LaserSource.Point

# Phase function parameters
phasef_theta_min = 0.0
phasef_theta_max = np.pi
phasef_ndiv = 1000

# Simulation parameters
n_photons = 1_000
n_layers_sweep = [2, 5, 10, 20]


def run_cbs_layered(exp, n_depth_layer):
    exp.log_readme("CBS simulation test")

    laser = Laser(laser_m_polarization_state, laser_n_polarization_state, wavelength_real, laser_radius, laser_type)
    
    zmin_medium_a = 0.0
    zmax_medium_a = n_depth_layer * mean_free_path_sim
    zmin_medium_b = zmax_medium_a
    zmax_medium_b = float('inf')

    radious_real_a = 0.15 # low anisotropy
    radious_real_b = 1.0  # high anisotropy
    
    phase_a = MiePhaseFunction(wavelength_real, radious_real_a, n_particle_real, n_medium_real, phasef_ndiv, phasef_theta_min, phasef_theta_max)
    phase_b = MiePhaseFunction(wavelength_real, radious_real_b, n_particle_real, n_medium_real, phasef_ndiv, phasef_theta_min, phasef_theta_max)
    medium_a = MieMedium(mu_absortion_sim, mu_scattering_sim, phase_a, mean_free_path_sim, radious_real_a, n_particle_real, n_medium_real, wavelength_real)
    medium_b = MieMedium(mu_absortion_sim, mu_scattering_sim, phase_b, mean_free_path_sim, radious_real_b, n_particle_real, n_medium_real, wavelength_real)
    sample = Sample(n_medium=n_medium_real)
    sample.add_layer(medium_a, zmin_medium_a, zmax_medium_a)
    sample.add_layer(medium_b, zmin_medium_b, zmax_medium_b)

    # Sensor parameters
    theta_max_far_field = np.deg2rad(45)
    phi_max_far_field = 2 * np.pi
    t_max_far_field = max(40, 4 * n_depth_layer + 20) * mean_free_path_sim
    n_theta_far_field = 500
    n_phi_far_field = 1
    n_t_far_field = int(t_max_far_field / (2 * mean_free_path_sim))
    estimator_enabled_far_field = False

    sens = SensorsGroup()
    det_timed = sens.add_detector(FarFieldCBSSensor(theta_max_far_field, phi_max_far_field, t_max_far_field, n_theta_far_field, n_phi_far_field, n_t_far_field, estimator_enabled_far_field))
    det_total = sens.add_detector(FarFieldCBSSensor(theta_max_far_field, phi_max_far_field, 0, n_theta_far_field, n_phi_far_field, 1, estimator_enabled_far_field))
    stats = sens.add_detector(StatisticsSensor(z=0, absorb=True))

    det_timed.set_theta_limit(0, theta_max_far_field)
    det_timed.theta_pp_max = np.deg2rad(30)
    det_timed.theta_stride = 1
    det_timed.phi_stride = 2

    det_total.set_theta_limit(0, theta_max_far_field)
    det_total.theta_pp_max = np.deg2rad(30)
    det_total.theta_stride = 1
    det_total.phi_stride = 1

    stats.set_events_histogram_bins(500)
    stats.set_theta_histogram_bins(0, np.pi/2, 180)
    stats.set_phi_histogram_bins(0, 2*np.pi, 360)
    stats.set_depth_histogram_bins(10*mean_free_path_sim, 100)

    monitor = ProgressMonitor()
    monitor.setup(total=n_photons, callback=on_progress, interval_pct=5)

    config = SimConfig(n_photons=n_photons, sample=sample, detector=sens, laser=laser, track_reverse_paths=True)
    config.n_threads = 7
    config.progress = monitor

    # 3) params
    exp.log_params(
        # Medium parameters
        mean_free_path_sim=mean_free_path_sim,
        mean_free_path_real=mean_free_path_real,
        radius_real_a=radious_real_a,
        radius_real_b=radious_real_b,
        n_depth_layer=n_depth_layer,
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

    # 4) run
    t0 = time.time()
    run_simulation_parallel(config)
    print("runtime_s:", time.time() - t0)

    exp.update_log_readme(
        status="Simulation completed",
        runtime_s=time.time() - t0,
        finished_at=datetime.now().isoformat(timespec="seconds")
    )

    # 5) guarda RAW del sensor
    exp.save_sensor(det_timed, "farfieldcbs_timed")
    exp.save_sensor(det_total, "farfieldcbs_total")
    exp.save_sensor(stats, "statistics")

    # 6) guarda READY-TO-PLOT (recomendado)
    cbs = postprocess_farfield_cbs(det_timed, n_photons)
    i_coh = 0
    i_inc = 0
    for coh_t in cbs.coherent:
        s0_coh = np.array(coh_t.S0, copy=False)
        s1_coh = np.array(coh_t.S1, copy=False)
        s2_coh = np.array(coh_t.S2, copy=False)
        s3_coh = np.array(coh_t.S3, copy=False)
        exp.save_derived(f"farfieldcbs_timed/coherent/t{i_coh}_s0", s0_coh)
        exp.save_derived(f"farfieldcbs_timed/coherent/t{i_coh}_s1", s1_coh)
        exp.save_derived(f"farfieldcbs_timed/coherent/t{i_coh}_s2", s2_coh)
        exp.save_derived(f"farfieldcbs_timed/coherent/t{i_coh}_s3", s3_coh)
        i_coh += 1
    for inc_t in cbs.incoherent:
        s0_inc = np.array(inc_t.S0, copy=False)
        s1_inc = np.array(inc_t.S1, copy=False)
        s2_inc = np.array(inc_t.S2, copy=False)
        s3_inc = np.array(inc_t.S3, copy=False)
        exp.save_derived(f"farfieldcbs_timed/incoherent/t{i_inc}_s0", s0_inc)
        exp.save_derived(f"farfieldcbs_timed/incoherent/t{i_inc}_s1", s1_inc)
        exp.save_derived(f"farfieldcbs_timed/incoherent/t{i_inc}_s2", s2_inc)
        exp.save_derived(f"farfieldcbs_timed/incoherent/t{i_inc}_s3", s3_inc)
        i_inc += 1



sweep_A = SweepManager("cbs_sweep_layered", base_dir, timestamped=True)
sweep_A.snapshot_master_script(__main__.__file__)
sweep_A.log_readme(
    f"Sweep of CBS simulations with different number of layers, using circularly polarized light (m={laser_m_polarization_state}, n={laser_n_polarization_state})"
)

for i, n_layers in enumerate(n_layers_sweep):
    run_name = f"n_layers_{n_layers}"
    sweep_A.run(i, run_name, lambda exp, n=n_layers: run_cbs_layered(exp, n))
