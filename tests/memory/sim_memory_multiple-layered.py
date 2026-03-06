import __main__
import time
import numpy as np
from datetime import datetime

from luminis_mc import (
    SweepManager, ProgressMonitor, on_progress,
    Laser, MieMedium, Sample, FarFieldCBSSensor, PlanarFluenceSensor, StatisticsSensor, SensorsGroup, SimConfig, MiePhaseFunction, CrossingDirection,
    run_simulation_parallel, postprocess_farfield_cbs,
    set_log_level, LogLevel, LaserSource
)

set_log_level(LogLevel.info)

exp_name = "polarization_memory"
base_dir = "/Users/niaggar/Documents/Thesis/Progress/09Mar26"


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

z_detectors = [1*mean_free_path_sim, 2*mean_free_path_sim, 3*mean_free_path_sim, 4*mean_free_path_sim, 5*mean_free_path_sim, 10*mean_free_path_sim, 15*mean_free_path_sim, 20*mean_free_path_sim, 25*mean_free_path_sim, 30*mean_free_path_sim]


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


    anysotropy_a = phase_a.get_anisotropy_factor()
    anisotropy_b = phase_b.get_anisotropy_factor()
    print(f"Layer A: radius={radious_real_a} μm, anisotropy={anysotropy_a[0]:.4f}")
    print(f"Layer B: radius={radious_real_b} μm, anisotropy={anisotropy_b[0]:.4f}")

    # Sensor parameters
    theta_max_far_field = np.deg2rad(45)
    phi_max_far_field = 2 * np.pi
    t_max_far_field = max(40, 4 * n_depth_layer + 20) * mean_free_path_sim
    n_theta_far_field = 500
    n_phi_far_field = 1
    n_t_far_field = int(t_max_far_field / (2 * mean_free_path_sim))
    estimator_enabled_far_field = False


    sens = SensorsGroup()
    det_total = sens.add_detector(FarFieldCBSSensor(theta_max_far_field, phi_max_far_field, 0, n_theta_far_field, n_phi_far_field, 1, estimator_enabled_far_field))
    stats = sens.add_detector(StatisticsSensor(z=0, absorb=True))

    det_total.set_theta_limit(0, theta_max_far_field)
    det_total.theta_pp_max = np.deg2rad(30)
    det_total.theta_stride = 1
    det_total.phi_stride = 1

    stats.set_events_histogram_bins(1000)
    stats.set_depth_histogram_bins(100*mean_free_path_sim, 300)

    sensor_len_x = 40 * mean_free_path_sim
    sensor_len_y = 40 * mean_free_path_sim
    sensor_len_t = 0 * mean_free_path_sim
    sensor_dx = 0.1 * mean_free_path_sim
    sensor_dy = 0.1 * mean_free_path_sim
    sensor_dt = 0 * mean_free_path_sim
    sensor_absorb = False
    sensor_estimate = False

    sensors_z_list = {z: { "sensor": None, "stats": None } for z in z_detectors}
    for z in z_detectors:
        planar_fluence_sensor = sens.add_detector(PlanarFluenceSensor(z, sensor_len_x, sensor_len_y, sensor_len_t, sensor_dx, sensor_dy, sensor_dt, sensor_absorb, sensor_estimate))
        planar_fluence_sensor.set_direction_limit(CrossingDirection.Forward)

        stats_z = sens.add_detector(StatisticsSensor(z=z, absorb=False))
        stats_z.set_direction_limit(CrossingDirection.Forward)
        stats_z.set_events_histogram_bins(500)

        sensors_z_list[z]["sensor"] = planar_fluence_sensor
        sensors_z_list[z]["stats"] = stats_z

    monitor = ProgressMonitor()
    monitor.setup(total=n_photons, callback=on_progress, interval_pct=5)

    config = SimConfig(n_photons=n_photons, sample=sample, detector=sens, laser=laser, track_reverse_paths=False)
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
        # layered sensor z
        z_detectors=z_detectors
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
    exp.save_sensor(det_total, "farfieldcbs_total")
    exp.save_sensor(stats, "statistics")

    for z in z_detectors:
        exp.save_sensor(sensors_z_list[z]["sensor"], f"planar_fluence_z_{z}")
        exp.save_sensor(sensors_z_list[z]["stats"], f"statistics_z_{z}")


    cbs_total = postprocess_farfield_cbs(det_total, n_photons)
    s0_total_coh = np.array(cbs_total.coherent[0].S0, copy=False)
    s1_total_coh = np.array(cbs_total.coherent[0].S1, copy=False)
    s2_total_coh = np.array(cbs_total.coherent[0].S2, copy=False)
    s3_total_coh = np.array(cbs_total.coherent[0].S3, copy=False)
    s0_total_inc = np.array(cbs_total.incoherent[0].S0, copy=False)
    s1_total_inc = np.array(cbs_total.incoherent[0].S1, copy=False)
    s2_total_inc = np.array(cbs_total.incoherent[0].S2, copy=False)
    s3_total_inc = np.array(cbs_total.incoherent[0].S3, copy=False)
    exp.save_derived(f"farfieldcbs_total/coherent/s0", s0_total_coh)
    exp.save_derived(f"farfieldcbs_total/coherent/s1", s1_total_coh)
    exp.save_derived(f"farfieldcbs_total/coherent/s2", s2_total_coh)
    exp.save_derived(f"farfieldcbs_total/coherent/s3", s3_total_coh)
    exp.save_derived(f"farfieldcbs_total/incoherent/s0", s0_total_inc)
    exp.save_derived(f"farfieldcbs_total/incoherent/s1", s1_total_inc)
    exp.save_derived(f"farfieldcbs_total/incoherent/s2", s2_total_inc)
    exp.save_derived(f"farfieldcbs_total/incoherent/s3", s3_total_inc)


sweep_A = SweepManager(exp_name, base_dir, timestamped=True)
sweep_A.snapshot_master_script(__main__.__file__)
sweep_A.log_readme(
    f"This sweep runs CBS simulations for layered media with different numbers of layers. Each layer has a different anisotropy factor, and the simulation captures how the CBS signal evolves as the number of layers increases. The results include the coherent and incoherent components of the CBS signal, as well as statistics on photon interactions at various depths within the medium."
)

for i, n_layers in enumerate(n_layers_sweep):
    print(n_layers)
    run_name = f"layers_{n_layers}"
    sweep_A.run(i, run_name, lambda exp, n=n_layers: run_cbs_layered(exp, n))
