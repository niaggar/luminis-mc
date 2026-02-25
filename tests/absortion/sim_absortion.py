import __main__
import time
import numpy as np
from datetime import datetime

from luminis_mc import (
    Experiment, ProgressMonitor, on_progress,
    Absorption,
    Laser, MieMedium, Sample, PlanarFluenceSensor, StatisticsSensor, SensorsGroup, SimConfig, MiePhaseFunction,
    run_simulation_parallel,
    set_log_level, LogLevel, LaserSource, TemporalProfile
)

set_log_level(LogLevel.info)

exp_name = "absorption_2-layer"
base_dir = "/Users/niaggar/Documents/Thesis/Progress/02Mar26"


with Experiment(exp_name, base_dir, timestamped=False) as exp:
    exp.log_script(__main__.__file__)
    exp.log_readme("Absorption simulation test")

    # Medium parameters in micrometers
    mean_free_path_sim = 1.0
    mean_free_path_real = 2.8
    radius_real = 10.0
    mean_free_path_sim = 1.0
    mean_free_path_real = 2.8
    n_particle_real = 1.58984
    n_medium_real = 1.33
    inv_mfp_sim = 1 / mean_free_path_sim
    mu_absortion_sim = 0.001 * inv_mfp_sim
    mu_scattering_sim = inv_mfp_sim - mu_absortion_sim
    z_min = 0.0
    z_max = float('inf')

    # Laser parameters
    wavelength_real = 0.52
    laser_m_polarization_state = 1/np.sqrt(2)
    laser_n_polarization_state = -1j/np.sqrt(2)
    laser_radius = 5 * mean_free_path_sim
    laser_type = LaserSource.Gaussian
    laser_temporal_profile = TemporalProfile.Gaussian
    laser_pulse_duration = 1 * mean_free_path_sim
    laser_repetition_rate = 1.0
    laser_time_offset = 0.0

    # Phase function parameters
    phasef_theta_min = 0.0
    phasef_theta_max = np.pi
    phasef_ndiv = 1000

    # Simulation parameters
    n_photons = 1_000_000

    laser = Laser(laser_m_polarization_state, laser_n_polarization_state, wavelength_real, laser_radius, laser_type)
    laser.set_temporal_profile(laser_temporal_profile, laser_pulse_duration, laser_repetition_rate, laser_time_offset)
    
    phase_1 = MiePhaseFunction(wavelength_real, radius_real, n_particle_real, n_medium_real, phasef_ndiv, phasef_theta_min, phasef_theta_max)
    medium_1 = MieMedium(mu_absortion_sim, mu_scattering_sim, phase_1, mean_free_path_sim, radius_real, n_particle_real, n_medium_real, wavelength_real)
    
    radius_real_2 = 0.01
    phase_2 = MiePhaseFunction(wavelength_real, radius_real_2, n_particle_real, n_medium_real, phasef_ndiv, phasef_theta_min, phasef_theta_max)
    medium_2 = MieMedium(mu_absortion_sim, mu_scattering_sim, phase_2, mean_free_path_sim, radius_real_2, n_particle_real, n_medium_real, wavelength_real)

    sample = Sample(n_medium=n_medium_real)
    sample.add_layer(medium_1, z_min, 6 * mean_free_path_sim)
    sample.add_layer(medium_2, 6 * mean_free_path_sim, z_max)

    anysotropy_1 = phase_1.get_anisotropy_factor()
    anysotropy_2 = phase_2.get_anisotropy_factor()
    print("Anisotropy factor 1:", anysotropy_1)
    print("Anisotropy factor 2:", anysotropy_2)

    # Sensor parameters
    sensor_z = 0
    sensor_len_x = 40 * mean_free_path_sim
    sensor_len_y = 40 * mean_free_path_sim
    sensor_len_t = 0 * mean_free_path_sim
    sensor_dx = 0.1 * mean_free_path_sim
    sensor_dy = 0.1 * mean_free_path_sim
    sensor_dt = 0 * mean_free_path_sim
    sensor_absorb = True
    sensor_estimate = False

    sens = SensorsGroup()
    planar_fluence_sensor = sens.add_detector(PlanarFluenceSensor(sensor_z, sensor_len_x, sensor_len_y, sensor_len_t, sensor_dx, sensor_dy, sensor_dt, sensor_absorb, sensor_estimate))
    stats = sens.add_detector(StatisticsSensor(sensor_z, sensor_absorb))

    stats.set_events_histogram_bins(500)
    stats.set_position_limit(-sensor_len_x/2, sensor_len_x/2, -sensor_len_y/2, sensor_len_y/2)

    # Absorption recorder parameters
    absorption_radius = 40 * mean_free_path_sim
    absorption_depth = 30 * mean_free_path_sim
    absorption_dr = 0.1 * mean_free_path_sim
    absorption_dz = 0.1 * mean_free_path_sim
    absorption_dt = 1.0
    absorption_tmax = 40.0

    absorption = Absorption(absorption_radius, absorption_depth, absorption_dr, absorption_dz, absorption_dt, absorption_tmax)

    monitor = ProgressMonitor()
    monitor.setup(total=n_photons, callback=on_progress, interval_pct=5)

    config = SimConfig(n_photons=n_photons, sample=sample, detector=sens, laser=laser, absorption=absorption, track_reverse_paths=False)
    config.n_threads = 7
    config.progress = monitor

    # 3) params
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
        anysotropy_1=anysotropy_1,
        anysotropy_2=anysotropy_2,
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
    exp.save_sensor(planar_fluence_sensor, "planarfluence")
    exp.save_sensor(stats, "statistics")
    exp.save_absorption(absorption, n_photons, "absorption")


