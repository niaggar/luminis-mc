import __main__
import time
import numpy as np
from datetime import datetime

from luminis_mc import (
    Experiment, ResultsLoader,
    Laser, MieMedium, PlanarFluenceSensor, StatisticsSensor, SensorsGroup, SimConfig, MiePhaseFunction,
    run_simulation_parallel, postprocess_farfield_cbs,
    set_log_level, LogLevel, LaserSource
)

set_log_level(LogLevel.info)

exp_name = "sim_ring"
base_dir = "/Users/niaggar/Documents/Thesis/Progress/23Feb26"


with Experiment(exp_name, base_dir) as exp:
    exp.log_script(__main__.__file__)
    exp.log_readme("Ring simulation test")

    # Medium parameters in micrometers
    mean_free_path_sim = 1.0
    mean_free_path_real = 2.8
    radius_real = 0.5
    n_particle_real = 1.59
    n_medium_real = 1.33
    inv_mfp_sim = 1 / mean_free_path_sim
    mu_absortion_sim = 0.01 * inv_mfp_sim
    mu_scattering_sim = inv_mfp_sim - mu_absortion_sim

    # Laser parameters
    wavelength_real = 0.52
    laser_m_polarization_state = 1/np.sqrt(2)
    laser_n_polarization_state = -1j/np.sqrt(2)
    laser_radius = 0 * mean_free_path_sim
    laser_type = LaserSource.Point

    # Phase function parameters
    phasef_theta_min = 0.0
    phasef_theta_max = np.pi
    phasef_ndiv = 1000

    # Simulation parameters
    n_photons = 1_000_000

    laser = Laser(laser_m_polarization_state, laser_n_polarization_state, wavelength_real, laser_radius, laser_type)
    phase = MiePhaseFunction(wavelength_real, radius_real, n_particle_real, n_medium_real, phasef_ndiv, phasef_theta_min, phasef_theta_max)
    medium = MieMedium(mu_absortion_sim, mu_scattering_sim, phase, mean_free_path_sim, radius_real, n_particle_real, n_medium_real, wavelength_real)

    anysotropy = phase.get_anisotropy_factor()
    print("Anisotropy factor:", anysotropy)


    # Sensor parameters
    sensor_z = 0
    sensor_len_x = 50 * mean_free_path_sim
    sensor_len_y = 50 * mean_free_path_sim
    sensor_len_t = 10 * mean_free_path_sim
    sensor_dx = 0.1 * mean_free_path_sim
    sensor_dy = 0.1 * mean_free_path_sim
    sensor_dt = 0.5 * mean_free_path_sim
    sensor_absorb = True
    sensor_estimate = True

    sens = SensorsGroup()
    det = sens.add_detector(PlanarFluenceSensor(sensor_z, sensor_len_x, sensor_len_y, sensor_len_t, sensor_dx, sensor_dy, sensor_dt, sensor_absorb, sensor_estimate))
    stats = sens.add_detector(StatisticsSensor(sensor_z, sensor_absorb))

    stats.set_events_histogram_bins(500)
    stats.set_position_limit(-sensor_len_x/2, sensor_len_x/2, -sensor_len_y/2, sensor_len_y/2)

    config = SimConfig(n_photons=n_photons, medium=medium, detector=sens, laser=laser, track_reverse_paths=True)
    config.n_threads = 7

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
        # Far-field parameters
        sensor_z=sensor_z,
        sensor_len_x=sensor_len_x,
        sensor_len_y=sensor_len_y,
        sensor_len_t=sensor_len_t,
        sensor_dx=sensor_dx,
        sensor_dy=sensor_dy,
        sensor_dt=sensor_dt,
        sensor_absorb=sensor_absorb,
        sensor_estimate=sensor_estimate
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
    exp.save_sensor(det, "planar_fluence")
    exp.save_sensor(stats, "statistics")



