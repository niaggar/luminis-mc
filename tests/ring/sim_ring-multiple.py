import __main__
import time
import numpy as np
from datetime import datetime

from luminis_mc import (
    SweepManager, ProgressMonitor, on_progress,
    Laser, MieMedium, Sample, PlanarFluenceSensor, StatisticsSensor, SensorsGroup, SimConfig, MiePhaseFunction,
    run_simulation_parallel,
    set_log_level, LogLevel, LaserSource
)


set_log_level(LogLevel.info)

exp_name = "sim_ring"
base_dir = "/Users/niaggar/Documents/Thesis/Progress/02Mar26"

# Medium parameters in micrometers
mean_free_path_sim = 1.0
mean_free_path_real = 2.8
n_particle_real = 1.59
n_medium_real = 1.33
inv_mfp_sim = 1 / mean_free_path_sim
mu_absortion_sim = 0.01 * inv_mfp_sim
mu_scattering_sim = inv_mfp_sim - mu_absortion_sim

# Laser parameters
wavelength_real = 0.632
laser_m_polarization_state_A = 1/np.sqrt(2)
laser_n_polarization_state_A = -1j/np.sqrt(2)
laser_m_polarization_state_B = 1
laser_n_polarization_state_B = 0
laser_radius = 0 * mean_free_path_sim
laser_type = LaserSource.Point

# Sensor parameters
sensor_z = 0
sensor_len_x = 50 * mean_free_path_sim
sensor_len_y = 50 * mean_free_path_sim
sensor_len_t = 50 * mean_free_path_sim
sensor_dx = 0.1 * mean_free_path_sim
sensor_dy = 0.1 * mean_free_path_sim
sensor_dt = 0.5 * mean_free_path_sim
sensor_absorb = True
sensor_estimate = False

# Phase function parameters
phasef_theta_min = 0.0
phasef_theta_max = np.pi
phasef_ndiv = 100_000


# Simulation parameters
# N_PHOTONS = 1_000_000_000
N_PHOTONS = 1_000
radius_sweep = [0.05, 0.1, 0.15, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.7, 1.0]


def run_ring(exp, radious, laser_m_polarization_state, laser_n_polarization_state):
    radius_real = radious

    laser = Laser(laser_m_polarization_state, laser_n_polarization_state, wavelength_real, laser_radius, laser_type)
    phase = MiePhaseFunction(wavelength_real, radius_real, n_particle_real, n_medium_real, phasef_ndiv, phasef_theta_min, phasef_theta_max)
    medium = MieMedium(mu_absortion_sim, mu_scattering_sim, phase, mean_free_path_sim, radius_real, n_particle_real, n_medium_real, wavelength_real)
    sample = Sample(n_medium=n_medium_real)
    sample.add_layer(medium, 0.0, float('inf'))

    anysotropy = phase.get_anisotropy_factor()
    print("Anisotropy factor:", anysotropy)

    sens = SensorsGroup()
    det = sens.add_detector(PlanarFluenceSensor(sensor_z, sensor_len_x, sensor_len_y, sensor_len_t, sensor_dx, sensor_dy, sensor_dt, sensor_absorb, sensor_estimate))
    stats = sens.add_detector(StatisticsSensor(sensor_z, sensor_absorb))
    stats.set_events_histogram_bins(500)
    stats.set_depth_histogram_bins(100 * mean_free_path_sim, 200)
    stats.set_position_limit(-sensor_len_x/2, sensor_len_x/2, -sensor_len_y/2, sensor_len_y/2)

    monitor = ProgressMonitor()
    monitor.setup(total=N_PHOTONS, callback=on_progress, interval_pct=5)

    config = SimConfig(n_photons=N_PHOTONS, sample=sample, detector=sens, laser=laser, track_reverse_paths=False)
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
        anisotropy_g=anysotropy,
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
        n_photons=N_PHOTONS,
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
    exp.save_sensor(det, "planarfluence")
    exp.save_sensor(stats, "statistics")




sweep_A = SweepManager("ring_sweep_A_circular", base_dir, timestamped=True)
sweep_A.snapshot_master_script(__main__.__file__)
sweep_A.log_readme(
    f"Sweep of ring simulations with different radius, using circularly polarized light (m={laser_m_polarization_state_A}, n={laser_n_polarization_state_A})"
)

for i, radius in enumerate(radius_sweep):
    run_name = f"ring_radius_{radius:.3f}"
    sweep_A.run(i, run_name, lambda exp, r=radius, lm=laser_m_polarization_state_A, ln=laser_n_polarization_state_A: run_ring(exp, r, lm, ln))


sweep_B = SweepManager("ring_sweep_B_linear", base_dir, timestamped=True)
sweep_B.snapshot_master_script(__main__.__file__)
sweep_B.log_readme(
    f"Sweep of ring simulations with different radius, using linearly polarized light (m={laser_m_polarization_state_B}, n={laser_n_polarization_state_B})"
)

for i, radius in enumerate(radius_sweep):
    run_name = f"ring_radius_{radius:.3f}"
    sweep_B.run(i, run_name, lambda exp, r=radius, lm=laser_m_polarization_state_B, ln=laser_n_polarization_state_B: run_ring(exp, r, lm, ln))
