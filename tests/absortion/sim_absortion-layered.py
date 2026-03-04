import __main__
import time
import numpy as np
from datetime import datetime

from luminis_mc import (
    SweepManager, ProgressMonitor, on_progress, Absorption,
    Laser, MieMedium, Sample, FarFieldCBSSensor, StatisticsSensor, SensorsGroup, SimConfig, MiePhaseFunction,
    run_simulation_parallel, postprocess_farfield_cbs,
    set_log_level, LogLevel, LaserSource
)

set_log_level(LogLevel.info)

exp_name = "absortion_layered"
base_dir = "/Users/niaggar/Documents/Thesis/Progress/09Mar26"


# Medium parameters in micrometers
mean_free_path_sim = 1.0
mean_free_path_real = 2.8
n_particle_real = 1.58984
n_medium_real = 1.33
inv_mfp_sim = 1 / mean_free_path_sim
mu_absortion_sim = 0.001 * inv_mfp_sim
mu_scattering_sim = inv_mfp_sim - mu_absortion_sim

# Laser parameters
wavelength_real = 0.5145
laser_m_polarization_state = 1/np.sqrt(2)
laser_n_polarization_state = -1j/np.sqrt(2)
laser_radius = 2 * mean_free_path_sim
laser_type = LaserSource.Gaussian

# Phase function parameters
phasef_theta_min = 0.0
phasef_theta_max = np.pi
phasef_ndiv = 1000

# Simulation parameters
n_photons = 100_000
n_layers_sweep = [2, 5, 10, 20, 30]


def run_absortion_layered(exp, n_depth_layer):
    exp.log_readme(f"Absorption simulation - layered sample with {n_depth_layer} layers")

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


    sens = SensorsGroup()
    stats = sens.add_detector(StatisticsSensor(z=0, absorb=True))
    stats.set_events_histogram_bins(500)
    stats.set_theta_histogram_bins(0, np.pi/2, 180)
    stats.set_phi_histogram_bins(0, 2*np.pi, 360)
    stats.set_depth_histogram_bins(50*mean_free_path_sim, 300)

    # Absorption recorder parameters
    absorption_radius = 20 * mean_free_path_sim
    absorption_depth = 50 * mean_free_path_sim
    absorption_dr = 0.1 * mean_free_path_sim
    absorption_dz = 0.1 * mean_free_path_sim
    absorption_dt = 1.0 * mean_free_path_sim
    absorption_tmax = 40 * mean_free_path_sim
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
    exp.save_absorption(absorption, n_photons, "absorption")
    exp.save_sensor(stats, "statistics")


sweep_A = SweepManager(exp_name, base_dir, timestamped=False)
sweep_A.snapshot_master_script(__main__.__file__)
sweep_A.log_readme(
    f"Sweep of CBS simulations with different number of layers, using circularly polarized light (m={laser_m_polarization_state}, n={laser_n_polarization_state})"
)

for i, n_layers in enumerate(n_layers_sweep):
    print(n_layers)
    run_name = f"layers_{n_layers}"
    sweep_A.run(i, run_name, lambda exp, n=n_layers: run_absortion_layered(exp, n))
