import __main__
import time
import numpy as np
from datetime import datetime

from luminis_mc import (
    SweepManager, ProgressMonitor, on_progress,
    Laser, RGDMedium, Sample, Absorption, PlanarFluenceSensor, StatisticsSensor, SensorsGroup, SimConfig, RayleighDebyeEMCPhaseFunction, CrossingDirection,
    run_simulation_parallel,
    set_log_level, LogLevel, LaserSource
)


set_log_level(LogLevel.info)

exp_name = "test"
base_dir = "/Users/niaggar/Documents/Thesis/Progress/16Mar26"


sweep = SweepManager(exp_name, base_dir, timestamped=False)
sweep.snapshot_master_script(__main__.__file__)
sweep.log_readme("CBS Rayleigh-Gans-Debye simulation test - sweep over particle radius")

params_sweep = [
    {
        "radius": 0.070 / 2,
        "volume_fraction": 0.1,
    },
    {
        "radius": 0.070 / 2,
        "volume_fraction": 0.2,
    },
    {
        "radius": 0.110 / 2,
        "volume_fraction": 0.1,
    },
    {
        "radius": 0.110 / 2,
        "volume_fraction": 0.2,
    },
    {
        "radius": 0.350 / 2,
        "volume_fraction": 0.1,
    },
    {
        "radius": 0.350 / 2,
        "volume_fraction": 0.2,
    }
]

n_particle = 1.59
n_medium = 1.33
mu_absortion = 0.0
wavelength = 0.514 # in micrometers, green light

# Laser parameters
laser_m_polarization_state = 1/np.sqrt(2)
laser_n_polarization_state = -1j/np.sqrt(2)
laser_radius = 0.0
laser_type = LaserSource.Gaussian

# Phase function parameters
phasef_theta_min = 0.0
phasef_theta_max = np.pi
phasef_ndiv = 100_000

# Simulation parameters
n_photons = 10_000

# Statistics sensor parameters
max_events = 1000

def run_single_simulation(exp, radius, volume_fraction):
    laser = Laser(laser_m_polarization_state, laser_n_polarization_state, wavelength, laser_radius, laser_type)
    phase = RayleighDebyeEMCPhaseFunction(wavelength, radius, n_particle, n_medium, phasef_ndiv, phasef_theta_min, phasef_theta_max)
    
    medium = RGDMedium(phase, radius, n_particle, n_medium, wavelength)
    sample = Sample(n_medium)
    sample.add_layer(medium, 0.0, float('inf'))

    scattering_efficiency = medium.scattering_efficiency()
    mean_free_path = (4.0 * radius) / (3.0 * volume_fraction * scattering_efficiency)
    mu_scattering = 1 / mean_free_path
    mu_absortion = 0.0

    medium.set_mean_free_path(mean_free_path)
    medium.set_scattering_coefficient(mu_scattering)
    medium.set_absorption_coefficient(mu_absortion)

    anysotropy = phase.get_anisotropy_factor()
    transport_mean_free_path = mean_free_path / (1 - anysotropy[0])
    m_relative = n_particle / n_medium
    size_parameter = 2 * np.pi * radius * n_medium / wavelength
    condition_1 = np.abs(m_relative - 1)
    condition_2 = size_parameter * np.abs(m_relative - 1)


    print("---- Simulation parameters -----")
    print(f"scattering_efficiency {scattering_efficiency:.3f}:")
    print(f"Calculated mean free path: {mean_free_path:.2f} micrometers, scattering coefficient: {mu_scattering:.4f} 1/micrometers")
    print(f"Anisotropy factor for radius {radius:.3f}: {anysotropy[0]:.4f}")
    print(f"Condition 1 (|m-1|): {condition_1:.4f}")
    print(f"Condition 2 (size parameter * |m-1|): {condition_2:.4f}")

    
    dynamic_dx = 0.2 * mean_free_path
    dynamic_len = 15.0 * transport_mean_free_path
    dynamic_z_max = 15.0 * transport_mean_free_path
    dynamic_z_detectors = np.linspace(0.0, dynamic_z_max, 15).tolist()

    print("---- Dynamic Detector Sizing -----")
    print(f"l_s (Mean Free Path) = {mean_free_path:.3f} um")
    print(f"l_star (Transport MFP) = {transport_mean_free_path:.3f} um")
    print(f"Sensor dx/dr set to = {dynamic_dx:.3f} um")
    print(f"Sensor extent set to = {dynamic_len:.3f} um")

    len_t = 0.0
    dt = 0.0
    
    sens = SensorsGroup()
    planar_backscattering = sens.add_detector(PlanarFluenceSensor(0.0, dynamic_len, dynamic_len, len_t, dynamic_dx, dynamic_dx, dt, False, True))

    sensors_z_list = {z: { "sensor": None, "stats": None } for z in dynamic_z_detectors}
    for z in dynamic_z_detectors:
        planar_fluence_sensor = sens.add_detector(PlanarFluenceSensor(z, dynamic_len, dynamic_len, 0.0, dynamic_dx, dynamic_dx, 0.0, False, True))
        planar_fluence_sensor.set_direction_limit(CrossingDirection.Forward)

        stats_z = sens.add_detector(StatisticsSensor(z=z, absorb=False))
        stats_z.set_direction_limit(CrossingDirection.Forward)
        stats_z.set_events_histogram_bins(max_events)

        sensors_z_list[z]["sensor"] = planar_fluence_sensor
        sensors_z_list[z]["stats"] = stats_z

    stats = sens.add_detector(StatisticsSensor(z=0, absorb=True))
    stats.set_events_histogram_bins(max_events)
    stats.set_depth_histogram_bins(dynamic_z_max, int(dynamic_z_max / dynamic_dx))

    dynamic_dt = 0.2 * mean_free_path
    dynamic_t_max = 15.0 * transport_mean_free_path
    absorption = Absorption(dynamic_len / 2, dynamic_z_max, dynamic_dx, dynamic_dx, dynamic_dt, dynamic_t_max)

    monitor = ProgressMonitor()
    monitor.setup(total=n_photons, callback=on_progress, interval_pct=5)

    config = SimConfig()
    config.n_photons = n_photons
    config.sample = sample
    config.detector = sens
    config.laser = laser
    config.absorption = absorption
    config.track_reverse_paths = False
    config.pin_threads_to_cores = True
    config.n_threads = 10
    config.progress = monitor

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
        transport_mean_free_path_lstar_um=transport_mean_free_path,
        
        # --- 4. Dynamic Grid Parameters (CRITICAL FOR PLOTTING) ---
        sensor_dx_um=dynamic_dx,
        sensor_len_um=dynamic_len,
        sensor_z_max_um=dynamic_z_max,
        sensor_dt_pathlength_um=dynamic_dt,
        sensor_t_max_pathlength_um=dynamic_t_max,
        
        # --- 5. Laser & Simulation Config ---
        wavelength_um=wavelength,
        laser_m_polarization_state=str(laser_m_polarization_state), # Cast complex to string to avoid JSON errors
        laser_n_polarization_state=str(laser_n_polarization_state),
        n_photons=n_photons,
    )

    # 4) run
    t0 = time.time()
    run_simulation_parallel(config)
    print("runtime_s:", time.time() - t0)

    # 5) guarda RAW del sensor
    exp.save_sensor(stats, "statistics")
    exp.save_sensor(planar_backscattering, "planar_backscattering")
    exp.save_absorption(absorption, n_photons, "absorption")
    for z in dynamic_z_detectors:
        exp.save_sensor(sensors_z_list[z]["sensor"], f"planar_fluence_z_{z}")
        exp.save_sensor(sensors_z_list[z]["stats"], f"statistics_z_{z}")


    # 6) guarda READY-TO-PLOT (recomendado)

    print(f"Simulation for radius={radius:.3f} nm, volume_fraction={volume_fraction:.1f} completed in {time.time() - t0:.2f} seconds.")
    print("------------------------------")



for i, data in enumerate(params_sweep):
    radius = data["radius"]
    volume_fraction = data["volume_fraction"]

    run_name = f"radius_{radius:.3f}_volumefraction_{volume_fraction:.3f}"
    fun = lambda exp, r=radius, v=volume_fraction: run_single_simulation(exp, r, v)

    print(f"Running simulation for radius={radius:.3f}, volume_fraction={volume_fraction:.1f}")
    sweep.run(i, run_name, fun)



