import __main__
import time
import numpy as np
from datetime import datetime

from luminis_mc import (
    SweepManager, ProgressMonitor, on_progress,
    Laser, RGDMedium, Sample, FarFieldCBSSensor, StatisticsSensor, SensorsGroup, SimConfig, RayleighDebyeEMCPhaseFunction,
    run_simulation_parallel, postprocess_farfield_cbs,
    set_log_level, LogLevel, LaserSource
)


set_log_level(LogLevel.info)

exp_name = "sim_cbs_RGD_multiple"
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
wavelength = 0.514

# Laser parameters
laser_m_polarization_state = 1
laser_n_polarization_state = 0
laser_radius = 0.0
laser_type = LaserSource.Point

# Phase function parameters
phasef_theta_min = 0.0
phasef_theta_max = np.pi
phasef_ndiv = 100_000

# Simulation parameters
n_photons = 1

# Events study
scattering_order_bins = [2, 3, 4, 5, 7, 10, 15, 20, 50]


# Sensor parameters
theta_max_far_field = np.deg2rad(5)
phi_max_far_field = 2 * np.pi
n_theta_far_field = 400
n_phi_far_field = 360
d_theta = theta_max_far_field / n_theta_far_field
d_phi = phi_max_far_field / n_phi_far_field

# Light speed is always 1 nm/s
t_max = 200
d_time = 50



# Statistics sensor parameters
stats_t_max = t_max
stats_d_time = d_time
max_events = 1000
max_depth = 100 * 10.0  # mean free path is expected to be around 10 nm, so this covers up to 100 mfp
n_depth = 10


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

    print(f"scattering_efficiency {scattering_efficiency:.3f} nm:")
    print(f"Calculated mean free path: {mean_free_path:.2f} nm, scattering coefficient: {mu_scattering:.4f} 1/nm")

    medium.set_mean_free_path(mean_free_path)
    medium.set_scattering_coefficient(mu_scattering)
    medium.set_absorption_coefficient(mu_absortion)

    sens = SensorsGroup()
    det = sens.add_detector(FarFieldCBSSensor(theta_max_far_field, phi_max_far_field, t_max, d_theta, d_phi, d_time, False))
    det.set_theta_limit(0, theta_max_far_field)

    stats = sens.add_detector(StatisticsSensor(z=0, absorb=True))
    stats.set_time_resolution(stats_t_max, stats_d_time)
    stats.set_events_histogram_bins(max_events)
    stats.set_theta_histogram_bins(0, np.pi/2, 180)
    stats.set_phi_histogram_bins(0, 2*np.pi, 360)
    stats.set_depth_histogram_bins(max_depth, n_depth)

    scattering_order_detectors = {
        order: None for order in scattering_order_bins
    }

    for order in scattering_order_bins:
        det_order = sens.add_detector(FarFieldCBSSensor(theta_max_far_field, phi_max_far_field, t_max, d_theta, d_phi, d_time, False))
        det_order.set_theta_limit(0, theta_max_far_field)
        det_order.set_events_limit(order, order)
        scattering_order_detectors[order] = det_order


    monitor = ProgressMonitor()
    monitor.setup(total=n_photons, callback=on_progress, interval_pct=5)

    config = SimConfig()
    config.n_photons = n_photons
    config.sample = sample
    config.detector = sens
    config.laser = laser
    config.track_reverse_paths = True
    config.pin_threads_to_cores = False
    config.n_threads = 7
    config.progress = monitor

    anysotropy = phase.get_anisotropy_factor()
    print(f"Anisotropy factor for radius {radius:.3f}: {anysotropy[0]:.4f}")

    m_relative = n_particle / n_medium
    condition_1 = np.abs(m_relative - 1)
    size_parameter = 2 * np.pi * radius / wavelength
    condition_2 = size_parameter * np.abs(m_relative - 1)
    print(f"Condition 1 (|m-1|): {condition_1:.4f}")
    print(f"Condition 2 (size parameter * |m-1|): {condition_2:.4f}")

    # 3) params
    exp.log_params(
        # Medium parameters
        mean_free_path_real=mean_free_path,
        radius_real=radius,
        n_particle_real=n_particle,
        n_medium_real=n_medium,
        mu_absortion_sim=mu_absortion,
        mu_scattering_sim=mu_scattering,
        # Laser parameters
        wavelength_real=wavelength,
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
        # Calculated parameters
        anisotropy_factor=anysotropy[0],
        size_parameter=size_parameter,
        condition_1=condition_1,
        condition_2=condition_2
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

    theta  = np.linspace(0, det.theta_max, det.N_theta)
    phi   = np.linspace(0, det.phi_max, det.N_phi)

    exp.save_derived("axes/theta_mrad", theta)
    exp.save_derived("axes/phi_rad", phi)


    for order, det_order in scattering_order_detectors.items():
        exp.save_sensor(det_order, f"farfield_cbs_scattering_order_{order}")

        cbs_order = postprocess_farfield_cbs(det_order, n_photons)
        s0_order_coh = np.array(cbs_order.coherent[0].S0, copy=False)
        s1_order_coh = np.array(cbs_order.coherent[0].S1, copy=False)
        s2_order_coh = np.array(cbs_order.coherent[0].S2, copy=False)
        s3_order_coh = np.array(cbs_order.coherent[0].S3, copy=False)
        s0_order_inc = np.array(cbs_order.incoherent[0].S0, copy=False)
        s1_order_inc = np.array(cbs_order.incoherent[0].S1, copy=False)
        s2_order_inc = np.array(cbs_order.incoherent[0].S2, copy=False)
        s3_order_inc = np.array(cbs_order.incoherent[0].S3, copy=False)
        exp.save_derived(f"farfield_cbs_scattering_order_{order}/coherent/s0", s0_order_coh)
        exp.save_derived(f"farfield_cbs_scattering_order_{order}/coherent/s1", s1_order_coh)
        exp.save_derived(f"farfield_cbs_scattering_order_{order}/coherent/s2", s2_order_coh)
        exp.save_derived(f"farfield_cbs_scattering_order_{order}/coherent/s3", s3_order_coh)
        exp.save_derived(f"farfield_cbs_scattering_order_{order}/incoherent/s0", s0_order_inc)
        exp.save_derived(f"farfield_cbs_scattering_order_{order}/incoherent/s1", s1_order_inc)
        exp.save_derived(f"farfield_cbs_scattering_order_{order}/incoherent/s2", s2_order_inc)
        exp.save_derived(f"farfield_cbs_scattering_order_{order}/incoherent/s3", s3_order_inc)



for i, data in enumerate(params_sweep):
    radius = data["radius"]
    volume_fraction = data["volume_fraction"]

    run_name = f"radius_{radius:.3f}_volumefraction_{volume_fraction:.3f}"
    fun = lambda exp, r=radius, v=volume_fraction: run_single_simulation(exp, r, v)

    print(f"Running simulation for radius={radius:.3f}, volume_fraction={volume_fraction:.1f}")
    sweep.run(i, run_name, fun)



