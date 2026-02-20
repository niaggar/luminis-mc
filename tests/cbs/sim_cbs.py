import __main__
import time
import numpy as np
from datetime import datetime

from luminis_mc import (
    Experiment, ResultsLoader,
    Laser, MieMedium, FarFieldCBSSensor, SensorsGroup, SimConfig, MiePhaseFunction,
    run_simulation_parallel, postprocess_farfield_cbs,
    set_log_level, LogLevel, LaserSource
)

set_log_level(LogLevel.info)

exp_name = "sim_cbs"
base_dir = "/Users/niaggar/Documents/Thesis/Progress/23Feb26"


with Experiment(exp_name, base_dir) as exp:
    exp.log_script(__main__.__file__)
    exp.log_readme("CBS simulation test")

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

    laser = Laser(laser_m_polarization_state, laser_n_polarization_state, wavelength_real, laser_radius, laser_type)
    phase = MiePhaseFunction(wavelength_real, radius_real, n_particle_real, n_medium_real, phasef_ndiv, phasef_theta_min, phasef_theta_max)
    medium = MieMedium(mu_absortion_sim, mu_scattering_sim, phase, mean_free_path_sim, radius_real, n_particle_real, n_medium_real, wavelength_real)

    # Sensor parameters
    theta_max_far_field = np.deg2rad(40)
    phi_max_far_field = 2 * np.pi
    n_theta_far_field = 200
    n_phi_far_field = 1

    sens = SensorsGroup()
    det = sens.add_detector(FarFieldCBSSensor(theta_max_far_field, phi_max_far_field, n_theta_far_field, n_phi_far_field))
    det.set_theta_limit(0, theta_max_far_field)
    det.use_partial_photon = True
    det.theta_pp_max = np.deg2rad(30)
    det.theta_stride = 1
    det.phi_stride = 2

    config = SimConfig(n_photons=n_photons, medium=medium, detector=sens, laser=laser, track_reverse_paths=True)
    config.n_threads = 8

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
    exp.save_sensor(det, "farfield_cbs")

    # 6) guarda READY-TO-PLOT (recomendado)
    cbs = postprocess_farfield_cbs(det, n_photons)
    S0_coh = np.array(cbs.coherent.S0, copy=False)
    S1_coh = np.array(cbs.coherent.S1, copy=False)
    S2_coh = np.array(cbs.coherent.S2, copy=False)
    S3_coh = np.array(cbs.coherent.S3, copy=False)
    S0_inc = np.array(cbs.incoherent.S0, copy=False)
    S1_inc = np.array(cbs.incoherent.S1, copy=False)
    S2_inc = np.array(cbs.incoherent.S2, copy=False)
    S3_inc = np.array(cbs.incoherent.S3, copy=False)
    theta  = np.linspace(0, det.theta_max, det.N_theta)
    phi   = np.linspace(0, det.phi_max, det.N_phi)

    exp.save_derived("farfield_cbs/theta", theta)
    exp.save_derived("farfield_cbs/phi", phi)
    exp.save_derived("farfield_cbs/S0_coh", S0_coh)
    exp.save_derived("farfield_cbs/S1_coh", S1_coh)
    exp.save_derived("farfield_cbs/S2_coh", S2_coh)
    exp.save_derived("farfield_cbs/S3_coh", S3_coh)
    exp.save_derived("farfield_cbs/S0_inc", S0_inc)
    exp.save_derived("farfield_cbs/S1_inc", S1_inc)
    exp.save_derived("farfield_cbs/S2_inc", S2_inc)
    exp.save_derived("farfield_cbs/S3_inc", S3_inc)



