"""
Prueba de la deteccion de CBS con camino forward + reverse.

Este script ejecuta una simulacion minima de retrodispersion coherente (CBS)
usando un unico `FarFieldCBSSensor`. Sirve para validar la nueva implementacion
de `detector.cpp` donde, para cada foton detectado, se calcula:

    - el campo del camino directo  (process_hit)
    - el campo del camino reverso  (reverse_field, algoritmo de 3 etapas)

y se acumulan por separado las contribuciones coherente (interferencia
forward+reverse) e incoherente. El cociente coherente/incoherente debe dar el
factor de realce (~2 en theta=0 para el canal helicidad-conservada).

El resultado se guarda con `SweepManager` siguiendo la misma estructura que el
resto de `results/`, de modo que `plots/figs_cbs_test.py` lo pueda leer.
"""

import __main__
import time
import numpy as np

from luminis_mc import (
    SweepManager,
    Laser, RGDMedium, Sample, FarFieldCBSSensor, StatisticsSensor, SensorsGroup,
    SimConfig, RayleighDebyeEMCPhaseFunction,
    run_simulation_parallel, postprocess_farfield_cbs,
    set_log_level, LogLevel, LaserSource,
)


set_log_level(LogLevel.info)

# ---------------------------------------------------------------------------
# Salida
# ---------------------------------------------------------------------------
exp_name = "single_events_study"
base_dir = "/home/niaggar/Developer/luminis-mc/temporal_results"

sweep = SweepManager(exp_name, base_dir, timestamped=False)
sweep.snapshot_master_script(__main__.__file__)
sweep.log_readme(
    "Test of the CBS profile comparing the full calcualtion (forward + reverse paths) with the estimator approach."
)

# ---------------------------------------------------------------------------
# Parametros fisicos (un solo caso, sencillo de interpretar)
# ---------------------------------------------------------------------------
params_sweep = [
    {"radius": 0.110, "volume_fraction": 1},
]

n_particle = 1.59
n_medium = 1.33
mu_absortion_percent = 0.0
wavelength = 0.514

# Laser: polarizacion circular (helicidad), incidencia normal
laser_m_polarization_state = 1 / np.sqrt(2)
laser_n_polarization_state = -1j / np.sqrt(2)
laser_radius = 4.0
laser_type = LaserSource.Gaussian

# Funcion de fase
phasef_theta_min = 0.0
phasef_theta_max = np.pi
phasef_ndiv = 100_000

# Numero de fotones (modesto: es una prueba, no una corrida de produccion)
n_photons_estimator = 100_000

# ---------------------------------------------------------------------------
# Sensor de campo lejano (cono CBS)
# ---------------------------------------------------------------------------
theta_max_far_field = np.deg2rad(10)
phi_max_far_field = 2 * np.pi
n_theta_far_field = 300
n_phi_far_field = 1
d_theta = theta_max_far_field / n_theta_far_field
d_phi = phi_max_far_field / n_phi_far_field

# Una sola ventana temporal -> integrado en el tiempo
t_max = 0.0
d_time = 0.0


events = [2, 3, 4, 5, 10, 15, 20, 30, 50, 100, 150, 200, 300, 500, 1000, 1001]





def run_estimator_simulation(exp, radius, volume_fraction):
    phase = RayleighDebyeEMCPhaseFunction(
        wavelength, radius, n_particle, n_medium,
        phasef_ndiv, phasef_theta_min, phasef_theta_max,
    )
    medium = RGDMedium(phase, radius, n_particle, n_medium, wavelength)
    sample = Sample(n_medium)
    sample.add_layer(medium, 0.0, float("inf"))

    scattering_efficiency = phase.scattering_efficiency()
    mean_free_path = (4.0 * radius) / (3.0 * volume_fraction * scattering_efficiency)
    inv_mean_free_path = 1.0 / mean_free_path
    mu_absortion = mu_absortion_percent * inv_mean_free_path
    mu_scattering = inv_mean_free_path - mu_absortion

    medium.set_mean_free_path(mean_free_path)
    medium.set_scattering_coefficient(mu_scattering)
    medium.set_absorption_coefficient(mu_absortion)

    laser = Laser(
        laser_m_polarization_state, laser_n_polarization_state,
        wavelength, laser_radius * mean_free_path, laser_type,
    )

    anysotropy = phase.get_anisotropy_factor()[0]
    transport_mean_free_path = mean_free_path / (1.0 - anysotropy)
    m_relative = n_particle / n_medium
    size_parameter = 2 * np.pi * radius * n_medium / wavelength
    k_medium = 2 * np.pi * n_medium / wavelength
    theta_coherent = 1.0 / (k_medium * transport_mean_free_path)  # ancho del cono (rad)

    print("---- Parametros de la simulacion -----")
    print(f"eficiencia de scattering: {scattering_efficiency:.3f}")
    print(f"camino libre medio l_s: {mean_free_path:.3f} um")
    print(f"camino de transporte l*: {transport_mean_free_path:.3f} um")
    print(f"factor de anisotropia g: {anysotropy:.4f}")
    print(f"ancho aprox del cono CBS: {np.rad2deg(theta_coherent):.3f} deg")

    # --- sensores ---
    sens = SensorsGroup()
    detectors = []
    for event in events:
        det = sens.add_detector(
            FarFieldCBSSensor(theta_max_far_field, phi_max_far_field, t_max, d_theta, d_phi, d_time, True)
        )
        det.set_theta_limit(0, theta_max_far_field)
        det.set_events_limit(event, event)
        detectors.append(det)

    stats = sens.add_detector(StatisticsSensor(z=0, absorb=True))
    stats.set_theta_limit(0, theta_max_far_field)

    # --- config ---
    config = SimConfig()
    config.n_photons = n_photons_estimator
    config.sample = sample
    config.detector = sens
    config.laser = laser
    config.track_reverse_paths = True          # <-- imprescindible para CBS
    config.pin_threads_to_cores = True
    config.n_threads = 15
    config.show_progress = True

    # --- params guardados ---
    exp.log_params(
        radius_um=radius,
        volume_fraction=volume_fraction,
        n_particle=n_particle,
        n_medium=n_medium,
        m_relative=m_relative,
        scattering_efficiency=scattering_efficiency,
        mu_scattering_um_inv=mu_scattering,
        mu_absortion_um_inv=mu_absortion,
        anisotropy_factor=anysotropy,
        size_parameter=size_parameter,
        mean_free_path_ls_um=mean_free_path,
        transport_mean_free_path_lstar_um=transport_mean_free_path,
        theta_coherent_rad=theta_coherent,
        wavelength_um=wavelength,
        laser_m_polarization_state=str(laser_m_polarization_state),
        laser_n_polarization_state=str(laser_n_polarization_state),
        n_photons=n_photons_estimator,
        theta_max_rad=theta_max_far_field,
        n_theta=n_theta_far_field,
        n_phi=n_phi_far_field,
    )

    # --- run ---
    t0 = time.time()
    run_simulation_parallel(config)
    print("runtime_s:", time.time() - t0)
    for det in detectors:
        print("hits:", det.hits)

    # --- guardar RAW + derivados ---
    for det in detectors:
        exp.save_sensor(det, "farfield_cbs")
    exp.save_sensor(stats, "statistics")

    for e, det in enumerate(detectors):
        cbs = postprocess_farfield_cbs(det, n_photons_estimator)
    
        theta = np.linspace(0, det.theta_max, det.N_theta)
        phi = np.linspace(0, det.phi_max, det.N_phi)
        exp.save_derived("axes/theta_rad", theta)
        exp.save_derived("axes/phi_rad", phi)

        # Una sola ventana temporal (t = 0): guardamos los mapas 2D (theta, phi).
        coh = cbs.coherent[0]
        inc = cbs.incoherent[0]
        exp.save_derived(f"farfield_cbs_{e}/coherent/s0", np.array(coh.S0, copy=False))
        exp.save_derived(f"farfield_cbs_{e}/coherent/s1", np.array(coh.S1, copy=False))
        exp.save_derived(f"farfield_cbs_{e}/coherent/s2", np.array(coh.S2, copy=False))
        exp.save_derived(f"farfield_cbs_{e}/coherent/s3", np.array(coh.S3, copy=False))
        exp.save_derived(f"farfield_cbs_{e}/incoherent/s0", np.array(inc.S0, copy=False))
        exp.save_derived(f"farfield_cbs_{e}/incoherent/s1", np.array(inc.S1, copy=False))
        exp.save_derived(f"farfield_cbs_{e}/incoherent/s2", np.array(inc.S2, copy=False))
        exp.save_derived(f"farfield_cbs_{e}/incoherent/s3", np.array(inc.S3, copy=False))



for i, data in enumerate(params_sweep):
    radius = data["radius"]
    volume_fraction = data["volume_fraction"]
    run_name = f"radius_{radius:.3f}_volumefraction_{volume_fraction:.3f}"

    fun_estimator = lambda exp, r=radius, v=volume_fraction: run_estimator_simulation(exp, r, v)
    
    print(f"Running CBS test for radius={radius:.3f}, f={volume_fraction:.2f}")
    print("--------- Running estimator simulation")
    sweep.run(i, run_name + "_estimator", fun_estimator)
