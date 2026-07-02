import __main__
import time
import numpy as np

from luminis_mc import (
    Experiment,
    SweepManager,
    Laser, RGDMedium, Sample, FarFieldCBSSensor, StatisticsSensor, SensorsGroup,
    SimConfig, RayleighDebyeEMCPhaseFunction,
    run_simulation_parallel, postprocess_farfield_cbs, derived_quantities,
    set_log_level, LogLevel, LaserSource,
)


set_log_level(LogLevel.info)

# ---------------------------------------------------------------------------
# Salida
# ---------------------------------------------------------------------------
exp_name = "test_slices_grid"
base_dir = "/Users/niaggar/Documents/Thesis/tests"

sweep = SweepManager(exp_name, base_dir, timestamped=False)
sweep.snapshot_master_script(__main__.__file__)
sweep.log_readme(
    "CBS validation (linear polarization, estimator) -- Iwai et al. (1995) Fig. 2(a)."
)

# ---------------------------------------------------------------------------
# Parametros fisicos
# ---------------------------------------------------------------------------
params_sweep = [
    {"radius": 0.175, "volume_fraction": 0.10},
]

n_particle = 1.59
n_medium = 1.33
mu_absortion_percent = 0.0
wavelength = 0.514

# Laser: polarizacion LINEAL a lo largo de m (X), incidencia normal.
# laser_m_polarization_state = 1.0
# laser_n_polarization_state = 0.0
laser_m_polarization_state = 1 / np.sqrt(2)
laser_n_polarization_state = -1j / np.sqrt(2)
laser_radius = 4.0                 # en unidades de l_s; ver nota sobre el haz
laser_type = LaserSource.Gaussian

# Funcion de fase
phasef_theta_min = 0.0
phasef_theta_max = np.pi
phasef_ndiv = 1_000

# Numero de fotones (1e3 = humo; subir a 1e4-1e5 para produccion limpia)
n_photons_estimator = 10_000

# ---------------------------------------------------------------------------
# Grilla angular del sensor de campo lejano
# ---------------------------------------------------------------------------
N_THETA = 500                    # fijo; theta_max se adapta por particula
N_PHI = 36                         # resuelve cos2phi (X-scan / Y-scan)
PHI_MAX = 2 * np.pi
THETA_MAX = np.deg2rad(1.0)
DTHETA = THETA_MAX / N_THETA
DPHI = PHI_MAX / N_PHI

# Una sola ventana temporal -> integrado en el tiempo
t_max = 0
d_time = 0


def run_estimator_simulation(exp: Experiment, *, radius, volume_fraction):
    """Construye el sistema, ejecuta la simulacion CBS y guarda los resultados."""
    # --- sistema fisico (imperativo) ---
    phase = RayleighDebyeEMCPhaseFunction(
        wavelength, radius, n_particle, n_medium,
        phasef_ndiv, phasef_theta_min, phasef_theta_max,
    )
    medium = RGDMedium(phase, radius, n_particle, n_medium, wavelength)
    sample = Sample(n_medium)
    sample.add_layer(medium, 0.0, float("inf"))

    dq = derived_quantities(medium, volume_fraction)   # Q_sca, g, l_s, l*, theta_coh, ...
    mean_free_path = dq["mean_free_path"]
    inv_mfp = 1.0 / mean_free_path
    mu_absortion = mu_absortion_percent * inv_mfp
    mu_scattering = inv_mfp - mu_absortion

    medium.set_mean_free_path(mean_free_path)
    medium.set_scattering_coefficient(mu_scattering)
    medium.set_absorption_coefficient(mu_absortion)

    laser = Laser(
        laser_m_polarization_state, laser_n_polarization_state,
        wavelength, laser_radius * mean_free_path, laser_type,
    )

    # --- grilla angular adaptada al cono de esta particula ---
    theta_coherent = dq["theta_coherent"]


    print("---- Parametros de la simulacion -----")
    print(f"radio: {radius:.3f} um")
    print(f"eficiencia de scattering: {dq['scattering_efficiency']:.3f}")
    print(f"camino libre medio l_s: {mean_free_path:.3f} um")
    print(f"camino de transporte l*: {dq['transport_mean_free_path']:.3f} um")
    print(f"factor de anisotropia g: {dq['anisotropy_g']:.4f}")
    print(f"theta_coherent: {theta_coherent * 1e3:.4f} mrad")

    # --- sensores ---
    sens = SensorsGroup()
    det = sens.add_detector(
        FarFieldCBSSensor(THETA_MAX, PHI_MAX, t_max, DTHETA, DPHI, d_time, True)
    )
    det.set_theta_limit(0, THETA_MAX)
    det.set_phi_slices([0, np.pi/4])

    stats = sens.add_detector(StatisticsSensor(z=0, absorb=True))
    stats.set_theta_limit(0, THETA_MAX)

    # --- config ---
    config = SimConfig()
    config.n_photons = n_photons_estimator
    config.sample = sample
    config.detector = sens
    config.laser = laser
    config.track_reverse_paths = True          # <-- imprescindible para CBS
    config.pin_threads_to_cores = False
    config.n_threads = 7
    config.show_progress = True

    # --- params: derivados + info de grilla (para reconstruir el eje en postproceso) ---
    extra = {
        **dq,
        "theta_max": THETA_MAX,
        "d_theta": DTHETA,
        "d_phi": DPHI,
        "n_theta": N_THETA,
        "n_phi": N_PHI,
        "polarization": "linear",
    }
    exp.save_params(config, extra=extra)

    # --- run ---
    t0 = time.time()
    run_simulation_parallel(config)
    print("runtime_s:", time.time() - t0)
    print("hits:", det.hits)

    # --- guardar RAW + procesados (estandarizado) ---
    sensors_to_save = {"farfield_cbs": det, "statistics": stats}
    exp.save_sensors(sensors_to_save)

    cbs = postprocess_farfield_cbs(det, n_photons_estimator)
    exp.save_processed("farfield_cbs", cbs, sensor=det)


for i, data in enumerate(params_sweep):
    radius = data["radius"]
    volume_fraction = data["volume_fraction"]
    run_name = f"r_{radius:.3f}"

    print(f"Running CBS test for radius={radius:.3f}")
    sweep.run(i, run_name + "_estimator", lambda exp, r=radius, v=volume_fraction: run_estimator_simulation(exp, radius=r, volume_fraction=v))