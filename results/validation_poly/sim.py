"""
Validacion de CBS para particulas de poliestireno -- polarizacion LINEAL.

Reproduce la Fig. 2(a) de Iwai et al. (1995): componente co-polarizada del cono
de retrodispersion para varios radios de particula (RGD, integrado en tiempo,
metodo del estimador). El barrido tambien deja lista la Fig. 2(b) cross-pol y la
Fig. 3 (anisotropia X/Y) porque phi queda resuelto.

Decisiones de grilla (ver discusion):
- Polarizacion lineal a lo largo de m: (E_m, E_n) = (1, 0). El canal co-pol se
  extrae luego como (S0 + S1) y el cross-pol como (S0 - S1).
- Resolucion azimutal n_phi = 36 para resolver la anisotropia cos2phi
  (X-scan ~ phi=0, Y-scan ~ phi=pi/2). Un unico bin de 2pi promedia en azimut y
  mezcla los scans X e Y: NO sirve para el caso lineal.
- theta adaptado por particula: theta_max = K * theta_coherent, con n_theta
  fijo. Cada cono ocupa el mismo numero de bins -> resolucion constante por cono
  y grilla reducida comun q = k*l*_*theta para overlay/colapso.

Integrado en tiempo (dt = 0), RGD, para comparar apples-to-apples con Iwai.
"""

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
exp_name = "cbs_poly_validation_linear"
base_dir = "/Users/niaggar/Documents/Thesis/tests"

sweep = SweepManager(exp_name, base_dir, timestamped=False)
sweep.snapshot_master_script(__main__.__file__)
sweep.log_readme(
    "CBS validation (linear polarization, estimator) -- Iwai et al. (1995) Fig. 2(a)."
)

# ---------------------------------------------------------------------------
# Parametros fisicos
# ---------------------------------------------------------------------------
# r = 0.035 / 0.055 / 0.175 um son el triplete de Iwai (sus "diametros"
# 70/110/350 nm). r = 0.020 es la referencia cuasi-Rayleigh (Q_sca ~ 0).
# OJO: r = 0.175 cae fuera del RGD estricto (2ka|m-1| ~ 1.1); se conserva por
# paridad con Iwai (chequeo de implementacion, no de exactitud fisica).
params_sweep = [
    {"radius": 0.020, "volume_fraction": 0.10},
    {"radius": 0.035, "volume_fraction": 0.10},
    {"radius": 0.055, "volume_fraction": 0.10},
    {"radius": 0.075, "volume_fraction": 0.10},
    {"radius": 0.175, "volume_fraction": 0.10},
]

n_particle = 1.59
n_medium = 1.33
mu_absortion_percent = 0.0
wavelength = 0.514

# Laser: polarizacion LINEAL a lo largo de m (X), incidencia normal.
laser_m_polarization_state = 1.0
laser_n_polarization_state = 0.0
laser_radius = 4.0                 # en unidades de l_s; ver nota sobre el haz
laser_type = LaserSource.Gaussian

# Funcion de fase
phasef_theta_min = 0.0
phasef_theta_max = np.pi
phasef_ndiv = 1_000

# Numero de fotones (1e3 = humo; subir a 1e4-1e5 para produccion limpia)
n_photons_estimator = 10

# ---------------------------------------------------------------------------
# Grilla angular del sensor de campo lejano
# ---------------------------------------------------------------------------
N_THETA = 1_000                    # fijo; theta_max se adapta por particula
N_PHI = 36                         # resuelve cos2phi (X-scan / Y-scan)
PHI_MAX = 2 * np.pi
THETA_CONE_FACTOR = 10.0           # theta_max = K * theta_coherent

# Una sola ventana temporal -> integrado en el tiempo
t_max = 0
d_time = 0

# Sensor analogo paralelo (estimator=False) para la compuerta de convergencia.
# Falso en produccion para no duplicar costo; True para el chequeo de dos sensores.
RUN_ANALOG_CHECK = False


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
    theta_max = THETA_CONE_FACTOR * theta_coherent
    d_theta = theta_max / N_THETA
    d_phi = PHI_MAX / N_PHI


    print("---- Parametros de la simulacion -----")
    print(f"radio: {radius:.3f} um")
    print(f"eficiencia de scattering: {dq['scattering_efficiency']:.3f}")
    print(f"camino libre medio l_s: {mean_free_path:.3f} um")
    print(f"camino de transporte l*: {dq['transport_mean_free_path']:.3f} um")
    print(f"factor de anisotropia g: {dq['anisotropy_g']:.4f}")
    print(f"theta_coherent: {theta_coherent * 1e3:.4f} mrad")
    print(f"theta_max (K={THETA_CONE_FACTOR:g}): {theta_max * 1e3:.3f} mrad")
    print(f"d_theta: {d_theta * 1e3:.5f} mrad")

    # --- sensores ---
    sens = SensorsGroup()
    det = sens.add_detector(
        FarFieldCBSSensor(theta_max, PHI_MAX, t_max, d_theta, d_phi, d_time, True)
    )
    det.set_theta_limit(0, theta_max)

    det_analog = None
    if RUN_ANALOG_CHECK:
        det_analog = sens.add_detector(
            FarFieldCBSSensor(theta_max, PHI_MAX, t_max, d_theta, d_phi, d_time, False)
        )
        det_analog.set_theta_limit(0, theta_max)

    stats = sens.add_detector(StatisticsSensor(z=0, absorb=True))
    stats.set_theta_limit(0, theta_max)

    # --- config ---
    config = SimConfig()
    config.n_photons = n_photons_estimator
    config.sample = sample
    config.detector = sens
    config.laser = laser
    config.track_reverse_paths = True          # <-- imprescindible para CBS
    config.pin_threads_to_cores = True
    config.n_threads = 7
    config.show_progress = True

    # --- params: derivados + info de grilla (para reconstruir el eje en postproceso) ---
    extra = {
        **dq,
        "theta_max": theta_max,
        "d_theta": d_theta,
        "d_phi": d_phi,
        "n_theta": N_THETA,
        "n_phi": N_PHI,
        "theta_cone_factor": THETA_CONE_FACTOR,
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
    if det_analog is not None:
        sensors_to_save["farfield_cbs_analog"] = det_analog
    exp.save_sensors(sensors_to_save)

    cbs = postprocess_farfield_cbs(det, n_photons_estimator)
    exp.save_processed("farfield_cbs", cbs, sensor=det)

    if det_analog is not None:
        cbs_analog = postprocess_farfield_cbs(det_analog, n_photons_estimator)
        exp.save_processed("farfield_cbs_analog", cbs_analog, sensor=det_analog)


for i, data in enumerate(params_sweep):
    radius = data["radius"]
    volume_fraction = data["volume_fraction"]
    run_name = f"r_{radius:.3f}"

    print(f"Running CBS test for radius={radius:.3f}")
    sweep.run(i, run_name + "_estimator",
              lambda exp, r=radius, v=volume_fraction:
                  run_estimator_simulation(exp, radius=r, volume_fraction=v))