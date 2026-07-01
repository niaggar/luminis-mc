"""
Validacion CBS -- corrida HOMOGENEA de referencia (una sola especie).

Genera el dataset de referencia contra el que se compara la capa de mezcla
(sim_mixture.py). Estructura IDENTICA a sim.py, con dos cambios deliberados:
  - UNA sola corrida (un solo radio), no un barrido de radios.
  - Grilla angular, numero de fotones y SEMILLA fijos y COMPARTIDOS con
    sim_mixture.py, para que ambos datasets caigan sobre el MISMO eje theta y
    la comparacion en fig_validation.py sea directa (resta punto a punto).

El par (sim_homogeneous.py, sim_mixture.py) implementa la prueba de equivalencia:

    capa HomogeneousLayer de densidad n
        ==
    MixtureLayer de DOS especies identicas de densidad n/2 cada una.

Si el transporte de la mezcla esta bien (mu_s_total = sum_i n_i sigma_i, seleccion
de especie ~ mu_s^(i), reconstruccion del campo reverso con first/last_scatter_medium),
la curva de realce co-polarizada debe coincidir con la homogenea dentro del ruido
Monte Carlo. La igualdad es ESTADISTICA, no bit a bit: la mezcla consume un draw
extra del rng por evento (seleccion de especie), asi que los streams divergen.

Integrado en tiempo (dt = 0), RGD, metodo del estimador. Polarizacion LINEAL a lo
largo de m (co-pol); el canal co-pol se extrae luego como (S0 + S1).
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

# ===========================================================================
# Salida
# ===========================================================================
EXP_NAME = "cbs_validation_homogeneous"
BASE_DIR = "/Users/niaggar/Documents/Thesis/tests"   # <-- ajustar a tu entorno

sweep = SweepManager(EXP_NAME, BASE_DIR, timestamped=False)
sweep.snapshot_master_script(__main__.__file__)
sweep.log_readme(
    "CBS validation -- HOMOGENEOUS reference (single species, linear pol, estimator). "
    "Reference dataset for the homogeneous-vs-mixture equivalence test."
)

# ===========================================================================
# Parametros fisicos
#   >>> ESTOS VALORES DEBEN SER IDENTICOS EN sim_mixture.py <<<
# ===========================================================================
RADIUS = 0.175                 # um  (fuera de RGD estricto; test de implementacion)
VOLUME_FRACTION = 0.10
N_PARTICLE = 1.59
N_MEDIUM = 1.33
WAVELENGTH = 0.514             # um
MU_A_PERCENT = 0.0             # sin absorcion (gate de reciprocidad limpio)

# Laser: polarizacion LINEAL a lo largo de m (X), incidencia normal.
LASER_M = 1.0
LASER_N = 0.0
LASER_RADIUS_MFP = 4.0         # en unidades de l_s
LASER_TYPE = LaserSource.Gaussian

# Funcion de fase
PHASEF_THETA_MIN = 0.0
PHASEF_THETA_MAX = np.pi
PHASEF_NDIV = 1_000

# ---------------------------------------------------------------------------
# Grilla angular del sensor de campo lejano
#   >>> IDENTICA EN sim_mixture.py <<<
# n_phi > 1 es OBLIGATORIO para el caso lineal (resuelve cos2phi -> X/Y scan).
# theta_max se adapta al cono de esta particula: cada cono ocupa n_theta bins.
# ---------------------------------------------------------------------------
N_THETA = 500
N_PHI = 1
PHI_MAX = 2 * np.pi
THETA_CONE_FACTOR = 10.0       # theta_max = K * theta_coherent

T_MAX = 0                      # una sola ventana temporal -> integrado en tiempo
D_TIME = 0

# ---------------------------------------------------------------------------
# Muestreo   >>> IDENTICO EN sim_mixture.py <<<
# ---------------------------------------------------------------------------
N_PHOTONS = 100_000            # 1e3 = humo; 1e5 para produccion limpia (estimador)
N_THREADS = 7
SEED = 12345                   # semilla fija COMPARTIDA con sim_mixture.py

# Sensor analogo paralelo (estimator=False) para la compuerta de convergencia.
RUN_ANALOG_CHECK = False


# ===========================================================================
# Corrida
# ===========================================================================
def run_homogeneous_simulation(exp: Experiment):
    """Construye una capa homogenea, corre CBS y persiste los resultados."""
    # --- sistema fisico (imperativo) ---
    phase = RayleighDebyeEMCPhaseFunction(
        WAVELENGTH, RADIUS, N_PARTICLE, N_MEDIUM,
        PHASEF_NDIV, PHASEF_THETA_MIN, PHASEF_THETA_MAX,
    )
    medium = RGDMedium(phase, RADIUS, N_PARTICLE, N_MEDIUM, WAVELENGTH)
    sample = Sample(N_MEDIUM)
    sample.add_layer(medium, 0.0, float("inf"))

    # --- fisica derivada (Q_sca, g, l_s, l*, theta_coh, ...) ---
    dq = derived_quantities(medium, VOLUME_FRACTION)
    mean_free_path = dq["mean_free_path"]
    inv_mfp = 1.0 / mean_free_path
    mu_absortion = MU_A_PERCENT * inv_mfp
    mu_scattering = inv_mfp - mu_absortion

    # set_mean_free_path es OBLIGATORIO (sample_free_path usa el campo mfp, no mu_s).
    medium.set_mean_free_path(mean_free_path)
    medium.set_scattering_coefficient(mu_scattering)
    medium.set_absorption_coefficient(mu_absortion)

    laser = Laser(
        LASER_M, LASER_N, WAVELENGTH,
        LASER_RADIUS_MFP * mean_free_path, LASER_TYPE,
    )

    # --- grilla angular adaptada al cono (misma receta que sim_mixture.py) ---
    theta_coherent = dq["theta_coherent"]
    theta_max = THETA_CONE_FACTOR * theta_coherent
    d_theta = theta_max / N_THETA
    d_phi = PHI_MAX / N_PHI

    print("---- Parametros de la simulacion (HOMOGENEA) ----")
    print(f"radio: {RADIUS:.3f} um")
    print(f"eficiencia de scattering: {dq['scattering_efficiency']:.3f}")
    print(f"camino libre medio l_s: {mean_free_path:.4f} um")
    print(f"camino de transporte l*: {dq['transport_mean_free_path']:.4f} um")
    print(f"factor de anisotropia g: {dq['anisotropy_g']:.4f}")
    print(f"theta_coherent: {theta_coherent * 1e3:.4f} mrad")
    print(f"theta_max (K={THETA_CONE_FACTOR:g}): {theta_max * 1e3:.4f} mrad")
    print(f"d_theta: {d_theta * 1e3:.5f} mrad")

    # --- sensores ---
    sens = SensorsGroup()
    det = sens.add_detector(
        FarFieldCBSSensor(theta_max, PHI_MAX, T_MAX, d_theta, d_phi, D_TIME, True)
    )
    det.set_theta_limit(0, theta_max)

    det_analog = None
    if RUN_ANALOG_CHECK:
        det_analog = sens.add_detector(
            FarFieldCBSSensor(theta_max, PHI_MAX, T_MAX, d_theta, d_phi, D_TIME, False)
        )
        det_analog.set_theta_limit(0, theta_max)

    stats = sens.add_detector(StatisticsSensor(z=0, absorb=True))
    stats.set_theta_limit(0, theta_max)

    # --- config ---
    config = SimConfig()
    config.n_photons = N_PHOTONS
    config.sample = sample
    config.detector = sens
    config.laser = laser
    config.track_reverse_paths = True          # <-- imprescindible para CBS
    config.pin_threads_to_cores = True
    config.n_threads = N_THREADS
    config.seed = SEED                         # semilla fija (comparacion homo vs mezcla)
    config.show_progress = True

    # --- params: derivados + info de grilla (para reconstruir el eje en postproceso) ---
    # Se fuerzan claves CANONICAS que fig_validation.py lee sin ambiguedad,
    # independientes de como derived_quantities nombre sus campos.
    extra = {
        **dq,
        "radius": RADIUS,
        "volume_fraction": VOLUME_FRACTION,
        "mean_free_path": mean_free_path,
        "transport_mean_free_path": dq["transport_mean_free_path"],
        "theta_coherent": theta_coherent,
        "theta_max": theta_max,
        "d_theta": d_theta,
        "d_phi": d_phi,
        "n_theta": N_THETA,
        "n_phi": N_PHI,
        "theta_cone_factor": THETA_CONE_FACTOR,
        "polarization": "linear",
        "layer_kind": "homogeneous",
    }
    exp.save_params(config, extra=extra)

    # --- run ---
    t0 = time.time()
    run_simulation_parallel(config)
    print("runtime_s:", time.time() - t0)
    print("hits:", det.hits)

    # --- guardar RAW + procesados ---
    sensors_to_save = {"farfield_cbs": det, "statistics": stats}
    if det_analog is not None:
        sensors_to_save["farfield_cbs_analog"] = det_analog
    exp.save_sensors(sensors_to_save)

    cbs = postprocess_farfield_cbs(det, N_PHOTONS)
    exp.save_processed("farfield_cbs", cbs, sensor=det)

    if det_analog is not None:
        cbs_analog = postprocess_farfield_cbs(det_analog, N_PHOTONS)
        exp.save_processed("farfield_cbs_analog", cbs_analog, sensor=det_analog)


# ===========================================================================
# Una sola corrida (sweep de un elemento -> load_sweep lo lee igual)
# ===========================================================================
run_name = f"homogeneous_r_{RADIUS:.3f}"
print(f"Running HOMOGENEOUS CBS reference for radius={RADIUS:.3f}")
sweep.run(0, run_name, lambda exp: run_homogeneous_simulation(exp))