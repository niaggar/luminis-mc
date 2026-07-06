import __main__
import time
import numpy as np

from luminis_mc import (
    Experiment,
    SweepManager,
    Laser, RGDMedium, Sample, FarFieldCBSSensor, StatisticsSensor, SensorsGroup,
    SimConfig, RayleighDebyeEMCPhaseFunction,
    run_simulation_parallel, postprocess_farfield_cbs,
    derived_quantities, derived_quantities_mixture,
    set_log_level, LogLevel, LaserSource,
    MixtureLayer
)


set_log_level(LogLevel.info)

# ===========================================================================
# Salida
# ===========================================================================
EXP_NAME = "study_mixture_layer_TWO_LAYERS_REV__PLIN"
BASE_DIR = "/home/niaggar/Developer/luminis-mc/temporal_results"

sweep = SweepManager(EXP_NAME, BASE_DIR, timestamped=False)
sweep.snapshot_master_script(__main__.__file__)
sweep.log_readme(
    "CBS validation -- MIXTURE of two IDENTICAL species (linear pol, estimator). "
    "Equivalence test against sim_homogeneous.py: mixture(n/2,n/2) == homogeneous(n)."
)

# ===========================================================================
# Parametros fisicos
#   >>> IDENTICOS A sim_homogeneous.py <<<
# ===========================================================================
RADIUS_1 = 0.075                 # um
RADIUS_2 = 0.035                 # um
VOLUME_FRACTION = 0.10
N_PARTICLE = 1.59
N_MEDIUM = 1.33
WAVELENGTH = 0.514             # um
MU_A_PERCENT = 0.0             # sin absorcion (gate de reciprocidad limpio)

thickness_multipliers = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]

# Laser: polarizacion LINEAL a lo largo de m (X), incidencia normal.
LASER_M = 1.0
LASER_N = 0.0
LASER_RADIUS_MFP = 4.0         # en unidades de l_s
LASER_TYPE = LaserSource.Gaussian

# Funcion de fase
PHASEF_THETA_MIN = 0.0
PHASEF_THETA_MAX = np.pi
PHASEF_NDIV = 10_000

# ---------------------------------------------------------------------------
# Grilla angular   >>> IDENTICA A sim_homogeneous.py <<<
# ---------------------------------------------------------------------------
N_THETA = 1000
N_PHI = 36
PHI_MAX = 2 * np.pi
THETA_MAX = np.deg2rad(1)

T_MAX_MULTI = 30
N_TIME_BINS = 500

# ---------------------------------------------------------------------------
# Muestreo   >>> IDENTICO A sim_homogeneous.py <<<
# ---------------------------------------------------------------------------
N_PHOTONS = 100_000
N_THREADS = 44


# ===========================================================================
# Helpers
# ===========================================================================
def build_species(rad):
    """Una especie RGD con funcion de fase EMC (seccion eficaz NO nula)."""
    phase = RayleighDebyeEMCPhaseFunction(
        WAVELENGTH, rad, N_PARTICLE, N_MEDIUM,
        PHASEF_NDIV, PHASEF_THETA_MIN, PHASEF_THETA_MAX,
    )
    return RGDMedium(phase, rad, N_PARTICLE, N_MEDIUM, WAVELENGTH)


def set_albedo(medium, mu_a_percent, mfp):
    """Fija un albedo bien definido para que la agregacion mu_a^(i) no caiga en 0/0.

    La escala absoluta de mu_s aqui NO afecta el transporte (la mu_s total la fija
    n_i sigma_i); solo importa el cociente mu_a/mu_s. Absorcion primero, luego
    scattering (recomputa mu_t).
    """
    mu_s = 1.0 - mu_a_percent
    mu_a = mu_a_percent
    medium.set_absorption_coefficient(mu_a)
    medium.set_scattering_coefficient(mu_s)
    medium.set_mean_free_path(mfp)


# ===========================================================================
# Corrida
# ===========================================================================
def run_mixture_simulation(exp: Experiment, thickness_multi: float):
    """Construye una MixtureLayer de dos especies, corre CBS y persiste."""

    especie_1 = build_species(RADIUS_1)
    especie_2 = build_species(RADIUS_2)
    dq_1 = derived_quantities(especie_1, VOLUME_FRACTION)
    dq_2 = derived_quantities(especie_2, VOLUME_FRACTION)

    set_albedo(especie_1, MU_A_PERCENT, dq_1['mean_free_path'])
    set_albedo(especie_2, MU_A_PERCENT, dq_2['mean_free_path'])

    layer_1_thickness = thickness_multi * dq_1['transport_mean_free_path']

    sample = Sample(N_MEDIUM)
    sample.add_layer(especie_1, 0, layer_1_thickness)
    sample.add_layer(especie_2, layer_1_thickness, float('inf'))


    print("---- Parametros de la simulacion (MEZCLA) ----")
    print(f"radio 1: {RADIUS_1:.3f} um   radio 2: {RADIUS_2:.3f} um")
    print(f"eficiencia de scattering 1: {dq_1['scattering_efficiency']:.3f}")
    print(f"eficiencia de scattering 2: {dq_2['scattering_efficiency']:.3f}")

    print(f"camino libre medio l_s 1: {dq_1['mean_free_path']:.4f} um")
    print(f"camino libre medio l_s 2: {dq_2['mean_free_path']:.4f} um")
    
    print(f"camino de transporte l* 1: {dq_1['transport_mean_free_path']:.4f} um")
    print(f"camino de transporte l* 2: {dq_2['transport_mean_free_path']:.4f} um")

    print(f"factor de anisotropia g 1: {dq_1['anisotropy_g']:.4f}")
    print(f"factor de anisotropia g 2: {dq_2['anisotropy_g']:.4f}")
    
    print(f"theta_coherent 1: {dq_1['theta_coherent'] * 1e3:.4f} mrad")
    print(f"theta_coherent 2: {dq_2['theta_coherent'] * 1e3:.4f} mrad")

    laser = Laser(
        LASER_M, LASER_N, WAVELENGTH,
        LASER_RADIUS_MFP * dq_1['transport_mean_free_path'], LASER_TYPE,
    )

    # --- grilla angular (misma receta y mismos numeros que la homogenea) ---
    d_theta = THETA_MAX / N_THETA
    d_phi = PHI_MAX / N_PHI
    t_max = T_MAX_MULTI * dq_1['transport_mean_free_path']
    dt = t_max / N_TIME_BINS
    
    # --- sensores ---
    sens = SensorsGroup()
    det = sens.add_detector(
        FarFieldCBSSensor(THETA_MAX, PHI_MAX, t_max, d_theta, d_phi, dt, True)
    )
    det.set_theta_limit(0, THETA_MAX)
    # det.set_phi_slices([0, np.pi/4, np.pi/2, 3*np.pi/4])

    stats = sens.add_detector(StatisticsSensor(z=0, absorb=True))
    stats.set_theta_limit(0, THETA_MAX)

    # --- config ---
    config = SimConfig()
    config.n_photons = N_PHOTONS
    config.sample = sample
    config.detector = sens
    config.laser = laser
    config.track_reverse_paths = True          # <-- imprescindible para CBS
    config.pin_threads_to_cores = False
    config.n_threads = N_THREADS
    config.show_progress = True

    extra = {
        "dq_1": dq_1,
        "dq_2": dq_2,
        "thickness_multi": thickness_multi,
        "theta_max": THETA_MAX,
        "d_theta": d_theta,
        "d_phi": d_phi,
        "n_theta": N_THETA,
        "n_phi": N_PHI,
        "t_max": t_max,
        "d_time": dt,
        "polarization": "linear",
    }

    # NOTA: save_params auto-captura la MixtureLayer via capture_params (§14).
    # Si esto lanzara sobre la mezcla, el problema esta en ese path (vars(medium)
    # sobre MixtureLayer), no en la fisica de la simulacion.
    exp.save_params(config, extra=extra)

    # --- run ---
    t0 = time.time()
    run_simulation_parallel(config)
    print("runtime_s:", time.time() - t0)
    print("hits:", det.hits)

    # --- guardar RAW + procesados ---
    sensors_to_save = {"farfield_cbs": det, "statistics": stats}
    exp.save_sensors(sensors_to_save)

    cbs = postprocess_farfield_cbs(det, N_PHOTONS)
    exp.save_processed("farfield_cbs", cbs, sensor=det)


    # keep-alive explicito: referenciar species/sample al final del scope asegura
    # que sobreviven a toda la corrida (el binding guarda punteros crudos).
    _keep_alive = (sample, especie_1, especie_2)
    del _keep_alive


# ===========================================================================
# Una sola corrida (sweep de un elemento -> load_sweep lo lee igual)
# ===========================================================================
for index, mult in enumerate(thickness_multipliers):
    run_name = f"multiplier_{mult:.2f}"

    print(f"\n\n=== Corrida: {run_name} ===\n")
    sweep.run(index, run_name, lambda exp, m=mult: run_mixture_simulation(exp, thickness_multi=m))