import __main__
import time
import numpy as np

from luminis_mc import (
    Experiment,
    SweepManager,
    Laser, RGDMedium, Sample, FarFieldCBSSensor, StatisticsSensor, SensorsGroup,
    SimConfig, RayleighDebyeEMCPhaseFunction,
    run_simulation_parallel, postprocess_farfield_cbs,
    derived_quantities,
    set_log_level, LogLevel, LaserSource,
)

set_log_level(LogLevel.info)

# ===========================================================================
# Salida
# ===========================================================================
EXP_NAME = "study_estimator_vs_analog"
BASE_DIR = "/home/niaggar/Developer/luminis-mc/temporal_results"

sweep = SweepManager(EXP_NAME, BASE_DIR, timestamped=False)
sweep.snapshot_master_script(__main__.__file__)
sweep.log_readme(
    "Estimator vs analog convergence study (F15/F16, Ch. 4). Homogeneous RGD "
    "layer, linear pol. R independent replicas per (mode, N) with distinct "
    "seeds -> sigma_rel at the cone peak vs N (F15) and merged top-N analog "
    "reference vs converged estimator profile (F16). Wall time persisted per "
    "run for the efficiency factor eps = 1/(sigma^2 * T)."
)

# ===========================================================================
# Parametros fisicos
# ===========================================================================
RADIUS_PRIMARY = [0.175]           # radio primario para la figura de Cap. 4
RADIUS_OPTIONAL = [0.035]          # correr solo si sobra tiempo
radius_values = RADIUS_PRIMARY     # + RADIUS_OPTIONAL

VOLUME_FRACTION = 0.10
N_PARTICLE = 1.59
N_MEDIUM = 1.33
WAVELENGTH = 0.514                 # um
MU_A_PERCENT = 0.0                 # sin absorcion (gate de reciprocidad limpio)

# Laser: polarizacion LINEAL a lo largo de m (X), incidencia normal.
LASER_M = 1.0
LASER_N = 0.0
LASER_RADIUS_MFP = 4.0             # en unidades de l*
LASER_TYPE = LaserSource.Gaussian

# Funcion de fase
PHASEF_THETA_MIN = 0.0
PHASEF_THETA_MAX = np.pi
PHASEF_NDIV = 10_000

# ---------------------------------------------------------------------------
# Grilla angular -- IDENTICA en ambos modos (requisito de F16)
# ---------------------------------------------------------------------------
N_THETA = 100                      # 1000 sobre-resuelve el cono e infla el
N_PHI = 36                         # ruido por bin en el pico
PHI_MAX = 2 * np.pi
THETA_MAX = np.deg2rad(1)

T_MAX_MULTI = 0
N_TIME_BINS = 0

# ---------------------------------------------------------------------------
# Muestreo: escaleras log-uniformes (media decada) + replicas independientes
# ---------------------------------------------------------------------------
N_THREADS = 44
SEED_BASE = 20260708

N_REPLICAS = 1

photons_estimator = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000]
photons_analog = [1_000_000, 3_000_000, 10_000_000, 30_000_000, 100_000_000]
# Nota F16: la referencia analogica se construye en el analisis fusionando las
# N_REPLICAS del punto mas alto (12 x 1e8 ~ 1.2e9 fotones efectivos, con banda
# de error de las replicas). No hace falta una corrida solitaria de 1e9.


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
    """Fija un albedo bien definido. Absorcion primero, luego scattering."""
    mu_s = 1.0 - mu_a_percent
    mu_a = mu_a_percent
    medium.set_absorption_coefficient(mu_a)
    medium.set_scattering_coefficient(mu_s)
    medium.set_mean_free_path(mfp)


# ===========================================================================
# Corrida
# ===========================================================================
def run_cbs(exp, radius, n_photons, rep, estimator: bool):
    especie = build_species(radius)
    dq = derived_quantities(especie, VOLUME_FRACTION)
    set_albedo(especie, MU_A_PERCENT, dq['mean_free_path'])

    sample = Sample(N_MEDIUM)
    sample.add_layer(especie, 0, float('inf'))

    laser = Laser(LASER_M, LASER_N, WAVELENGTH,
                  LASER_RADIUS_MFP * dq['transport_mean_free_path'], LASER_TYPE)

    d_theta = THETA_MAX / N_THETA
    d_phi = PHI_MAX / N_PHI
    t_max = T_MAX_MULTI * dq['transport_mean_free_path']
    dt = 0.0

    sens = SensorsGroup()
    det = sens.add_detector(
        FarFieldCBSSensor(THETA_MAX, PHI_MAX, t_max, d_theta, d_phi, dt, estimator)
    )
    det.set_theta_limit(0, THETA_MAX)
    stats = sens.add_detector(StatisticsSensor(z=0, absorb=True))
    stats.set_theta_limit(0, THETA_MAX)

    # Misma semilla para todos los N de una replica: streams anidados por
    # indice de foton (mix_seed) -> curvas de convergencia suaves por replica.
    # Semillas distintas ENTRE replicas -> sigma(N) valido.
    seed = SEED_BASE + 7919 * rep

    config = SimConfig()
    config.n_photons = n_photons
    config.seed = seed
    config.sample = sample
    config.detector = sens
    config.laser = laser
    config.track_reverse_paths = True
    config.pin_threads_to_cores = True   # Linux: affinity real, timing estable
    config.n_threads = N_THREADS
    config.show_progress = False         # menos ruido con muchas corridas

    t0 = time.time()
    run_simulation_parallel(config)
    runtime_s = time.time() - t0
    mode = "estimator" if estimator else "analog"
    print(f"mode: {mode} | rep: {rep} | n_photons: {n_photons} "
          f"| runtime_s: {runtime_s:.2f} | hits: {det.hits}")

    # save_params DESPUES de correr para persistir el tiempo de pared
    # (necesario para eps = 1/(sigma^2 * T)).
    extra = {
        **dq,
        "theta_max": THETA_MAX,
        "d_theta": d_theta,
        "d_phi": d_phi,
        "n_theta": N_THETA,
        "n_phi": N_PHI,
        "t_max": t_max,
        "d_time": dt,
        "polarization": "linear",
        "mode": mode,
        "rep": rep,
        "seed": seed,
        "runtime_s": runtime_s,
        "n_threads": N_THREADS,
    }
    exp.save_params(config, extra=extra)

    exp.save_sensors({"farfield_cbs": det, "statistics": stats})
    exp.save_processed("farfield_cbs", postprocess_farfield_cbs(det, n_photons), sensor=det)

    _keep_alive = (sample, especie); del _keep_alive


# Orden: replica externa, N interno ascendente. Si hay que abortar a mitad,
# quedan replicas COMPLETAS (utilizables) en vez de N sueltos.
for index, rad in enumerate(radius_values):
    for rep in range(N_REPLICAS):
        for n in photons_estimator:
            name = f"radius_{rad:.3f}_estimator_N{n}_rep{rep:02d}"
            sweep.run(index, name,
                      lambda exp, rad=rad, n=n, rep=rep: run_cbs(exp, rad, n, rep, estimator=True))
        for n in photons_analog:
            name = f"radius_{rad:.3f}_analog_N{n}_rep{rep:02d}"
            sweep.run(index, name,
                      lambda exp, rad=rad, n=n, rep=rep: run_cbs(exp, rad, n, rep, estimator=False))