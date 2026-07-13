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
# QUE ES ESTE ESTUDIO
# ---------------------------------------------------------------------------
#   Validacion ESTIMATOR (next-event) vs ANALOG (Cap. 4, figuras F15/F16).
#   Capa RGD homogenea, integrado en tiempo (SIN eje temporal).
#     * F15 (convergencia/eficiencia): sigma_rel del pico E(0) vs N + factor de
#       eficiencia eps = 1/(sigma^2 * T) con el tiempo de pared persistido.
#     * F16 (insesgadez): perfil del estimador convergido vs referencia analoga
#       fusionada (top-N, N_REPLICAS) -> deben coincidir dentro de la banda.
#   REQUISITO CLAVE: la grilla angular es IDENTICA en ambos modos (misma ancla
#   l*, mismos Q, mismos bins) -> la unica diferencia es el modo de scoring.
# ===========================================================================

# ===========================================================================
# Salida
# ===========================================================================
EXP_NAME = "study_estimator_vs_analog"
BASE_DIR = "/home/niaggar/Developer/luminis-mc/temporal_results"

sweep = SweepManager(EXP_NAME, BASE_DIR, timestamped=False)
sweep.snapshot_master_script(__main__.__file__)

# ===========================================================================
# Parametros fisicos
# ===========================================================================
radius_values = [0.055]
VOLUME_FRACTION = 0.20
N_PARTICLE = 1.59
N_MEDIUM = 1.33
WAVELENGTH = 0.514                 # um
MU_A_PERCENT = 0.0                 # sin absorcion (gate de reciprocidad limpio)

# ---------------------------------------------------------------------------
# Laser: Jones (m, n). Etiqueta DERIVADA de los parametros (nunca hardcodeada).
#   Valores actuales = (1/sqrt2, i/sqrt2) -> polarizacion CIRCULAR (helicidad).
#   Para lineal a lo largo de X usar (m, n) = (1, 0).
# ---------------------------------------------------------------------------
LASER_M = 1.0 / np.sqrt(2)
LASER_N = 1j / np.sqrt(2)
LASER_RADIUS = 2500          # um
LASER_TYPE = LaserSource.Gaussian
POLARIZATION = "circular"

# Funcion de fase
PHASEF_THETA_MIN = 0.0
PHASEF_THETA_MAX = np.pi
PHASEF_NDIV = 10_000

# ---------------------------------------------------------------------------
# Grilla angular -- doble ventana en unidades reducidas q = k*l*_teorico*theta,
# anclada POR ESPECIE a l* (estilo §5.1/§5.3). IDENTICA en estimator y analog
# (requisito de F16). Resolucion MODERADA a proposito: el analog es ineficiente
# en retrodispersion (pocos hits cerca de theta=0), sobre-resolver inflaria el
# ruido por bin en el pico justo donde se mide sigma_rel; E(0) se extrae por
# ajuste parabolico, que no necesita malla fina.
# ---------------------------------------------------------------------------
N_THETA_1 = 200                    # ventana fina (cono)
N_THETA_2 = 100                    # ventana cola (fondo/baseline)
Q_FINE, Q_TAIL = 7.5, 40.0
N_PHI = 4
PHI_MAX = 2 * np.pi
k = 2 * np.pi * N_MEDIUM / WAVELENGTH

# ---------------------------------------------------------------------------
# SIN escala temporal: sensores integrados (bin unico). t_max = dt = 0.
# ---------------------------------------------------------------------------
T_MAX = 0.0
DT = 0.0

# ---------------------------------------------------------------------------
# Muestreo: escaleras log-uniformes (media decada) + replicas independientes
# ---------------------------------------------------------------------------
N_THREADS = 46
SEED_BASE = 20260708
N_REPLICAS = 5

photons_estimator = [1_000, 10_000, 30_000, 100_000, 300_000]
photons_analog = [1_000_000, 10_000_000, 50_000_000, 100_000_000, 500_000_000]
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
    medium.set_absorption_coefficient(mu_a_percent)
    medium.set_scattering_coefficient(1.0 - mu_a_percent)
    medium.set_mean_free_path(mfp)


def make_sensors(theta_1, theta_2, estimator):
    """Doble detector far-field (fino + cola) + estadistica, INTEGRADO en tiempo.
    El flag `estimator` (ultimo arg del sensor) selecciona next-event vs analog;
    va en AMBOS detectores para que el modo sea homogeneo en toda la corrida."""
    d_theta_1 = theta_1 / N_THETA_1
    d_theta_2 = theta_2 / N_THETA_2
    d_phi = PHI_MAX / N_PHI

    sens = SensorsGroup()
    det_1 = sens.add_detector(FarFieldCBSSensor(theta_1, PHI_MAX, T_MAX, d_theta_1, d_phi, DT, estimator))
    det_1.set_theta_limit(0, theta_1)
    det_1.set_phi_slices([0, np.pi / 2])

    det_2 = sens.add_detector(FarFieldCBSSensor(theta_2, PHI_MAX, T_MAX, d_theta_2, d_phi, DT, estimator))
    det_2.set_theta_limit(theta_1 * 0.9, theta_2)          # solape para stitching
    det_2.set_phi_slices([0, np.pi / 2])

    stats = sens.add_detector(StatisticsSensor(z=0, absorb=True))
    stats.set_theta_limit(0, theta_2)
    return sens, det_1, det_2, stats, d_theta_1, d_theta_2, d_phi


# ===========================================================================
# Setup COMPARTIDO por radio (una sola vez: la funcion de fase no se reconstruye
# en cada una de las ~132 corridas; refs en dicts mantienen keep-alive pybind).
# ===========================================================================
SPECIES, DQ, THETA = {}, {}, {}

print("==== Setup (estimator vs analog, integrado) ====")
print(f"  polarizacion DERIVADA: {POLARIZATION}   (m={LASER_M:.4f}, n={LASER_N})")
print("  r[um] |  l*[um] | theta_1[deg] | theta_2[deg] |  w[um]  | w/l*")
for rad in radius_values:
    esp = build_species(rad)
    dq = derived_quantities(esp, VOLUME_FRACTION)
    set_albedo(esp, MU_A_PERCENT, dq['mean_free_path'])

    l_star = dq['transport_mean_free_path']
    theta_1 = Q_FINE / (k * l_star)
    theta_2 = Q_TAIL / (k * l_star)

    SPECIES[rad], DQ[rad], THETA[rad] = esp, dq, (theta_1, theta_2)

    print(f"  {rad:5.3f} | {l_star:7.2f} | {np.rad2deg(theta_1):12.4f} | "
          f"{np.rad2deg(theta_2):12.4f} | {w:7.1f} | {LASER_RADIUS:.1f}")


# ===========================================================================
# Corrida
# ===========================================================================
def run_cbs(exp, radius, n_photons, rep, estimator: bool):
    especie = SPECIES[radius]
    dq = DQ[radius]
    theta_1, theta_2 = THETA[radius]

    sample = Sample(N_MEDIUM)
    sample.add_layer(especie, 0, float('inf'))

    laser = Laser(LASER_M, LASER_N, WAVELENGTH, LASER_RADIUS, LASER_TYPE)

    sens, det_1, det_2, stats, d_theta_1, d_theta_2, d_phi = make_sensors(theta_1, theta_2, estimator)

    # Misma semilla para todos los N de una replica: streams anidados por indice
    # de foton -> curvas de convergencia suaves por replica. Semillas distintas
    # ENTRE replicas -> sigma(N) valido (varianza empirica entre replicas).
    seed = SEED_BASE + 7919 * rep

    config = SimConfig()
    config.n_photons = n_photons
    config.seed = seed
    config.sample = sample
    config.detector = sens
    config.laser = laser
    config.track_reverse_paths = True
    config.pin_threads_to_cores = True    # Linux: affinity real, timing estable (eps)
    config.n_threads = N_THREADS
    config.show_progress = False          # menos ruido con muchas corridas

    t0 = time.time()
    run_simulation_parallel(config)
    runtime_s = time.time() - t0
    mode = "estimator" if estimator else "analog"
    print(f"mode: {mode:9s} | rep: {rep:02d} | n_photons: {n_photons:>11,} "
          f"| runtime_s: {runtime_s:8.2f} | hits: {det_1.hits + det_2.hits}")

    # save_params DESPUES de correr para persistir el tiempo de pared
    # (necesario para eps = 1/(sigma^2 * T)).
    extra = {
        **dq,
        "theta_1": theta_1,
        "theta_2": theta_2,
        "d_theta_1": d_theta_1,
        "d_theta_2": d_theta_2,
        "d_phi": d_phi,
        "n_theta_1": N_THETA_1,
        "n_theta_2": N_THETA_2,
        "n_phi": N_PHI,
        "q_fine": Q_FINE,
        "q_tail": Q_TAIL,
        "t_max": T_MAX,
        "d_time": DT,
        "laser_radius_um": LASER_RADIUS,
        "polarization": POLARIZATION,
        "mode": mode,
        "rep": rep,
        "seed": seed,
        "runtime_s": runtime_s,
        "n_threads": N_THREADS,
    }
    exp.save_params(config, extra=extra)

    exp.save_sensors({"farfield_cbs_1": det_1, "farfield_cbs_2": det_2, "statistics": stats})
    exp.save_processed("farfield_cbs_1", postprocess_farfield_cbs(det_1, n_photons), sensor=det_1)
    exp.save_processed("farfield_cbs_2", postprocess_farfield_cbs(det_2, n_photons), sensor=det_2)

    _keep_alive = (sample, especie); del _keep_alive


# ===========================================================================
# README (f-string: no se desactualiza en lo que importa)
# ===========================================================================
sweep.log_readme(
    f"Validacion ESTIMATOR (next-event) vs ANALOG -- F15/F16, Cap. 4. Capa RGD "
    f"homogenea, polarizacion {POLARIZATION.upper()} (m={LASER_M:.4f}, "
    f"n={LASER_N}, etiqueta derivada del Jones), radios {radius_values} um, "
    f"f={VOLUME_FRACTION}, sin absorcion. INTEGRADO en tiempo (sin eje temporal, "
    f"t_max=dt=0). Haz gaussiano w={LASER_RADIUS:.1f} um (variable por radio; "
    f"OJO: w/l*={LASER_RADIUS:.1f} esta por DEBAJO del umbral de onda plana "
    f"de los otros estudios, w=2500 um con w/l*>=4.5 -> para un test de sesgo "
    f"es valido porque ambos modos comparten el mismo haz, pero el cono absoluto "
    f"no es directamente comparable a §5.1). Doble ventana angular en unidades "
    f"reducidas q=k*l*_teorico*theta anclada POR ESPECIE a l*, IDENTICA en "
    f"estimator y analog (requisito de F16): fina q in [0,{Q_FINE}] "
    f"({N_THETA_1} bins), cola q in [{Q_FINE},{Q_TAIL}] ({N_THETA_2} bins), "
    f"solape [0.9,1.0]*theta_1 para stitching. La UNICA diferencia entre modos "
    f"es el flag de scoring del sensor. F15: sigma_rel del pico E(0) vs N + "
    f"factor de eficiencia eps=1/(sigma^2*T) con tiempo de pared persistido "
    f"(pin_threads_to_cores=True para timing estable). F16: referencia analoga "
    f"fusionando las {N_REPLICAS} replicas del punto mas alto (12 x 1e8 ~ 1.2e9 "
    f"efectivos, con banda) vs perfil del estimador convergido -> coincidencia "
    f"dentro de error = insesgadez. Escaleras log-uniformes de media decada: "
    f"estimator {photons_estimator}, analog {photons_analog}. {N_REPLICAS} "
    f"replicas independientes por (modo, N), semilla SEED_BASE={SEED_BASE} + "
    f"7919*rep (MISMA semilla para todos los N de una replica -> convergencia "
    f"suave; distinta ENTRE replicas -> sigma(N) valido). Orden replica-externa, "
    f"N-interno ascendente: si se aborta a mitad quedan replicas COMPLETAS. "
    f"NOTA analog: la ventana reducida es mas ANGOSTA que un barrido en angulo "
    f"absoluto para l* grande, y el analog es ineficiente en retrodispersion -> "
    f"revisar hits en det_1 (ventana fina) en el N analog mas alto; si el pico "
    f"queda hambriento, subir N analog o ensanchar levemente. Varianza VALIDA = "
    f"empirica entre replicas por bin; las estimaciones internas Poisson/chi2 "
    f"del next-event son invalidas (correlacion angular)."
)


# ===========================================================================
# Loop
#   Replica externa, N interno ascendente. Contador monotono para el prefijo
#   zero-padded del SweepManager (un numero de corrida distinto por run).
# ===========================================================================
run_counter = 0
for index, rad in enumerate(radius_values):
    for rep in range(N_REPLICAS):
        for n in photons_estimator:
            name = f"radius_{rad:.3f}_estimator_N{n}_rep{rep:02d}"
            sweep.run(run_counter, name,
                      lambda exp, rad=rad, n=n, rep=rep: run_cbs(exp, rad, n, rep, estimator=True))
            run_counter += 1
        for n in photons_analog:
            name = f"radius_{rad:.3f}_analog_N{n}_rep{rep:02d}"
            sweep.run(run_counter, name,
                      lambda exp, rad=rad, n=n, rep=rep: run_cbs(exp, rad, n, rep, estimator=False))
            run_counter += 1