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
    MixtureLayer,
)

set_log_level(LogLevel.info)

# ===========================================================================
# QUE ES ESTE ESTUDIO
# ---------------------------------------------------------------------------
#   Validacion ESTIMATOR (next-event) vs ANALOG en medios NO triviales: dos
#   capas y mezcla. NO es un estudio de convergencia ni de eficiencia: un solo
#   N alto FIJO por modo y UNA sola configuracion por medio. Objetivo unico:
#   verificar que estimator y analog dan el MISMO perfil (insesgadez) en
#   geometrias multi-medio, donde el scoring del estimador ejercita rutas que
#   no aparecen en el caso homogeneo -- p.ej. la normalizacion I_norm cacheada
#   por (medium, k), el cruce de interfaz, y la agregacion de la mezcla. Si el
#   estimador tuviera un sesgo especifico de estas geometrias, aparece aqui como
#   una separacion de los dos perfiles fuera de la banda de replicas.
#   Integrado en tiempo (el perfil angular es el observable del chequeo; con n
#   uniforme=1.33 en todas las capas no hay bookkeeping de camino optico que
#   validar por separado).
# ===========================================================================

# ===========================================================================
# Salida
# ===========================================================================
EXP_NAME = "study_estimator_vs_analog_mix"
BASE_DIR = "/home/niaggar/Developer/luminis-mc/temporal_results"

sweep = SweepManager(EXP_NAME, BASE_DIR, timestamped=False)
sweep.snapshot_master_script(__main__.__file__)

# ===========================================================================
# Que correr
# ===========================================================================
RUN_LAYERS = True
RUN_MIXTURE = True

# ===========================================================================
# Parametros fisicos comunes
# ===========================================================================
VOLUME_FRACTION = 0.10
N_PARTICLE = 1.59
N_MEDIUM = 1.33
WAVELENGTH = 0.514               # um
MU_A_PERCENT = 0.0               # sin absorcion (gate de reciprocidad limpio)

# --- CONFIG UNICA capas: interfaz a THICKNESS_MULT * l*_top (regimen donde los
#     fotones cruzan y muestrean AMBAS capas -> ejercita el cruce de interfaz) ---
RADIUS_TOP = 0.035               # um  (l* mayor)
RADIUS_BOT = 0.075               # um  (l* menor -> cono mas ancho = ancla angular)
THICKNESS_MULT = 1.0             # interfaz a 1 l*_top

# --- CONFIG UNICA mezcla: x = maxima mezcla (donde la superposicion falla mas
#     fuerte, §5.3 -> el test mas exigente del scoring de mezcla) ---
RADIUS_1 = 0.075                 # um  (especie de fraccion x)
RADIUS_2 = 0.035                 # um  (especie de fraccion 1-x)
FRACTION_X = 0.5                 # composicion (pesada por scattering) de esp. 1

# ---------------------------------------------------------------------------
# Polarizacion LINEAL (m=1, n=0): valida el MISMO canal de las tandas de
# produccion (mezcla y capas son PLIN). Etiqueta derivada, nunca hardcodeada.
# ---------------------------------------------------------------------------
LASER_M = 1 / np.sqrt(2)
LASER_N = 1j / np.sqrt(2)
LASER_TYPE = LaserSource.Gaussian
POLARIZATION = "circular"

LASER_RADIUS = 2500              # um (FIJO, onda plana, consistente con beam2500)

# Funcion de fase
PHASEF_THETA_MIN = 0.0
PHASEF_THETA_MAX = np.pi
PHASEF_NDIV = 10_000

# ---------------------------------------------------------------------------
# Grilla angular: doble ventana en unidades reducidas q = k*l*_medio*theta,
# anclada al l* de CADA medio. IDENTICA entre estimator y analog dentro de un
# mismo medio (requisito del chequeo). Integrado en tiempo: T_MAX = DT = 0.
# ---------------------------------------------------------------------------
N_THETA_1 = 400                  # ventana fina (cono)
N_THETA_2 = 100                  # ventana cola (fondo/baseline)
Q_FINE, Q_TAIL = 7.5, 40.0
N_PHI = 4
PHI_MAX = 2 * np.pi
k = 2 * np.pi * N_MEDIUM / WAVELENGTH

T_MAX = 0.0
DT = 0.0

# ---------------------------------------------------------------------------
# Muestreo: N alto FIJO por modo (SIN escalera). N distinto a proposito:
#   * estimator converge rapido -> N_ESTIMATOR moderado ya da perfil tenso.
#   * analog es ineficiente en retrodispersion -> N_ANALOG ~100x mayor para ser
#     una referencia igual de tensa. Si quieres un unico numero, igualalos.
#   N_REPLICAS: banda para poder afirmar "coinciden DENTRO de error" (varianza
#   empirica entre replicas; es la unica valida con el estimador next-event).
# ---------------------------------------------------------------------------
N_ESTIMATOR = 300_000
N_ANALOG = 1_000_000_000
N_REPLICAS = 1
N_THREADS = 46
SEED_BASE = 20260714             # distinto de las demas tandas


def make_seed(medium_code, estimator, rep):
    """Streams independientes por (medio, modo, replica): el acuerdo no puede
    ser un artefacto de compartir semilla entre estimator y analog."""
    mode_code = 0 if estimator else 1
    return SEED_BASE + medium_code * 100_000 + mode_code * 10_000 + rep


# ===========================================================================
# Helpers
# ===========================================================================
def number_density(radius, volume_fraction):
    """n = f / ((4/3) pi r^3)   [particulas / um^3]."""
    return volume_fraction / ((4.0 / 3.0) * np.pi * radius ** 3)


def build_species(rad):
    """Una especie RGD con funcion de fase EMC (seccion eficaz NO nula)."""
    phase = RayleighDebyeEMCPhaseFunction(
        WAVELENGTH, rad, N_PARTICLE, N_MEDIUM,
        PHASEF_NDIV, PHASEF_THETA_MIN, PHASEF_THETA_MAX,
    )
    return RGDMedium(phase, rad, N_PARTICLE, N_MEDIUM, WAVELENGTH)


def set_albedo(medium, mu_a_percent, mfp=None):
    """Albedo bien definido (evita 0/0 en la agregacion). La escala absoluta de
    mu_s no afecta el transporte. mfp: solo para capas homogeneas; en mezcla la
    mu_s la fijan las densidades, asi que se omite."""
    medium.set_absorption_coefficient(mu_a_percent)
    medium.set_scattering_coefficient(1.0 - mu_a_percent)
    if mfp is not None:
        medium.set_mean_free_path(mfp)


def make_sensors(theta_1, theta_2, estimator):
    """Doble detector far-field (fino + cola) + estadistica, INTEGRADO en tiempo.
    El flag `estimator` (ultimo arg) va en AMBOS detectores -> modo homogeneo."""
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


def base_config(sample, laser, sens, seed, n_photons):
    config = SimConfig()
    config.n_photons = n_photons
    config.sample = sample
    config.detector = sens
    config.laser = laser
    config.track_reverse_paths = True          # imprescindible para CBS
    config.pin_threads_to_cores = False
    config.n_threads = N_THREADS
    config.show_progress = False
    config.seed = seed
    return config


def persist(exp, det_1, det_2, stats, n_photons):
    exp.save_sensors({"farfield_cbs_1": det_1, "farfield_cbs_2": det_2, "statistics": stats})
    exp.save_processed("farfield_cbs_1", postprocess_farfield_cbs(det_1, n_photons), sensor=det_1)
    exp.save_processed("farfield_cbs_2", postprocess_farfield_cbs(det_2, n_photons), sensor=det_2)


# ===========================================================================
# Setup CAPAS (una sola vez)
# ===========================================================================
SPECIES_TOP = build_species(RADIUS_TOP)
SPECIES_BOT = build_species(RADIUS_BOT)
DQ_TOP = derived_quantities(SPECIES_TOP, VOLUME_FRACTION)
DQ_BOT = derived_quantities(SPECIES_BOT, VOLUME_FRACTION)
set_albedo(SPECIES_TOP, MU_A_PERCENT, DQ_TOP['mean_free_path'])
set_albedo(SPECIES_BOT, MU_A_PERCENT, DQ_BOT['mean_free_path'])

L_STAR_TOP = DQ_TOP['transport_mean_free_path']
L_STAR_BOT = DQ_BOT['transport_mean_free_path']
LSTAR_ANGLE_LAYERS = min(L_STAR_TOP, L_STAR_BOT)           # cono mas ancho
THETA_1_L = Q_FINE / (k * LSTAR_ANGLE_LAYERS)
THETA_2_L = Q_TAIL / (k * LSTAR_ANGLE_LAYERS)
Z_INTERFACE = THICKNESS_MULT * L_STAR_TOP

# ===========================================================================
# Setup MEZCLA (una sola vez)
# ===========================================================================
SPECIES_1 = build_species(RADIUS_1)
SPECIES_2 = build_species(RADIUS_2)
set_albedo(SPECIES_1, MU_A_PERCENT)                        # mezcla: sin mfp (densidades)
set_albedo(SPECIES_2, MU_A_PERCENT)
DQ_1 = derived_quantities(SPECIES_1, VOLUME_FRACTION)
DQ_2 = derived_quantities(SPECIES_2, VOLUME_FRACTION)

SIGMA_1 = DQ_1['scattering_efficiency'] * np.pi * RADIUS_1 ** 2
SIGMA_2 = DQ_2['scattering_efficiency'] * np.pi * RADIUS_2 ** 2
MU_S_TOTAL = number_density(RADIUS_1, VOLUME_FRACTION) * SIGMA_1   # ancla en x=1
ND_1 = FRACTION_X * MU_S_TOTAL / SIGMA_1
ND_2 = (1.0 - FRACTION_X) * MU_S_TOTAL / SIGMA_2
DENSITIES = [ND_1, ND_2]

DQ_MIX = derived_quantities_mixture([SPECIES_1, SPECIES_2], DENSITIES)
LSTAR_MIX = DQ_MIX['transport_mean_free_path']
THETA_1_M = Q_FINE / (k * LSTAR_MIX)
THETA_2_M = Q_TAIL / (k * LSTAR_MIX)

print("==== Setup (estimator vs analog en medios; chequeo de insesgadez) ====")
print(f"  N_ESTIMATOR={N_ESTIMATOR:,}   N_ANALOG={N_ANALOG:,}   N_REPLICAS={N_REPLICAS}")
print(f"  polarizacion: {POLARIZATION}   haz: {LASER_RADIUS} um")
if RUN_LAYERS:
    print(f"  [CAPAS]  top r={RADIUS_TOP} (l*={L_STAR_TOP:.2f}) | bot r={RADIUS_BOT} (l*={L_STAR_BOT:.2f})")
    print(f"           interfaz z={Z_INTERFACE:.2f} um = {THICKNESS_MULT:.1f} l*_top | "
          f"ancla angular l*={LSTAR_ANGLE_LAYERS:.2f} | "
          f"theta_1={np.rad2deg(THETA_1_L):.3f} deg theta_2={np.rad2deg(THETA_2_L):.3f} deg")
if RUN_MIXTURE:
    print(f"  [MEZCLA] r1={RADIUS_1} r2={RADIUS_2} x={FRACTION_X} | l*_mix={LSTAR_MIX:.2f} | "
          f"mu_s_share=[{ND_1*SIGMA_1/MU_S_TOTAL:.3f}, {ND_2*SIGMA_2/MU_S_TOTAL:.3f}] | "
          f"theta_1={np.rad2deg(THETA_1_M):.3f} deg theta_2={np.rad2deg(THETA_2_M):.3f} deg")


# ===========================================================================
# Corrida CAPAS
# ===========================================================================
def run_layers(exp: Experiment, estimator: bool, rep: int):
    mode = "estimator" if estimator else "analog"
    n_photons = N_ESTIMATOR if estimator else N_ANALOG
    seed = make_seed(0, estimator, rep)

    sample = Sample(N_MEDIUM)
    sample.add_layer(SPECIES_TOP, 0.0, Z_INTERFACE)
    sample.add_layer(SPECIES_BOT, Z_INTERFACE, float("inf"))

    laser = Laser(LASER_M, LASER_N, WAVELENGTH, LASER_RADIUS, LASER_TYPE)
    sens, det_1, det_2, stats, d_theta_1, d_theta_2, d_phi = make_sensors(THETA_1_L, THETA_2_L, estimator)
    config = base_config(sample, laser, sens, seed, n_photons)

    extra = {
        "dq_top": DQ_TOP,
        "dq_bot": DQ_BOT,
        "medium_kind": "two_layers",
        "mode": mode,
        "radius_top": RADIUS_TOP,
        "radius_bot": RADIUS_BOT,
        "z_interface": Z_INTERFACE,
        "thickness_mult": THICKNESS_MULT,
        "l_star_top": L_STAR_TOP,
        "l_star_bot": L_STAR_BOT,
        "lstar_angle_anchor": LSTAR_ANGLE_LAYERS,
        "laser_radius_um": LASER_RADIUS,
        "theta_1": THETA_1_L, "theta_2": THETA_2_L,
        "d_theta_1": d_theta_1, "d_theta_2": d_theta_2, "d_phi": d_phi,
        "n_theta_1": N_THETA_1, "n_theta_2": N_THETA_2, "n_phi": N_PHI,
        "q_fine": Q_FINE, "q_tail": Q_TAIL,
        "t_max": T_MAX, "d_time": DT,
        "polarization": POLARIZATION,
        "n_photons": n_photons, "seed": seed, "replica": rep,
    }

    t0 = time.time()
    run_simulation_parallel(config)
    runtime_s = time.time() - t0
    extra["runtime_s"] = runtime_s
    print(f"[CAPAS]  mode: {mode:9s} rep: {rep:02d} | N: {n_photons:>12,} | "
          f"runtime_s: {runtime_s:8.2f} | hits_fine: {det_1.hits} hits_tail: {det_2.hits}")

    exp.save_params(config, extra=extra)
    persist(exp, det_1, det_2, stats, n_photons)

    _keep_alive = (sample,); del _keep_alive


# ===========================================================================
# Corrida MEZCLA
# ===========================================================================
def run_mixture(exp: Experiment, estimator: bool, rep: int):
    mode = "estimator" if estimator else "analog"
    n_photons = N_ESTIMATOR if estimator else N_ANALOG
    seed = make_seed(1, estimator, rep)

    species = [SPECIES_1, SPECIES_2]
    sample = Sample(N_MEDIUM)
    sample.add_mixture_layer(species, DENSITIES, 0.0, float("inf"))

    layer = sample.layers[0]
    mfp_layer = float(layer.mfp_total) if isinstance(layer, MixtureLayer) else 0.0

    laser = Laser(LASER_M, LASER_N, WAVELENGTH, LASER_RADIUS, LASER_TYPE)
    sens, det_1, det_2, stats, d_theta_1, d_theta_2, d_phi = make_sensors(THETA_1_M, THETA_2_M, estimator)
    config = base_config(sample, laser, sens, seed, n_photons)

    extra = {
        **DQ_MIX,
        "medium_kind": "mixture",
        "mode": mode,
        "fraction_x": FRACTION_X,
        "radius": [RADIUS_1, RADIUS_2],
        "number_densities": DENSITIES,
        "mu_s_total": MU_S_TOTAL,
        "mu_s_share": [ND_1 * SIGMA_1 / MU_S_TOTAL, ND_2 * SIGMA_2 / MU_S_TOTAL],
        "mfp_total_layer": mfp_layer,
        "lstar_mix": LSTAR_MIX,
        "laser_radius_um": LASER_RADIUS,
        "theta_1": THETA_1_M, "theta_2": THETA_2_M,
        "d_theta_1": d_theta_1, "d_theta_2": d_theta_2, "d_phi": d_phi,
        "n_theta_1": N_THETA_1, "n_theta_2": N_THETA_2, "n_phi": N_PHI,
        "q_fine": Q_FINE, "q_tail": Q_TAIL,
        "t_max": T_MAX, "d_time": DT,
        "polarization": POLARIZATION,
        "n_photons": n_photons, "seed": seed, "replica": rep,
    }

    t0 = time.time()
    run_simulation_parallel(config)
    runtime_s = time.time() - t0
    extra["runtime_s"] = runtime_s
    print(f"[MEZCLA] mode: {mode:9s} rep: {rep:02d} | N: {n_photons:>12,} | "
          f"runtime_s: {runtime_s:8.2f} | hits_fine: {det_1.hits} hits_tail: {det_2.hits}")

    exp.save_params(config, extra=extra)
    persist(exp, det_1, det_2, stats, n_photons)

    _keep_alive = (sample,); del _keep_alive


# ===========================================================================
# README (f-string: no se desactualiza en lo que importa)
# ===========================================================================
sweep.log_readme(
    f"Validacion ESTIMATOR (next-event) vs ANALOG en medios NO triviales -- "
    f"chequeo de INSESGADEZ (no de convergencia). Un solo N alto FIJO por modo, "
    f"UNA configuracion por medio. Objetivo: verificar que estimator y analog "
    f"dan el MISMO perfil angular en geometrias multi-medio, donde el scoring "
    f"del estimador ejercita rutas ausentes del caso homogeneo (normalizacion "
    f"I_norm por (medium,k), cruce de interfaz, agregacion de mezcla). "
    f"Polarizacion {POLARIZATION.upper()} (m={LASER_M}, n={LASER_N}), haz FIJO "
    f"w={LASER_RADIUS} um (onda plana), sin absorcion, INTEGRADO en tiempo "
    f"(t_max=dt=0; n=1.33 uniforme -> sin bookkeeping de camino optico que "
    f"validar aparte). CAPAS: top r={RADIUS_TOP} um (l*={L_STAR_TOP:.2f}), bot "
    f"r={RADIUS_BOT} um (l*={L_STAR_BOT:.2f}), interfaz a {THICKNESS_MULT:.1f} "
    f"l*_top = {Z_INTERFACE:.2f} um (los fotones cruzan y muestrean ambas capas). "
    f"MEZCLA: r1={RADIUS_1}/r2={RADIUS_2} um a x={FRACTION_X} (mezcla maxima = "
    f"test mas exigente del scoring, §5.3), mu_s_total fijo {MU_S_TOTAL:.4e} 1/um. "
    f"Doble ventana angular en unidades reducidas q=k*l*_medio*theta anclada al "
    f"l* de CADA medio (capas: min l*={LSTAR_ANGLE_LAYERS:.2f}; mezcla: "
    f"l*_mix={LSTAR_MIX:.2f}), IDENTICA entre estimator y analog dentro de un "
    f"medio: fina q in [0,{Q_FINE}] ({N_THETA_1} bins), cola q in "
    f"[{Q_FINE},{Q_TAIL}] ({N_THETA_2} bins), solape [0.9,1.0]*theta_1. La UNICA "
    f"diferencia entre modos es el flag de scoring del sensor. N_ESTIMATOR="
    f"{N_ESTIMATOR:,} (converge) vs N_ANALOG={N_ANALOG:,} (referencia tensa; "
    f"distintos a proposito, el analog es ineficiente en retrodispersion) -- "
    f"ambos FIJOS, sin escalera. {N_REPLICAS} replicas/modo para la BANDA "
    f"(varianza empirica entre replicas, unica valida con next-event); "
    f"insesgadez = perfiles coinciden dentro de esa banda. Semillas "
    f"SEED_BASE={SEED_BASE} + 100000*medio + 10000*modo + rep (streams "
    f"independientes: el acuerdo no es artefacto de semilla compartida). "
    f"RUN_LAYERS={RUN_LAYERS}, RUN_MIXTURE={RUN_MIXTURE}."
)


# ===========================================================================
# Loop  (replica-externa, modo-interno: cada rep deja el PAR estimator+analog
# completo; si se aborta a mitad quedan pares utilizables. Contador monotono.)
# ===========================================================================
run_counter = 0

if RUN_LAYERS:
    for rep in range(N_REPLICAS):
        for est in (True, False):
            tag = "estimator" if est else "analog"
            name = f"layers_z{Z_INTERFACE:.2f}_{tag}_rep{rep:02d}"
            print(f"\n=== Corrida: {name} ===")
            sweep.run(run_counter, name,
                      lambda exp, est=est, rep=rep: run_layers(exp, est, rep))
            run_counter += 1

if RUN_MIXTURE:
    for rep in range(N_REPLICAS):
        for est in (True, False):
            tag = "estimator" if est else "analog"
            name = f"mixture_x{FRACTION_X:.2f}_{tag}_rep{rep:02d}"
            print(f"\n=== Corrida: {name} ===")
            sweep.run(run_counter, name,
                      lambda exp, est=est, rep=rep: run_mixture(exp, est, rep))
            run_counter += 1