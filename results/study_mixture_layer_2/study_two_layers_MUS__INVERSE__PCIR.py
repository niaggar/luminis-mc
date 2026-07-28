import __main__
import time
import hashlib
import numpy as np

from luminis_mc import (
    Experiment,
    SweepManager,
    Laser, RGDMedium, MieMedium, Sample,
    FarFieldCBSSensor, StatisticsSensor, SensorsGroup,
    SimConfig, RayleighDebyeEMCPhaseFunction, MiePhaseFunction,
    run_simulation_parallel, postprocess_farfield_cbs,
    derived_quantities,
    set_log_level, LogLevel, LaserSource,
)

from utils.time import build_time_grid

set_log_level(LogLevel.info)

# ===========================================================================
# CBS ESTRATIFICADO -- matriz de 8 campañas
#   2 familias (mu_s igualado / phi fijo) x 2 ordenes (normal / inverso)
#   x 2 polarizaciones (circular / lineal)
#
# Los 8 archivos salen del MISMO template y difieren en exactamente cuatro
# bloques: EXP_NAME, SEED_BASE_STRAT, ORDEN y FAMILIA (+ el bloque de
# polarizacion). Cualquier cambio de fisica o de grilla debe hacerse en el
# generador, nunca en un archivo suelto.
#
# --- Que es comparable con que ---
#   DENTRO de una familia (4 campañas): medios, anclas, ventana angular,
#     d_theta, grilla temporal y profundidades ABSOLUTAS son identicos bin a
#     bin. eta(q), Delta_theta(t) y R(t) se restan directamente, sin
#     re-binning. Lo certifica GRID_FINGERPRINT: las 4 comparten valor.
#   ENTRE familias: los medios son distintos por construccion (phi_large
#     cambia), asi que l*, las anclas y por tanto theta_i, dt y t_max NO
#     coinciden -> fingerprints distintos. La comparacion entre familias es
#     legitima SOLO en unidades reducidas (q = k l* theta, t/tau*) o sobre
#     escalares adimensionales (eta(0), rho, razones de ancho). Las
#     profundidades z[um] SI coinciden: el barrido escala con una longitud de
#     REFERENCIA comun (l_s de la particula pequeña a phi=0.10), asi que la
#     geometria del apilamiento es la misma en las 8 campañas.
# ===========================================================================

# ===========================================================================
# Salida
# ===========================================================================
EXP_NAME = "study_two_layers_MUS__INVERSE__PCIR__beam2500"
BASE_DIR = "/home/niaggar/Developer/luminis-mc/temporal_results"

sweep = SweepManager(EXP_NAME, BASE_DIR, timestamped=False)
sweep.snapshot_master_script(__main__.__file__)

# ===========================================================================
# Especies (definidas por ROL, no por posicion)
# ===========================================================================
RADIUS_SMALL = 0.035             # um -> RGD/EMC
RADIUS_LARGE = 0.100             # um -> Mie (fuera del dominio RGD)

MODEL_SMALL = "rgd"
MODEL_LARGE = "mie"

VOLUME_FRACTION_SMALL = 0.10     # FIJA en las dos familias (referencia)

# ---------------------------------------------------------------------------
# FAMILIA  <<< define phi_large y la resolucion angular >>>
#   Familia MU_S IGUALADO: phi_small=0.10 es la REFERENCIA y phi_large se
#   DERIVA de phi_large = 4 a_large mu_s_ref/(3 Q_large). => l_s comun a las
#   dos capas; el UNICO contraste es la anisotropia g -> l*_large/l*_small ~ 1.8.
#   Contraste moderado => 300 bins bastan para resolver el cono estrecho.
FAMILY = "matched_mus"
N_THETA_1 = 300
N_PHOTONS = 200_000
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ORDEN DE APILAMIENTO
#   INVERSO -> entra la luz por la particula GRANDE; la PEQUEÑA queda
#              enterrada y semi-infinita.
ORDER = "inverse"
RADIUS_TOP, MODEL_TOP = RADIUS_LARGE, MODEL_LARGE
RADIUS_BOT, MODEL_BOT = RADIUS_SMALL, MODEL_SMALL
# ---------------------------------------------------------------------------

# ===========================================================================
# Parametros fisicos
# ===========================================================================
N_PARTICLE = 1.59
N_MEDIUM = 1.33
WAVELENGTH = 0.514               # um
MU_A_PERCENT = 0.0               # sin absorcion (gate de reciprocidad limpio)

# ---------------------------------------------------------------------------
# Espesor de la capa de ENTRADA, en unidades de una longitud de REFERENCIA
# comun a las 8 campañas: L_SCALE_REF = l_s de la particula PEQUEÑA a phi=0.10.
#   Se escala con una referencia FIJA y no con l*_in a proposito: l*_in cambia
#   con el orden Y con la familia, asi que m*l*_in daria profundidades fisicas
#   distintas en cada campaña y nada seria comparable en um. Con L_SCALE_REF
#   los z[um] son los MISMOS en las 8. El espesor en unidades locales
#   (d/l*_top, d/l_s_top) queda como columna derivada en params.
# ---------------------------------------------------------------------------
thickness_multipliers_ref = [0.00001, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

# ---------------------------------------------------------------------------
# POLARIZACION
#   CIRCULAR (m=1/sqrt2, n=i/sqrt2). Canal que preserva la helicidad:
#   eta(0)=2 protegido por reciprocidad, sin dilucion de vertice.
LASER_M = 1 / np.sqrt(2)
LASER_N = 1j / np.sqrt(2)
POLARIZATION = "circular"
# ---------------------------------------------------------------------------
LASER_TYPE = LaserSource.Gaussian
LASER_RADIUS = 2500              # um  (haz FIJO; w/l* >> 1 -> onda plana)

# Funcion de fase
PHASEF_THETA_MIN = 0.0
PHASEF_THETA_MAX = np.pi
PHASEF_NDIV = 10_000

# ---------------------------------------------------------------------------
# Grilla angular COMPARTIDA, q = k*l*_ancla*theta.
#   Ancla = MIN sobre las ESPECIES -> contiene el cono mas ANCHO del barrido y
#   el ensanchamiento de tiempos tempranos. Invariante al orden.
#   N_THETA_1 lo fija la FAMILIA: cuanto mayor es el contraste l*_max/l*_min,
#   mas bins hacen falta para no sub-resolver el cono ESTRECHO (que en q vale
#   ~ l*_min/l*_max).
# ---------------------------------------------------------------------------
N_THETA_2 = 100                  # ventana cola (fondo/baseline)
Q_FINE, Q_TAIL = 7.5, 40.0
N_PHI = 36
PHI_MAX = 2 * np.pi
k = 2 * np.pi * N_MEDIUM / WAVELENGTH

# ---------------------------------------------------------------------------
# Grilla temporal COMPARTIDA.
#   Ancla = MAX sobre las ESPECIES (dinamica mas LENTA), OPUESTA a la angular.
#   Invariante al orden -> t_max y dt identicos dentro de la familia.
# ---------------------------------------------------------------------------
TIME_NBINS = 100
TIME_TMAX_TAUSTAR = 40

# ---------------------------------------------------------------------------
N_THREADS = 15
N_REPLICAS = 5
SEED_BASE_STRAT = 20260713

C0 = 0.299792458                 # um/fs


# ===========================================================================
# Helpers
# ===========================================================================
_PHASE_KEEPALIVE = []            # los medios guardan un puntero crudo a phase


def build_species(rad, model):
    """Una especie con funcion de fase con seccion eficaz NO nula (EMC o Mie)."""
    if model == "rgd":
        phase = RayleighDebyeEMCPhaseFunction(
            WAVELENGTH, rad, N_PARTICLE, N_MEDIUM,
            PHASEF_NDIV, PHASEF_THETA_MIN, PHASEF_THETA_MAX,
        )
        medium = RGDMedium(phase, rad, N_PARTICLE, N_MEDIUM, WAVELENGTH)
    elif model == "mie":
        phase = MiePhaseFunction(
            WAVELENGTH, rad, N_PARTICLE, N_MEDIUM,
            PHASEF_NDIV, PHASEF_THETA_MIN, PHASEF_THETA_MAX,
        )
        medium = MieMedium(phase, rad, N_PARTICLE, N_MEDIUM, WAVELENGTH)
    else:
        raise ValueError(f"model desconocido: {model!r}")

    _PHASE_KEEPALIVE.append(phase)
    return medium


def match_volume_fraction(medium_ref, phi_ref, medium_tgt, rad_tgt):
    """phi de la especie objetivo que iguala mu_s de la de referencia.
    Q_sca no depende de phi: la llamada de sondeo puede usar cualquier phi."""
    dq_ref = derived_quantities(medium_ref, phi_ref)
    mus_ref = 1.0 / dq_ref["mean_free_path"]

    dq_probe = derived_quantities(medium_tgt, phi_ref)
    q_tgt = dq_probe["scattering_efficiency"]

    return (4.0 * rad_tgt * mus_ref) / (3.0 * q_tgt)


def set_albedo(medium, mu_a_percent, mfp):
    """Albedo bien definido + camino libre medio. El muestreo de paso libre usa
    SIEMPRE mean_free_path; mu_s/mu_a solo fijan el reparto de absorcion."""
    medium.set_absorption_coefficient(mu_a_percent)
    medium.set_scattering_coefficient(1.0 - mu_a_percent)
    medium.set_mean_free_path(mfp)


def make_sensors():
    """Doble detector far-field (fino + cola) + estadistica. Toda la geometria
    de bins viene de constantes de modulo -> identica en las 4 campañas de la
    familia."""
    sens = SensorsGroup()
    det_1 = sens.add_detector(
        FarFieldCBSSensor(THETA_1, PHI_MAX, T_MAX, D_THETA_1, D_PHI, D_TIME, True))
    det_1.set_theta_limit(0, THETA_1)
    det_1.set_phi_slices([0])

    det_2 = sens.add_detector(
        FarFieldCBSSensor(THETA_2, PHI_MAX, T_MAX, D_THETA_2, D_PHI, D_TIME, True))
    det_2.set_theta_limit(THETA_1 * 0.9, THETA_2)          # solape para stitching
    det_2.set_phi_slices([0])

    stats = sens.add_detector(StatisticsSensor(z=0, absorb=True))
    stats.set_theta_limit(0, THETA_2)
    return sens, det_1, det_2, stats


def base_config(sample, laser, sens, seed):
    config = SimConfig()
    config.n_photons = N_PHOTONS
    config.sample = sample
    config.detector = sens
    config.laser = laser
    config.track_reverse_paths = True          # imprescindible para CBS
    config.pin_threads_to_cores = False
    config.n_threads = N_THREADS
    config.show_progress = True
    config.seed = seed
    return config


# ===========================================================================
# Setup COMPARTIDO (una sola vez)
# ===========================================================================
SPECIES_SMALL = build_species(RADIUS_SMALL, MODEL_SMALL)
SPECIES_LARGE = build_species(RADIUS_LARGE, MODEL_LARGE)

# --- phi de la especie GRANDE segun la FAMILIA ---
if FAMILY == "matched_mus":
    VOLUME_FRACTION_LARGE = match_volume_fraction(
        SPECIES_SMALL, VOLUME_FRACTION_SMALL, SPECIES_LARGE, RADIUS_LARGE)
elif FAMILY == "fixed_phi":
    VOLUME_FRACTION_LARGE = VOLUME_FRACTION_SMALL
else:
    raise ValueError(f"FAMILY desconocida: {FAMILY!r}")

DQ_SMALL = derived_quantities(SPECIES_SMALL, VOLUME_FRACTION_SMALL)
DQ_LARGE = derived_quantities(SPECIES_LARGE, VOLUME_FRACTION_LARGE)

L_S_SMALL = DQ_SMALL["mean_free_path"]
L_S_LARGE = DQ_LARGE["mean_free_path"]
MU_S_SMALL = 1.0 / L_S_SMALL
MU_S_LARGE = 1.0 / L_S_LARGE

# verificacion dura del invariante de cada familia
_rel_mus = abs(L_S_LARGE - L_S_SMALL) / L_S_SMALL
if FAMILY == "matched_mus":
    assert _rel_mus < 1e-6, f"mu_s NO igualado: desviacion relativa {_rel_mus:.2e}"
else:
    assert abs(VOLUME_FRACTION_LARGE - VOLUME_FRACTION_SMALL) < 1e-12, "phi NO igualada"

set_albedo(SPECIES_SMALL, MU_A_PERCENT, L_S_SMALL)
set_albedo(SPECIES_LARGE, MU_A_PERCENT, L_S_LARGE)

L_STAR_SMALL = DQ_SMALL["transport_mean_free_path"]
L_STAR_LARGE = DQ_LARGE["transport_mean_free_path"]

# longitud de REFERENCIA para el barrido: comun a las 8 campañas
L_SCALE_REF = L_S_SMALL

# --- asignacion a posiciones (depende del ORDEN) ---
if ORDER == "normal":
    SPECIES_TOP, DQ_TOP, VOLUME_FRACTION_TOP = SPECIES_SMALL, DQ_SMALL, VOLUME_FRACTION_SMALL
    SPECIES_BOT, DQ_BOT, VOLUME_FRACTION_BOT = SPECIES_LARGE, DQ_LARGE, VOLUME_FRACTION_LARGE
elif ORDER == "inverse":
    SPECIES_TOP, DQ_TOP, VOLUME_FRACTION_TOP = SPECIES_LARGE, DQ_LARGE, VOLUME_FRACTION_LARGE
    SPECIES_BOT, DQ_BOT, VOLUME_FRACTION_BOT = SPECIES_SMALL, DQ_SMALL, VOLUME_FRACTION_SMALL
else:
    raise ValueError(f"ORDER desconocido: {ORDER!r}")

L_STAR_TOP = DQ_TOP["transport_mean_free_path"]   # medio de ENTRADA
L_STAR_BOT = DQ_BOT["transport_mean_free_path"]
L_S_TOP = DQ_TOP["mean_free_path"]
L_S_BOT = DQ_BOT["mean_free_path"]

# --- anclas sobre el CONJUNTO DE ESPECIES: invariantes al orden ---
L_STAR_ANGLE_ANCHOR = min(L_STAR_SMALL, L_STAR_LARGE)   # cono mas ANCHO
L_STAR_TIME_ANCHOR = max(L_STAR_SMALL, L_STAR_LARGE)    # dinamica mas LENTA
ANGLE_ANCHOR_SPECIES = "small" if L_STAR_SMALL <= L_STAR_LARGE else "large"
TIME_ANCHOR_SPECIES = "large" if L_STAR_LARGE >= L_STAR_SMALL else "small"
ANGLE_ANCHOR_LAYER = "top" if L_STAR_TOP == L_STAR_ANGLE_ANCHOR else "bot"
TIME_ANCHOR_LAYER = "top" if L_STAR_TOP == L_STAR_TIME_ANCHOR else "bot"

# contraste de transporte y resolucion del cono ESTRECHO en la ventana fina
LSTAR_CONTRAST = L_STAR_TIME_ANCHOR / L_STAR_ANGLE_ANCHOR
Q_NARROW = 1.0 / LSTAR_CONTRAST          # ancho caracteristico del cono estrecho en q

# --- grilla angular ---
THETA_1 = Q_FINE / (k * L_STAR_ANGLE_ANCHOR)
THETA_2 = Q_TAIL / (k * L_STAR_ANGLE_ANCHOR)
D_THETA_1 = THETA_1 / N_THETA_1
D_THETA_2 = THETA_2 / N_THETA_2
D_PHI = PHI_MAX / N_PHI
BINS_ACROSS_NARROW = Q_NARROW / (Q_FINE / N_THETA_1)

# --- grilla temporal ---
GRID = build_time_grid(
    L_STAR_TIME_ANCHOR, N_MEDIUM,
    n_bins=TIME_NBINS, t_max_taustar=TIME_TMAX_TAUSTAR
)
T_MAX = GRID["t_max_sim"]
D_TIME = GRID["dt_sim"]

# --- profundidades ABSOLUTAS (identicas en las 8 campañas) ---
Z_INTERFACES = [m * L_SCALE_REF for m in thickness_multipliers_ref]


# --- alcance en profundidad, evaluado para AMBOS ordenes ---
def depth_reach(l_star_entry):
    """(M, mult en l*_in, z_probe en um). Criterio difusivo RMS: sondear una
    interfaz a profundidad d exige M >= (3/2)(d/l*_in)^2."""
    tau_entry_fs = l_star_entry * N_MEDIUM / C0
    m_entry = GRID["t_max_fs"] / tau_entry_fs
    mult = np.sqrt(2.0 * m_entry / 3.0)
    return m_entry, mult, mult * l_star_entry


M_TOP, MULT_REACH, Z_PROBE = depth_reach(L_STAR_TOP)
Z_PROBE_COMMON = min(depth_reach(L_STAR_SMALL)[2], depth_reach(L_STAR_LARGE)[2])

# --- fingerprint: medios + grillas + profundidades. Igual en las 4 campañas
#     de la familia; distinto entre familias (por diseño). ---
_fp_payload = "|".join(f"{v:.12e}" for v in (
    [RADIUS_SMALL, RADIUS_LARGE, VOLUME_FRACTION_SMALL, VOLUME_FRACTION_LARGE,
     MU_S_SMALL, MU_S_LARGE, L_S_SMALL, L_S_LARGE, L_STAR_SMALL, L_STAR_LARGE,
     THETA_1, THETA_2, D_THETA_1, D_THETA_2, D_PHI,
     T_MAX, D_TIME, float(TIME_NBINS), float(N_THETA_1), float(N_THETA_2),
     float(N_PHI), LASER_RADIUS * 1.0]
    + list(Z_INTERFACES)
))
GRID_FINGERPRINT = hashlib.sha256(_fp_payload.encode()).hexdigest()[:16]


def _tag(l_star):
    return "ancho/rapido" if l_star == L_STAR_ANGLE_ANCHOR else "estrecho/lento"


print(f"==== Setup | familia {FAMILY.upper()} | orden {ORDER.upper()} | pol {POLARIZATION.upper()} ====")
print(f"TOP (entra luz): r={RADIUS_TOP} um [{MODEL_TOP.upper()}]  phi={VOLUME_FRACTION_TOP:.6f}  "
      f"g={DQ_TOP['anisotropy_g']:.4f}  l_s={L_S_TOP:.2f} um  l*={L_STAR_TOP:.2f} um  ({_tag(L_STAR_TOP)})")
print(f"BOT (semi-inf) : r={RADIUS_BOT} um [{MODEL_BOT.upper()}]  phi={VOLUME_FRACTION_BOT:.6f}  "
      f"g={DQ_BOT['anisotropy_g']:.4f}  l_s={L_S_BOT:.2f} um  l*={L_STAR_BOT:.2f} um  ({_tag(L_STAR_BOT)})")
print(f"mu_s: small={MU_S_SMALL:.6f} um^-1 (l_s={L_S_SMALL:.2f} um)   "
      f"large={MU_S_LARGE:.6f} um^-1 (l_s={L_S_LARGE:.2f} um)   razon={MU_S_LARGE/MU_S_SMALL:.3f}")
print(f"l*: small={L_STAR_SMALL:.2f} um   large={L_STAR_LARGE:.2f} um   "
      f"contraste max/min = {LSTAR_CONTRAST:.3f}")
print(f"theta_cbs: small={np.degrees(DQ_SMALL['theta_coherent']):.4f} deg  "
      f"large={np.degrees(DQ_LARGE['theta_coherent']):.4f} deg")
print("---- grillas (identicas en las 4 campañas de esta familia) ----")
print(f"ancla angulo = min l* = l*_{ANGLE_ANCHOR_SPECIES} ({L_STAR_ANGLE_ANCHOR:.2f} um) "
      f"[en este orden, capa {ANGLE_ANCHOR_LAYER.upper()}]")
print(f"ancla tiempo = max l* = l*_{TIME_ANCHOR_SPECIES} ({L_STAR_TIME_ANCHOR:.2f} um) "
      f"[en este orden, capa {TIME_ANCHOR_LAYER.upper()}]")
print(f"theta_1={np.rad2deg(THETA_1):.5f} deg ({N_THETA_1} bins, d_theta_1="
      f"{np.rad2deg(D_THETA_1):.6f} deg)   theta_2={np.rad2deg(THETA_2):.5f} deg ({N_THETA_2} bins)")
print(f"resolucion del cono ESTRECHO: q_narrow={Q_NARROW:.4f} -> "
      f"{BINS_ACROSS_NARROW:.1f} bins  {'(OK)' if BINS_ACROSS_NARROW >= 15 else '(SUB-RESUELTO: subir N_THETA_1)'}")
print(f"GRID: dt={GRID['dt_fs']:.2f} fs  t_max={GRID['t_max_fs']:.0f} fs  tau*={GRID['tau_star_fs']:.2f} fs")
print(f"haz: {LASER_RADIUS} um   w/l*_small={LASER_RADIUS/L_STAR_SMALL:.1f}  "
      f"w/l*_large={LASER_RADIUS/L_STAR_LARGE:.1f}   fotones: {N_PHOTONS:,}")
print(f"alcance: este orden {Z_PROBE:.0f} um ({MULT_REACH:.2f} l*_top, M_top={M_TOP:.1f})   "
      f"PAREADO normal/inverso {Z_PROBE_COMMON:.0f} um = {Z_PROBE_COMMON/L_SCALE_REF:.2f} L_ref")
print(f"L_SCALE_REF (l_s de la pequeña a phi=0.10) = {L_SCALE_REF:.3f} um")
print(f"FINGERPRINT: {GRID_FINGERPRINT}  (igual en las 4 campañas de esta familia)")

print("  z[um] | d/L_ref | d/l*_top | d/l_s_top | M_top nec. | este orden | pareado")
for z in Z_INTERFACES:
    mult_local = z / L_STAR_TOP
    ok_self = "SI" if z <= Z_PROBE else "no"
    ok_pair = "SI" if z <= Z_PROBE_COMMON else "no"
    print(f"  {z:7.1f} | {z/L_SCALE_REF:7.2f} | {mult_local:8.2f} | {z/L_S_TOP:9.2f} | "
          f"{1.5*mult_local**2:10.1f} | {ok_self:^10} | {ok_pair:^7}")


# ===========================================================================
# Metadata comun
# ===========================================================================
def common_extra(config, rep):
    return {
        "family": FAMILY,
        "order": ORDER,
        "grid_fingerprint": GRID_FINGERPRINT,
        "dq_top": DQ_TOP,
        "dq_bot": DQ_BOT,
        "dq_small": DQ_SMALL,
        "dq_large": DQ_LARGE,
        "radius_top": RADIUS_TOP,
        "radius_bot": RADIUS_BOT,
        "radius_small": RADIUS_SMALL,
        "radius_large": RADIUS_LARGE,
        "model_top": MODEL_TOP,
        "model_bot": MODEL_BOT,
        "volume_fraction_top": VOLUME_FRACTION_TOP,
        "volume_fraction_bot": VOLUME_FRACTION_BOT,
        "volume_fraction_small": VOLUME_FRACTION_SMALL,
        "volume_fraction_large": VOLUME_FRACTION_LARGE,
        "mu_s_small": MU_S_SMALL,
        "mu_s_large": MU_S_LARGE,
        "l_s_small": L_S_SMALL,
        "l_s_large": L_S_LARGE,
        "l_s_top": L_S_TOP,
        "l_s_bot": L_S_BOT,
        "invariant": "constant_mu_s_per_layer" if FAMILY == "matched_mus" else "constant_volume_fraction",
        "l_star_top": L_STAR_TOP,
        "l_star_bot": L_STAR_BOT,
        "l_star_small": L_STAR_SMALL,
        "l_star_large": L_STAR_LARGE,
        "l_star_contrast": LSTAR_CONTRAST,
        "l_star_angle_anchor": L_STAR_ANGLE_ANCHOR,
        "l_star_time_anchor": L_STAR_TIME_ANCHOR,
        "angle_anchor": f"min_lstar_species ({ANGLE_ANCHOR_SPECIES})",
        "time_anchor": f"max_lstar_species ({TIME_ANCHOR_SPECIES})",
        "thickness_basis": "L_SCALE_REF = l_s_small(phi=0.10)",
        "l_scale_ref": L_SCALE_REF,
        "M_top": M_TOP,
        "mult_reach": MULT_REACH,
        "z_probe": Z_PROBE,
        "z_probe_common": Z_PROBE_COMMON,
        "laser_radius_um": LASER_RADIUS,
        "theta_1": THETA_1,
        "theta_2": THETA_2,
        "d_theta_1": D_THETA_1,
        "d_theta_2": D_THETA_2,
        "d_phi": D_PHI,
        "n_theta_1": N_THETA_1,
        "n_theta_2": N_THETA_2,
        "n_phi": N_PHI,
        "q_fine": Q_FINE,
        "q_tail": Q_TAIL,
        "q_narrow": Q_NARROW,
        "bins_across_narrow": BINS_ACROSS_NARROW,
        "t_max": T_MAX,
        "d_time": D_TIME,
        "time_grid": GRID,
        "polarization": POLARIZATION,
        "n_photons": N_PHOTONS,
        "seed": config.seed,
        "replica": rep,
    }


# ===========================================================================
# Corridas
# ===========================================================================
def run_two_layers(exp: Experiment, z_interface: float, mult_index: int, rep: int):
    """Dos capas homogeneas apiladas; interfaz a z_interface [um]."""

    sample = Sample(N_MEDIUM)
    sample.add_layer(SPECIES_TOP, 0.0, z_interface)
    sample.add_layer(SPECIES_BOT, z_interface, float("inf"))

    laser = Laser(LASER_M, LASER_N, WAVELENGTH, LASER_RADIUS, LASER_TYPE)
    sens, det_1, det_2, stats = make_sensors()
    config = base_config(sample, laser, sens, SEED_BASE_STRAT + mult_index * 1000 + rep)

    extra = common_extra(config, rep)
    extra.update({
        "z_interface": z_interface,
        "thickness_in_ref": z_interface / L_SCALE_REF,
        "thickness_in_ls_top": z_interface / L_S_TOP,
        "mult_local": z_interface / L_STAR_TOP,
        "reachable_self": bool(z_interface <= Z_PROBE),
        "reachable_paired": bool(z_interface <= Z_PROBE_COMMON),
        "layer_kind": "stratified_two_layers",
    })
    exp.save_params(config, extra=extra)

    t0 = time.time()
    run_simulation_parallel(config)
    print("runtime_s:", time.time() - t0, "| hits:", det_1.hits + det_2.hits)

    exp.save_sensors({"farfield_cbs_1": det_1, "farfield_cbs_2": det_2, "statistics": stats})
    exp.save_processed("farfield_cbs_1", postprocess_farfield_cbs(det_1, N_PHOTONS), sensor=det_1)
    exp.save_processed("farfield_cbs_2", postprocess_farfield_cbs(det_2, N_PHOTONS), sensor=det_2)

    _keep_alive = (sample, laser, sens)
    del _keep_alive


def run_one_layer(exp: Experiment, rep: int, specie, which: str, seed_offset: int):
    """Capa unica semi-infinita (control), etiquetada por ESPECIE. Dentro de una
    familia los controles son el mismo medio en las 4 campañas y pueden
    agruparse como replicas independientes."""

    sample = Sample(N_MEDIUM)
    sample.add_layer(specie, 0.0, float("inf"))

    laser = Laser(LASER_M, LASER_N, WAVELENGTH, LASER_RADIUS, LASER_TYPE)
    sens, det_1, det_2, stats = make_sensors()
    config = base_config(sample, laser, sens, SEED_BASE_STRAT + seed_offset + rep)

    extra = common_extra(config, rep)
    extra.update({
        "z_interface": None,
        "thickness_in_ref": None,
        "thickness_in_ls_top": None,
        "mult_local": None,
        "reachable_self": None,
        "reachable_paired": None,
        "layer_kind": "stratified_one_layer",
        "single_layer_species": which,
    })
    exp.save_params(config, extra=extra)

    t0 = time.time()
    run_simulation_parallel(config)
    print("runtime_s:", time.time() - t0, "| hits:", det_1.hits + det_2.hits)

    exp.save_sensors({"farfield_cbs_1": det_1, "farfield_cbs_2": det_2, "statistics": stats})
    exp.save_processed("farfield_cbs_1", postprocess_farfield_cbs(det_1, N_PHOTONS), sensor=det_1)
    exp.save_processed("farfield_cbs_2", postprocess_farfield_cbs(det_2, N_PHOTONS), sensor=det_2)

    _keep_alive = (sample, laser, sens)
    del _keep_alive


# ===========================================================================
# README
# ===========================================================================
sweep.log_readme(
    f"CBS estratificado -- familia {FAMILY.upper()}, orden {ORDER.upper()}, "
    f"polarizacion {POLARIZATION.upper()} (m=1/sqrt2, n=i/sqrt2), estimador. Una de 8 "
    f"campañas (2 familias x 2 ordenes x 2 polarizaciones) generadas del mismo "
    f"template. INVARIANTE de esta familia: "
    f"{'mu_s identico en ambas capas (phi_large derivada de phi_large=4 a_large mu_s_ref/(3 Q_large))' if FAMILY == 'matched_mus' else 'phi=0.10 identica en ambas capas (mu_s NO igualado)'}. "
    f"Especies: PEQUEÑA r={RADIUS_SMALL} um [{MODEL_SMALL.upper()}] "
    f"phi={VOLUME_FRACTION_SMALL:.6f} l_s={L_S_SMALL:.2f} um l*={L_STAR_SMALL:.2f} um "
    f"g={DQ_SMALL['anisotropy_g']:.4f}; GRANDE r={RADIUS_LARGE} um "
    f"[{MODEL_LARGE.upper()}] phi={VOLUME_FRACTION_LARGE:.6f} l_s={L_S_LARGE:.2f} um "
    f"l*={L_STAR_LARGE:.2f} um g={DQ_LARGE['anisotropy_g']:.4f}. Contraste de "
    f"transporte l*_max/l*_min = {LSTAR_CONTRAST:.2f}. Apilamiento: TOP (entra la "
    f"luz) r={RADIUS_TOP} um, BOT semi-infinita r={RADIUS_BOT} um. ESPESOR en "
    f"unidades de L_SCALE_REF={L_SCALE_REF:.2f} um (l_s de la pequeña a phi=0.10), "
    f"referencia COMUN a las 8 campañas -> las profundidades ABSOLUTAS z de "
    f"{Z_INTERFACES[0]:.3f} a {Z_INTERFACES[-1]:.1f} um coinciden en todas. "
    f"Ventana angular anclada al MIN de l* sobre las ESPECIES "
    f"(l*_{ANGLE_ANCHOR_SPECIES}={L_STAR_ANGLE_ANCHOR:.2f} um, invariante al orden): "
    f"fina q in [0,{Q_FINE}] ({N_THETA_1} bins, d_theta_1={np.rad2deg(D_THETA_1):.6f} "
    f"deg), cola q in [{Q_FINE},{Q_TAIL}] ({N_THETA_2} bins). El cono ESTRECHO ocupa "
    f"q~{Q_NARROW:.3f} = {BINS_ACROSS_NARROW:.0f} bins. Grilla TEMPORAL anclada al MAX de "
    f"l* (l*_{TIME_ANCHOR_SPECIES}={L_STAR_TIME_ANCHOR:.2f} um): {TIME_NBINS} bins hasta "
    f"{TIME_TMAX_TAUSTAR} tau* (bin 0 = integrado). Grillas IDENTICAS bin a bin en "
    f"las 4 campañas de esta familia (fingerprint {GRID_FINGERPRINT}); ENTRE "
    f"familias las anclas difieren, asi que la comparacion cruzada solo vale en "
    f"unidades reducidas (q, t/tau*) o sobre escalares adimensionales. Alcance "
    f"difusivo de este orden {Z_PROBE:.0f} um; alcance PAREADO normal/inverso "
    f"{Z_PROBE_COMMON:.0f} um: mas alla, la interfaz es control semi-infinito y no "
    f"punto de comparacion. {N_PHOTONS:,} fotones, {N_REPLICAS} replicas/espesor, "
    f"semillas SEED_BASE_STRAT={SEED_BASE_STRAT} + 1000*mult_index + rep."
)


# ===========================================================================
# Loop
# ===========================================================================
run_counter = 0
for index, z_interface in enumerate(Z_INTERFACES):
    for rep in range(N_REPLICAS):
        name = f"z_interface_{z_interface:.2f}__rep{rep}"
        print(f"\n\n=== Corrida: {name} ===")
        sweep.run(run_counter, name,
                  lambda exp, z=z_interface, i=index, rep=rep: run_two_layers(exp, z, i, rep))
        run_counter += 1


# Capa unica (control), etiquetada por especie
for rep in range(N_REPLICAS):
    name = f"single_layer_small__rep{rep}"
    print(f"\n\n=== Corrida: {name} ===")
    sweep.run(run_counter, name,
              lambda exp, rep=rep: run_one_layer(exp, rep, SPECIES_SMALL, "small", 900_000))
    run_counter += 1

    name = f"single_layer_large__rep{rep}"
    print(f"\n\n=== Corrida: {name} ===")
    sweep.run(run_counter, name,
              lambda exp, rep=rep: run_one_layer(exp, rep, SPECIES_LARGE, "large", 950_000))
    run_counter += 1
