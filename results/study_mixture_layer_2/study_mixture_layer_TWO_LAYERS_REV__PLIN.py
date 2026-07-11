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

from utils.time import build_time_grid

set_log_level(LogLevel.info)

# ===========================================================================
# Salida
# ===========================================================================
EXP_NAME = "study_two_layers_REV__PLIN__beam2500"
BASE_DIR = "/home/niaggar/Developer/luminis-mc/temporal_results"

sweep = SweepManager(EXP_NAME, BASE_DIR, timestamped=False)
sweep.snapshot_master_script(__main__.__file__)

# ===========================================================================
# Geometria de capas
#   TOP = superior (entra la luz);  BOT = inferior semi-infinita.
#   UNICO bloque que difiere del script inverso.
# ===========================================================================
RADIUS_TOP = 0.075               # um  (l* mayor -> cono ESTRECHO, dinamica LENTA)
RADIUS_BOT = 0.035               # um  (l* menor -> cono ANCHO, dinamica RAPIDA)

# ===========================================================================
# Parametros fisicos
# ===========================================================================
VOLUME_FRACTION = 0.10
N_PARTICLE = 1.59
N_MEDIUM = 1.33
WAVELENGTH = 0.514               # um
MU_A_PERCENT = 0.0               # sin absorcion (gate de reciprocidad limpio)

# espesor de TOP en unidades de l*_top. Densificado en el rango intermedio
# (1-5) donde vive la firma de interfaz; 5,10 son control semi-infinito.
thickness_multipliers = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]

# ---------------------------------------------------------------------------
# Polarizacion: LINEAL (m=1, n=0). Etiqueta derivada del laser -> no diverge.
# ---------------------------------------------------------------------------
LASER_M = 1.0
LASER_N = 0.0
LASER_TYPE = LaserSource.Gaussian
POLARIZATION = "linear" if (int(LASER_M), int(LASER_N)) == (1, 0) else "circular"

# ---------------------------------------------------------------------------
# Haz FIJO en um (aparato fijo = fidelidad experimental), consistente con las
# demas tandas. w/l* >> 1 en ambos medios -> onda plana, sin limitar el cono.
# ---------------------------------------------------------------------------
LASER_RADIUS = 2500              # um

# Funcion de fase
PHASEF_THETA_MIN = 0.0
PHASEF_THETA_MAX = np.pi
PHASEF_NDIV = 10_000

# ---------------------------------------------------------------------------
# Grilla angular COMPARTIDA, en unidades reducidas q = k*l*_ancla*theta.
#   Ancla ANGULAR = min(l*) = l*_bot (cono mas ANCHO) -> la ventana contiene el
#   cono mas ancho de todo el barrido Y el ensanchamiento de tiempos tempranos.
#   OJO resolucion: el limite grueso (domina top, l* grande) tiene cono ~5x mas
#   angosto y queda sub-resuelto en la ventana fina; ese limite es el control
#   homogeneo-top ya caracterizado en §5.1, asi que es aceptable. La fisica de
#   interfaz vive en el rango delgado-intermedio (cono ancho, bien resuelto).
#   N_THETA_1 alto aqui porque el time-resolved reparte hits en (theta x t).
# ---------------------------------------------------------------------------
N_THETA_1 = 400                  # ventana fina (cono); mas que en §5.3 por el 3D
N_THETA_2 = 100                  # ventana cola (fondo/baseline)
Q_FINE, Q_TAIL = 7.5, 40.0
N_PHI = 36
PHI_MAX = 2 * np.pi
k = 2 * np.pi * N_MEDIUM / WAVELENGTH

# ---------------------------------------------------------------------------
# Grilla temporal COMPARTIDA (todos los espesores Y el experimento inverso).
#   Ancla TEMPORAL = max(l*) = l*_top (dinamica mas LENTA). Anclas OPUESTAS a la
#   angular. Como ambos experimentos usan las mismas dos especies, max(l*) es el
#   mismo -> t_max y dt IDENTICOS -> Delta_theta(t) comparable normal vs inverso.
#
#   Acople profundidad<->ventana: sondear una interfaz a mult*l*_top requiere
#   M_top >= (3/2) mult^2, con M_top = t_max/tau*_top (medio de ENTRADA).
#   El t* del quiebre de Delta_theta(t) escala con mult^2: firma de la interfaz.
# ---------------------------------------------------------------------------
TIME_NBINS = 100                  # resolucion temporal para ver el interior
TIME_TMAX_TAUSTAR = 40           # M: ventana en tau* del ancla (max l*)

# ---------------------------------------------------------------------------
# Muestreo
#   300k (mas que §5.3): el sensor 3D (theta x t) reparte hits -> los bins
#   temporales tardios necesitan mas fotones para no diluirse.
# ---------------------------------------------------------------------------
N_PHOTONS = 300_000
N_THREADS = 46
N_REPLICAS = 5
SEED_BASE_STRAT = 20260712       # distinto de homogeneo (…10) y mezcla (…11)

C0 = 0.299792458                 # um/fs


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
    """Albedo bien definido + camino libre medio de la capa homogenea.
    La escala absoluta de mu_s no afecta el transporte; solo el cociente mu_a/mu_s."""
    medium.set_absorption_coefficient(mu_a_percent)
    medium.set_scattering_coefficient(1.0 - mu_a_percent)
    medium.set_mean_free_path(mfp)


def make_sensors(theta_1, theta_2):
    """Doble detector far-field (fino + cola) + estadistica. Grilla COMPARTIDA
    (angular anclada a min l*, temporal a max l*)."""
    d_theta_1 = theta_1 / N_THETA_1
    d_theta_2 = theta_2 / N_THETA_2
    d_phi = PHI_MAX / N_PHI
    t_max = GRID["t_max_sim"]
    dt = GRID["dt_sim"]

    sens = SensorsGroup()
    det_1 = sens.add_detector(FarFieldCBSSensor(theta_1, PHI_MAX, t_max, d_theta_1, d_phi, dt, True))
    det_1.set_theta_limit(0, theta_1)
    det_1.set_phi_slices([0, np.pi / 4, np.pi / 2])

    det_2 = sens.add_detector(FarFieldCBSSensor(theta_2, PHI_MAX, t_max, d_theta_2, d_phi, dt, True))
    det_2.set_theta_limit(theta_1 * 0.9, theta_2)          # solape para stitching
    det_2.set_phi_slices([0, np.pi / 4, np.pi / 2])

    stats = sens.add_detector(StatisticsSensor(z=0, absorb=True))
    stats.set_theta_limit(0, theta_2)
    return sens, det_1, det_2, stats, d_theta_1, d_theta_2, d_phi


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
SPECIES_TOP = build_species(RADIUS_TOP)
SPECIES_BOT = build_species(RADIUS_BOT)
DQ_TOP = derived_quantities(SPECIES_TOP, VOLUME_FRACTION)
DQ_BOT = derived_quantities(SPECIES_BOT, VOLUME_FRACTION)
set_albedo(SPECIES_TOP, MU_A_PERCENT, DQ_TOP['mean_free_path'])
set_albedo(SPECIES_BOT, MU_A_PERCENT, DQ_BOT['mean_free_path'])

L_STAR_TOP = DQ_TOP['transport_mean_free_path']   # medio de entrada (fija espesor y alcance)
L_STAR_BOT = DQ_BOT['transport_mean_free_path']

# --- anclas OPUESTAS ---
L_STAR_TIME_ANCHOR = max(L_STAR_TOP, L_STAR_BOT)  # tiempo <- dinamica mas LENTA
L_STAR_ANGLE_ANCHOR = min(L_STAR_TOP, L_STAR_BOT) # angulo <- cono mas ANCHO

# ventana temporal (identica normal e inverso: max l* es el mismo)
GRID = build_time_grid(
    L_STAR_TIME_ANCHOR, N_MEDIUM,
    n_bins=TIME_NBINS, t_max_taustar=TIME_TMAX_TAUSTAR
)
# ventana angular en unidades reducidas, anclada al cono mas ancho
THETA_1 = Q_FINE / (k * L_STAR_ANGLE_ANCHOR)
THETA_2 = Q_TAIL / (k * L_STAR_ANGLE_ANCHOR)

# --- alcance en profundidad, medido en el medio de ENTRADA (top) ---
tau_top_fs = L_STAR_TOP * N_MEDIUM / C0
M_top = GRID["t_max_fs"] / tau_top_fs             # ventana en unidades de tau*_top
MULT_REACH = np.sqrt(2.0 * M_top / 3.0)           # mult con interfaz alcanzable (RMS)
z_probe = MULT_REACH * L_STAR_TOP

print("==== Setup (dos capas, NORMAL) ====")
print(f"TOP (entra luz): r={RADIUS_TOP} um  l*_top={L_STAR_TOP:.2f} um  g={DQ_TOP['anisotropy_g']:.4f}  (estrecho/lento)")
print(f"BOT (semi-inf) : r={RADIUS_BOT} um  l*_bot={L_STAR_BOT:.2f} um  g={DQ_BOT['anisotropy_g']:.4f}  (ancho/rapido)")
print(f"ancla tiempo (max l*): {L_STAR_TIME_ANCHOR:.2f} um   ancla angulo (min l*): {L_STAR_ANGLE_ANCHOR:.2f} um")
print(f"haz: {LASER_RADIUS} um   w/l*_top={LASER_RADIUS/L_STAR_TOP:.1f}  w/l*_bot={LASER_RADIUS/L_STAR_BOT:.1f}")
print(f"ventana angular: theta_1={np.rad2deg(THETA_1):.3f} deg  theta_2={np.rad2deg(THETA_2):.3f} deg")
print(f"ventana temporal: {GRID['t_max_fs']:.0f} fs = {M_top:.1f} tau*_top  ->  "
      f"z_probe = {z_probe:.1f} um = {MULT_REACH:.2f} l*_top")
print(f"GRID: dt={GRID['dt_fs']:.2f} fs  t_max={GRID['t_max_fs']:.0f} fs  tau*={GRID['tau_star_fs']:.2f} fs")

L_STAR_TOP_NORMAL = L_STAR_BOT
Z_INTERFACES = [m * L_STAR_TOP_NORMAL for m in thickness_multipliers]

print("  z[um] | mult_top_local | M_top nec. | alcanzable?")
for z in Z_INTERFACES:
    mult_local = z / L_STAR_TOP           # espesor en l* del medio de entrada REAL
    ok = "SI" if mult_local <= MULT_REACH else "NO (control semi-inf)"
    print(f"  {z:7.1f} | {mult_local:14.2f} | {1.5*mult_local**2:10.1f} | {ok}")



# ===========================================================================
# Corrida
# ===========================================================================
def run_two_layers(exp: Experiment, z_interface: float, mult_index: int, rep: int):
    """Dos capas homogeneas apiladas; interfaz a mult*l*_top. Corre CBS y persiste."""

    sample = Sample(N_MEDIUM)
    sample.add_layer(SPECIES_TOP, 0.0, z_interface)
    sample.add_layer(SPECIES_BOT, z_interface, float("inf"))

    laser = Laser(LASER_M, LASER_N, WAVELENGTH, LASER_RADIUS, LASER_TYPE)
    sens, det_1, det_2, stats, d_theta_1, d_theta_2, d_phi = make_sensors(THETA_1, THETA_2)
    config = base_config(sample, laser, sens, SEED_BASE_STRAT + mult_index * 1000 + rep)

    mult_local = z_interface / L_STAR_TOP           # espesor en l* del medio de entrada REAL

    extra = {
        "dq_top": DQ_TOP,
        "dq_bot": DQ_BOT,
        "radius_top": RADIUS_TOP,
        "radius_bot": RADIUS_BOT,
        "z_interface": z_interface,
        "l_star_top": L_STAR_TOP,
        "l_star_bot": L_STAR_BOT,
        "l_star_time_anchor": L_STAR_TIME_ANCHOR,
        "l_star_angle_anchor": L_STAR_ANGLE_ANCHOR,
        "M_top": M_top,
        "mult_local": mult_local,
        "mult_reach": MULT_REACH,
        "laser_radius_um": LASER_RADIUS,
        "theta_1": THETA_1,
        "theta_2": THETA_2,
        "d_theta_1": d_theta_1,
        "d_theta_2": d_theta_2,
        "d_phi": d_phi,
        "n_theta_1": N_THETA_1,
        "n_theta_2": N_THETA_2,
        "n_phi": N_PHI,
        "q_fine": Q_FINE,
        "q_tail": Q_TAIL,
        "t_max": GRID["t_max_sim"],
        "d_time": GRID["dt_sim"],
        "time_grid": GRID,
        "time_anchor": "max_lstar (top)",
        "angle_anchor": "min_lstar (bot)",
        "polarization": POLARIZATION,
        "seed": config.seed,
        "replica": rep,
        "layer_kind": "stratified_two_layers",
        "order": "normal",
    }
    exp.save_params(config, extra=extra)

    t0 = time.time()
    run_simulation_parallel(config)
    print("runtime_s:", time.time() - t0, "| hits:", det_1.hits + det_2.hits)

    exp.save_sensors({"farfield_cbs_1": det_1, "farfield_cbs_2": det_2, "statistics": stats})
    exp.save_processed("farfield_cbs_1", postprocess_farfield_cbs(det_1, N_PHOTONS), sensor=det_1)
    exp.save_processed("farfield_cbs_2", postprocess_farfield_cbs(det_2, N_PHOTONS), sensor=det_2)

    _keep_alive = (sample,)
    del _keep_alive


# ===========================================================================
# README (f-string: no se desactualiza en lo que importa)
# ===========================================================================
sweep.log_readme(
    f"CBS estratificado NORMAL -- dos capas apiladas, polarizacion "
    f"{POLARIZATION.upper()} (m={LASER_M:.0f}, n={LASER_N:.0f}), estimador. "
    f"TOP (entra luz) r={RADIUS_TOP} um = l* mayor (cono estrecho, dinamica lenta); "
    f"BOT semi-infinita r={RADIUS_BOT} um = l* menor (cono ancho, rapido). "
    f"Barrido del espesor de TOP en unidades de l*_top. Haz gaussiano FIJO "
    f"w={LASER_RADIUS} um (onda plana). Doble ventana angular COMPARTIDA en "
    f"unidades reducidas anclada al cono MAS ANCHO = min(l*) = l*_bot "
    f"({L_STAR_ANGLE_ANCHOR:.2f} um): fina q in [0,{Q_FINE}] ({N_THETA_1} bins), "
    f"cola q in [{Q_FINE},{Q_TAIL}] ({N_THETA_2} bins). Grilla TEMPORAL "
    f"COMPARTIDA anclada a la dinamica mas LENTA = max(l*) = l*_top "
    f"({L_STAR_TIME_ANCHOR:.2f} um, ancla OPUESTA a la angular), IDENTICA al "
    f"experimento inverso -> Delta_theta(t) comparable: {TIME_NBINS} bins "
    f"geometricos hasta {TIME_TMAX_TAUSTAR} tau* (bin 0 = integrado; 1..N = "
    f"time-resolved, observable interno). Acople profundidad<->tiempo: interfaz "
    f"a mult*l*_top requiere M_top>=(3/2)mult^2; el t* del quiebre de "
    f"Delta_theta(t) escala con mult^2 (firma de la interfaz enterrada). "
    f"{N_REPLICAS} replicas/espesor, semillas SEED_BASE_STRAT={SEED_BASE_STRAT} "
    f"+ 1000*mult_index + rep, en params. Alcance en profundidad medido en el "
    f"medio de entrada (M_top)."
)


# ===========================================================================
# Loop
# ===========================================================================
run_counter = 0
for index, z_interface in enumerate(Z_INTERFACES):
    for rep in range(N_REPLICAS):
        name = f"z_interface_{z_interface:.2f}__rep{rep}"
        print(f"\n\n=== Corrida: {name} ===")
        sweep.run(run_counter, name, lambda exp, z=z_interface, i=index, rep=rep: run_two_layers(exp, z, i, rep))
        run_counter += 1