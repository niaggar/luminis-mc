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
EXP_NAME = "study_two_layers__PLIN"
BASE_DIR = "/home/niaggar/Developer/luminis-mc/temporal_results"

sweep = SweepManager(EXP_NAME, BASE_DIR, timestamped=False)
sweep.snapshot_master_script(__main__.__file__)
sweep.log_readme(
    "CBS estratificado NORMAL -- dos capas apiladas (pol. lineal, estimador). "
    "TOP (entra la luz) r=0.035 um = LENTA (l* mayor); BOT semi-infinita r=0.075 um. "
    "Barrido del espesor de TOP en unidades de l*_top. Rejilla temporal UNICA anclada "
    "a la especie mas lenta (max l*), IDENTICA al experimento inverso -> ejes de tiempo "
    "comparables. Alcance en profundidad medido en el medio de entrada (M_top)."
)

# ===========================================================================
# Geometria de capas
#   TOP = superior (entra la luz);  BOT = inferior semi-infinita.
#   Este es el UNICO bloque que difiere del script inverso.
# ===========================================================================
RADIUS_TOP = 0.035               # um  (lenta, l* mayor)
RADIUS_BOT = 0.075               # um  (rapida)

# ===========================================================================
# Parametros fisicos (comunes a ambos experimentos)
# ===========================================================================
VOLUME_FRACTION = 0.10
N_PARTICLE = 1.59
N_MEDIUM = 1.33
WAVELENGTH = 0.514               # um
MU_A_PERCENT = 0.0               # sin absorcion (gate de reciprocidad limpio)

# espesor de TOP en unidades de l*_top
thickness_multipliers = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]

# ---------------------------------------------------------------------------
# Polarizacion: LINEAL a lo largo de m (X), incidencia normal.
# ---------------------------------------------------------------------------
LASER_M = 1.0
LASER_N = 0.0
LASER_RADIUS_MFP = 4.0           # radio del haz en unidades de l* del ANCLA (max l*)
LASER_TYPE = LaserSource.Gaussian

# Funcion de fase
PHASEF_THETA_MIN = 0.0
PHASEF_THETA_MAX = np.pi
PHASEF_NDIV = 10_000

# ---------------------------------------------------------------------------
# Grilla angular
#   N_THETA=100 (0.01 deg/celda) resuelve de sobra el cono; 1000 reparte los pocos
#   hits del cono 10x mas fino, peor aun con bins temporales (theta x t).
# ---------------------------------------------------------------------------
N_THETA = 1000
N_PHI = 36
PHI_MAX = 2 * np.pi
THETA_MAX = np.deg2rad(1)

# ---------------------------------------------------------------------------
# Grilla temporal (COMPARTIDA por todos los espesores Y por el experimento inverso)
#   Anclada a la especie mas lenta (max l*). Como ambos experimentos usan las mismas
#   dos especies, max(l*) es el mismo -> t_max y dt IDENTICOS -> conos superponibles.
#
#   Acople profundidad<->ventana: para sondear una interfaz a mult*l*_top hace falta
#   M_top >= (3/2) mult^2, con M_top = t_max / tau*_top (en unidades del medio de ENTRADA).
# ---------------------------------------------------------------------------
TIME_NBINS = 40                  # ~1.3 tau*_ancla / bin
TIME_TMAX_TAUSTAR = 40           # M: ventana en tau* del ancla (max l*)

# ---------------------------------------------------------------------------
# Muestreo
# ---------------------------------------------------------------------------
N_PHOTONS = 100_000
N_THREADS = 44

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


# ===========================================================================
# Setup COMPARTIDO (una sola vez)
#   Las especies dependen solo del radio -> se construyen una vez.
#   Scope de modulo -> siempre vivas para el binding (keep-alive de raiz).
# ===========================================================================
SPECIES_TOP = build_species(RADIUS_TOP)
SPECIES_BOT = build_species(RADIUS_BOT)
DQ_TOP = derived_quantities(SPECIES_TOP, VOLUME_FRACTION)
DQ_BOT = derived_quantities(SPECIES_BOT, VOLUME_FRACTION)
set_albedo(SPECIES_TOP, MU_A_PERCENT, DQ_TOP['mean_free_path'])
set_albedo(SPECIES_BOT, MU_A_PERCENT, DQ_BOT['mean_free_path'])

L_STAR_TOP = DQ_TOP['transport_mean_free_path']   # medio de entrada (fija espesor y alcance)
L_STAR_BOT = DQ_BOT['transport_mean_free_path']
L_STAR_ANCHOR = max(L_STAR_TOP, L_STAR_BOT)       # tiempo <- especie MAS LENTA

# rejilla temporal (identica en normal e inverso, porque max l* es el mismo)
GRID = build_time_grid(
    L_STAR_ANCHOR, N_MEDIUM,
    n_bins=TIME_NBINS, t_max_taustar=TIME_TMAX_TAUSTAR, binning="geometric",
)
# haz anclado al l* MAYOR: ancho vs. cualquiera de los dos medios, e identico en
# ambos experimentos (evita limitar el cono por tamano de haz en el medio lento).
LASER_RADIUS_UM = LASER_RADIUS_MFP * L_STAR_ANCHOR

# --- alcance en profundidad, medido en el medio de ENTRADA (top) ---
tau_top_fs = L_STAR_TOP * N_MEDIUM / C0
M_top = GRID["t_max_fs"] / tau_top_fs             # ventana en unidades de tau*_top
MULT_REACH = np.sqrt(2.0 * M_top / 3.0)           # mult maximo con interfaz alcanzable
z_probe = MULT_REACH * L_STAR_TOP                 # alcance difusivo RMS en el top

print("==== Setup (dos capas, NORMAL) ====")
print(f"TOP (entra luz): r={RADIUS_TOP} um  l*_top={L_STAR_TOP:.2f} um  g={DQ_TOP['anisotropy_g']:.4f}")
print(f"BOT (semi-inf) : r={RADIUS_BOT} um  l*_bot={L_STAR_BOT:.2f} um  g={DQ_BOT['anisotropy_g']:.4f}")
print(f"ancla tiempo   : max l* = {L_STAR_ANCHOR:.2f} um")
print(f"radio de haz   : {LASER_RADIUS_UM:.2f} um (= {LASER_RADIUS_MFP} max-l*)")
print(f"ventana        : {GRID['t_max_fs']:.0f} fs = {M_top:.1f} tau*_top  ->  "
      f"z_probe = {z_probe:.1f} um = {MULT_REACH:.2f} l*_top")
print(f"GRID: {GRID}")
print("  mult | z_interfaz [um] | M_top necesario | interfaz alcanzable?")
for _m in thickness_multipliers:
    _ok = "SI" if _m <= MULT_REACH else "NO (control semi-inf)"
    print(f"  {_m:5.1f} | {_m * L_STAR_TOP:13.1f} | {1.5 * _m ** 2:15.1f} | {_ok}")


# ===========================================================================
# Corrida
# ===========================================================================
def run_two_layers(exp: Experiment, thickness_multi: float):
    """Dos capas homogeneas apiladas; interfaz a mult*l*_top. Corre CBS y persiste."""
    z_interface = thickness_multi * L_STAR_TOP

    sample = Sample(N_MEDIUM)
    sample.add_layer(SPECIES_TOP, 0.0, z_interface)
    sample.add_layer(SPECIES_BOT, z_interface, float("inf"))

    reachable = thickness_multi <= MULT_REACH
    print(f"\nmult={thickness_multi:.2f}  interfaz z={z_interface:.1f} um "
          f"({thickness_multi:.2f} l*_top)  alcanzable={'SI' if reachable else 'NO'}")

    laser = Laser(LASER_M, LASER_N, WAVELENGTH, LASER_RADIUS_UM, LASER_TYPE)

    # --- grilla angular + temporal (compartida) ---
    d_theta = THETA_MAX / N_THETA
    d_phi = PHI_MAX / N_PHI
    t_max = GRID["t_max_sim"]
    dt = GRID["dt_sim"]

    # --- sensores ---
    sens = SensorsGroup()
    det = sens.add_detector(
        FarFieldCBSSensor(THETA_MAX, PHI_MAX, t_max, d_theta, d_phi, dt, True)
    )
    det.set_theta_limit(0, THETA_MAX)
    det.set_phi_slices([0, np.pi / 4, np.pi / 2])

    stats = sens.add_detector(StatisticsSensor(z=0, absorb=True))
    stats.set_theta_limit(0, THETA_MAX)

    # --- config ---
    config = SimConfig()
    config.n_photons = N_PHOTONS
    config.sample = sample
    config.detector = sens
    config.laser = laser
    config.track_reverse_paths = True          # imprescindible para CBS
    config.pin_threads_to_cores = False
    config.n_threads = N_THREADS
    config.show_progress = True

    extra = {
        "dq_top": DQ_TOP,
        "dq_bot": DQ_BOT,
        "radius_top": RADIUS_TOP,
        "radius_bot": RADIUS_BOT,
        "thickness_multi": thickness_multi,
        "z_interface": z_interface,
        "l_star_top": L_STAR_TOP,
        "l_star_bot": L_STAR_BOT,
        "l_star_anchor": L_STAR_ANCHOR,
        "M_top": M_top,
        "mult_reach": MULT_REACH,
        "interface_reachable": bool(reachable),
        "laser_radius_um": LASER_RADIUS_UM,
        "theta_max": THETA_MAX,
        "d_theta": d_theta,
        "d_phi": d_phi,
        "n_theta": N_THETA,
        "n_phi": N_PHI,
        "t_max": t_max,
        "d_time": dt,
        "time_grid": GRID,
        "polarization": "linear",
        "layer_kind": "stratified_two_layers",
        "order": "normal",
    }
    exp.save_params(config, extra=extra)

    # --- run ---
    t0 = time.time()
    run_simulation_parallel(config)
    print("runtime_s:", time.time() - t0, "| hits:", det.hits)

    # --- guardar RAW + procesados ---
    exp.save_sensors({"farfield_cbs": det, "statistics": stats})
    exp.save_processed("farfield_cbs", postprocess_farfield_cbs(det, N_PHOTONS), sensor=det)

    _keep_alive = (sample,)
    del _keep_alive


# ===========================================================================
# Loop  (m=mult: default-binding correcto)
# ===========================================================================
for index, mult in enumerate(thickness_multipliers):
    run_name = f"multiplier_{mult:.2f}"
    print(f"\n\n=== Corrida: {run_name} ===")
    sweep.run(index, run_name, lambda exp, m=mult: run_two_layers(exp, thickness_multi=m))