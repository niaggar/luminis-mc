import __main__
import os
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
#   CONVERGENCIA y COSTO COMPUTACIONAL de ESTIMATOR (next-event) vs ANALOG en
#   el medio de DOS CAPAS. A diferencia del chequeo de insesgadez, aqui SI hay
#   escalera: UNA sola muestra y UNA sola grilla angular, barriendo el numero
#   de fotones. Observables del estudio:
#     (a) sesgo/convergencia: perfil eta(q) en cada N vs la referencia (analog
#         a N_max y estimator a N_max) -> debe colapsar al MISMO limite;
#     (b) ruido: sigma del escalar (eta(0), FWHM) vs N -> debe caer como N^-1/2;
#     (c) costo: wall/CPU time vs N -> lineal en N; y la figura de merito
#         FOM = 1 / (sigma^2 * T), que es lo que decide cual metodo conviene.
#   El unico grado de libertad que cambia entre corridas es (modo, N, replica).
#   Muestra, grilla angular, laser, semilla-base y n_threads son FIJOS.
#
#   DISENO DE REPLICAS: cada punto de la escalera con N_total fotones se corre
#   como N_REPLICAS trozos INDEPENDIENTES de N_total/N_REPLICAS fotones. Al
#   sumar los trozos se recupera la estadistica de N_total (los acumuladores
#   son aditivos y la normalizacion es por fotón), y la dispersion entre trozos
#   da la BANDA sin costo extra. Es la unica varianza valida con next-event.
#   Los puntos de la escalera son INDEPENDIENTES entre si (no anidados), asi
#   que sigma(N) se puede ajustar como medidas independientes.
# ===========================================================================

# ===========================================================================
# Salida
# ===========================================================================
BASE_DIR = "/home/niaggar/Developer/luminis-mc/temporal_results"

# ===========================================================================
# Parametros fisicos (IDENTICOS a la tanda de insesgadez: misma muestra)
# ===========================================================================
VOLUME_FRACTION = 0.10
N_PARTICLE = 1.59
N_MEDIUM = 1.33
WAVELENGTH = 0.514               # um
MU_A_PERCENT = 0.0               # sin absorcion (gate de reciprocidad limpio)

RADIUS_TOP = 0.035               # um  (l* mayor)
RADIUS_BOT = 0.075               # um  (l* menor -> cono mas ancho = ancla angular)
THICKNESS_MULT = 1.0             # interfaz a 1 l*_top

# ---------------------------------------------------------------------------
# Polarizacion: la etiqueta se DERIVA de (m, n), nunca se escribe a mano.
# (m, n) = (1/sqrt2, i/sqrt2)  ->  CIRCULAR.  (m, n) = (1, 0)  ->  LINEAL.
# ---------------------------------------------------------------------------
LASER_M = 1 / np.sqrt(2)
LASER_N = 1j / np.sqrt(2)
LASER_TYPE = LaserSource.Gaussian
LASER_RADIUS = 2500              # um (FIJO, onda plana, consistente con beam2500)


def polarization_label(m, n):
    """Etiqueta derivada del estado de Jones (evita el desfase comentario/valor)."""
    m, n = complex(m), complex(n)
    if abs(n) < 1e-12 or abs(m) < 1e-12:
        return "linear"
    ratio = n / m
    if abs(ratio.real) < 1e-9 and abs(abs(ratio.imag) - 1.0) < 1e-9:
        return "circular"
    return "elliptical"


POLARIZATION = polarization_label(LASER_M, LASER_N)
EXP_NAME = f"study_convergence_layers__{POLARIZATION.upper()}__beam{LASER_RADIUS}"

# Funcion de fase
PHASEF_THETA_MIN = 0.0
PHASEF_THETA_MAX = np.pi
PHASEF_NDIV = 10_000

# ---------------------------------------------------------------------------
# Grilla angular: UN SOLO detector, grilla UNIFORME en theta desde el pico
# hasta la cola, q = k*l**theta in [0, Q_MAX], anclada al MENOR l* (cono mas
# ancho). Un unico sensor elimina el solape/stitching entre ventanas y la
# posibilidad de que los dos detectores puntuen distinto en modo directo; el
# precio es resolucion vs alcance con un solo d_theta (ver N_THETA abajo).
# FIJA para TODA la campana -> los perfiles de distintos N y distintos modos
# comparten bins y se restan bin a bin.
#
# phi: grilla UNIFORME en ambos modos. NO se usa set_phi_slices: en modo
# DIRECTO (analog) el binning de phi asume la grilla uniforme y una grilla
# explicita mis-bina (§12 del code_description). Con incidencia circular el
# patron es azimutalmente simetrico, asi que se promedian las N_PHI columnas
# en el analisis -> maxima estadistica y observable identico en ambos modos.
#
# COSTO: el estimator fuerza deteccion sobre N_THETA*N_PHI bins EN CADA evento,
# asi que su tiempo por foton es ~lineal en N_THETA. El analog es insensible a
# N_THETA. Subir la resolucion penaliza SOLO al estimator y sesga la comparacion
# de costo a su favor si se elige mal: N_THETA se fija una vez y no se toca.
# ---------------------------------------------------------------------------
Q_MAX = 40.0
N_THETA = 2000                   # dq = 0.01 -> ~750 bins dentro del cono (q<7.5)
N_PHI = 1
PHI_MAX = 2 * np.pi
k = 2 * np.pi * N_MEDIUM / WAVELENGTH

T_MAX = 0.0                      # integrado en tiempo
DT = 0.0

# ---------------------------------------------------------------------------
# Escalera de fotones (TOTAL por punto; cada punto se parte en N_REPLICAS).
# Tope duro del analog: 1e9. Costo total de una escalera geometrica ~1.45x el
# punto mas caro, asi que la escalera casi no cuesta mas que la corrida grande.
# El estimator converge ~1e3 veces mas rapido -> su escalera va 3 decadas abajo.
# ---------------------------------------------------------------------------
N_LADDER = {
    "estimator": [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 500_000],
    "analog":    [1_000_000, 3_000_000, 10_000_000, 30_000_000, 100_000_000, 500_000_000, 1_000_000_000],
}
RUN_MODES = ("estimator", "analog")

N_REPLICAS = 5                   # t_{0.975,4} = 2.78 en el analisis
N_THREADS = 46
SEED_BASE = 20260727

# --- Pilotos de costo: mide fotones/s por modo y proyecta la campana ---------
RUN_PILOT = True
PILOT_N = {"estimator": 2_000, "analog": 2_000_000}
MAX_WALL_HOURS = 0.5            # aborta si la proyeccion excede este techo
FORCE_RUN = False                # True: ignora el techo y corre igual


def make_seed(mode, idx, rep):
    """Streams independientes por (modo, punto de la escalera, replica)."""
    mode_code = 0 if mode == "estimator" else 1
    return SEED_BASE + mode_code * 1_000_000 + idx * 1_000 + rep


def n_tag(n):
    """Etiqueta compacta y ordenable para nombres de corrida: 1e06, 3e08, ..."""
    e = int(np.floor(np.log10(n)))
    m = n / 10 ** e
    return f"{m:.0f}e{e:02d}"


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
    """set_mean_free_path es OBLIGATORIO: sample_free_path usa mfp, no mu_s."""
    medium.set_absorption_coefficient(mu_a_percent)
    medium.set_scattering_coefficient(1.0 - mu_a_percent)
    medium.set_mean_free_path(mfp)


def make_sensors(estimator):
    """UN solo detector far-field (pico -> cola, grilla uniforme) + estadistica,
    integrado en tiempo. Grilla IDENTICA en ambos modos: lo unico que cambia es
    el flag de scoring."""
    d_theta = THETA_MAX / N_THETA
    d_phi = PHI_MAX / N_PHI

    sens = SensorsGroup()
    det = sens.add_detector(FarFieldCBSSensor(THETA_MAX, PHI_MAX, T_MAX, d_theta, d_phi, DT, estimator))
    det.set_theta_limit(0, THETA_MAX)

    stats = sens.add_detector(StatisticsSensor(z=0, absorb=True))
    stats.set_theta_limit(0, THETA_MAX)
    return sens, det, stats, d_theta, d_phi


def build_sample():
    sample = Sample(N_MEDIUM)
    sample.add_layer(SPECIES_TOP, 0.0, Z_INTERFACE)
    sample.add_layer(SPECIES_BOT, Z_INTERFACE, float("inf"))
    return sample


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


# ===========================================================================
# Setup de la muestra (una sola vez; los medios viven a nivel de modulo por la
# keep-alive gotcha de los raw pointers)
# ===========================================================================
SPECIES_TOP = build_species(RADIUS_TOP)
SPECIES_BOT = build_species(RADIUS_BOT)
DQ_TOP = derived_quantities(SPECIES_TOP, VOLUME_FRACTION)
DQ_BOT = derived_quantities(SPECIES_BOT, VOLUME_FRACTION)
set_albedo(SPECIES_TOP, MU_A_PERCENT, DQ_TOP['mean_free_path'])
set_albedo(SPECIES_BOT, MU_A_PERCENT, DQ_BOT['mean_free_path'])

L_STAR_TOP = DQ_TOP['transport_mean_free_path']
L_STAR_BOT = DQ_BOT['transport_mean_free_path']
LSTAR_ANCHOR = min(L_STAR_TOP, L_STAR_BOT)                 # cono mas ancho
THETA_MAX = Q_MAX / (k * LSTAR_ANCHOR)
D_Q = Q_MAX / N_THETA                                      # resolucion en q
Z_INTERFACE = THICKNESS_MULT * L_STAR_TOP


# ===========================================================================
# Corrida unitaria
# ===========================================================================
def run_point(exp, mode, n_photons, idx, rep, timings):
    """Una replica de un punto de la escalera. exp=None -> piloto (no persiste)."""
    estimator = (mode == "estimator")
    seed = make_seed(mode, idx, rep)

    sample = build_sample()
    laser = Laser(LASER_M, LASER_N, WAVELENGTH, LASER_RADIUS, LASER_TYPE)
    sens, det, stats, d_theta, d_phi = make_sensors(estimator)
    config = base_config(sample, laser, sens, seed, n_photons)

    t0, c0 = time.perf_counter(), time.process_time()
    run_simulation_parallel(config)
    wall_s = time.perf_counter() - t0
    cpu_s = time.process_time() - c0

    thr = n_photons / wall_s if wall_s > 0 else float("nan")
    print(f"[{mode:9s}] idx: {idx} rep: {rep:02d} | N: {n_photons:>13,} | "
          f"wall_s: {wall_s:9.2f} | cpu_s: {cpu_s:10.2f} | "
          f"ph/s: {thr:10.3e} | hits: {det.hits}")

    if exp is None:
        del sens, sample, laser, config
        return wall_s

    n_total = n_photons * N_REPLICAS
    extra = {
        "study": "convergence_timing",
        "medium_kind": "two_layers",
        "mode": mode,
        "ladder_index": idx,
        "replica": rep,
        "n_photons": n_photons,              # de ESTA replica
        "n_photons_point_total": n_total,    # del punto completo (suma de replicas)
        "n_replicas": N_REPLICAS,
        "seed": seed,
        "dq_top": DQ_TOP,
        "dq_bot": DQ_BOT,
        "radius_top": RADIUS_TOP,
        "radius_bot": RADIUS_BOT,
        "z_interface": Z_INTERFACE,
        "thickness_mult": THICKNESS_MULT,
        "l_star_top": L_STAR_TOP,
        "l_star_bot": L_STAR_BOT,
        "lstar_angle_anchor": LSTAR_ANCHOR,
        "laser_radius_um": LASER_RADIUS,
        "polarization": POLARIZATION,
        "theta_max": THETA_MAX,
        "d_theta": d_theta, "d_phi": d_phi,
        "n_theta": N_THETA, "n_phi": N_PHI,
        "q_max": Q_MAX, "d_q": D_Q,
        "t_max": T_MAX, "d_time": DT,
        "n_threads": N_THREADS,
        "runtime_s": wall_s,                 # wall clock de la replica
        "cpu_time_s": cpu_s,                 # CPU sumado sobre threads
        "throughput_photons_per_s": thr,
        "hits": int(det.hits),
    }

    exp.save_params(config, extra=extra)
    exp.save_sensors({"farfield_cbs": det, "statistics": stats})
    exp.save_processed("farfield_cbs", postprocess_farfield_cbs(det, n_photons), sensor=det)

    timings.append((mode, idx, rep, n_photons, n_total, wall_s, cpu_s, thr,
                    int(det.hits), seed))

    _keep_alive = (sample, laser, sens); del _keep_alive
    return wall_s


# ===========================================================================
# Piloto de costo
# ===========================================================================
def project_cost():
    """Mide fotones/s por modo con una corrida corta y proyecta la campana."""
    total_h = 0.0
    print("\n==== Piloto de costo ====")
    for mode in RUN_MODES:
        n_pilot = PILOT_N[mode]
        wall = run_point(None, mode, n_pilot, idx=99, rep=99, timings=None)
        rate = n_pilot / wall
        budget = sum(N_LADDER[mode])
        hours = budget / rate / 3600.0
        total_h += hours
        print(f"  {mode:9s}: {rate:.3e} ph/s | presupuesto {budget:,} fotones "
              f"-> {hours:6.2f} h")
    print(f"  TOTAL proyectado: {total_h:.2f} h "
          f"(sin contar overhead por corrida, ~{sum(len(v) for v in N_LADDER.values()) * N_REPLICAS} corridas)")
    return total_h


# ===========================================================================
# Campana
# ===========================================================================
if RUN_PILOT:
    projected_h = project_cost()
    if projected_h > MAX_WALL_HOURS and not FORCE_RUN:
        raise SystemExit(
            f"\nABORTADO: proyeccion {projected_h:.2f} h > MAX_WALL_HOURS="
            f"{MAX_WALL_HOURS} h. Recorta N_LADDER o pon FORCE_RUN=True.")

sweep = SweepManager(EXP_NAME, BASE_DIR, timestamped=False)
sweep.snapshot_master_script(__main__.__file__)

print("\n==== Setup (convergencia y costo: estimator vs analog, dos capas) ====")
print(f"  polarizacion: {POLARIZATION} (derivada de m,n) | haz: {LASER_RADIUS} um | "
      f"threads: {N_THREADS}")
print(f"  top r={RADIUS_TOP} (l*={L_STAR_TOP:.2f}) | bot r={RADIUS_BOT} (l*={L_STAR_BOT:.2f})")
print(f"  interfaz z={Z_INTERFACE:.2f} um = {THICKNESS_MULT:.1f} l*_top | "
      f"ancla angular l*={LSTAR_ANCHOR:.2f} um")
print(f"  detector unico: theta_max={np.rad2deg(THETA_MAX):.4f} deg (q={Q_MAX}) | "
      f"{N_THETA} bins uniformes (dq={D_Q:.4f}) | N_PHI={N_PHI}")
for mode in RUN_MODES:
    print(f"  escalera {mode:9s}: {[n_tag(n) for n in N_LADDER[mode]]} "
          f"x {N_REPLICAS} replicas de N/{N_REPLICAS}")

sweep.log_readme(
    f"CONVERGENCIA y COSTO COMPUTACIONAL de ESTIMATOR (next-event) vs ANALOG en "
    f"el medio de DOS CAPAS. UNA sola muestra y UNA sola grilla angular; el unico "
    f"grado de libertad es (modo, N, replica). Muestra: top r={RADIUS_TOP} um "
    f"(l*={L_STAR_TOP:.2f}), bot r={RADIUS_BOT} um (l*={L_STAR_BOT:.2f}), interfaz a "
    f"{THICKNESS_MULT:.1f} l*_top = {Z_INTERFACE:.2f} um (los fotones cruzan y "
    f"muestrean ambas capas). Polarizacion {POLARIZATION.upper()} (m={LASER_M}, "
    f"n={LASER_N}; etiqueta derivada de m,n), haz FIJO w={LASER_RADIUS} um (onda "
    f"plana), sin absorcion, INTEGRADO en tiempo (t_max=dt=0). UN SOLO detector "
    f"far-field con grilla UNIFORME desde el pico hasta la cola (sin ventanas ni "
    f"stitching, para no exponer al modo directo a diferencias de scoring entre "
    f"detectores), FIJA para toda la campana: q=k*l**theta in [0,{Q_MAX}] anclado al "
    f"menor l* ({LSTAR_ANCHOR:.2f} um), {N_THETA} bins uniformes (dq={D_Q:.4f}, "
    f"theta_max={np.rad2deg(THETA_MAX):.4f} deg); grilla phi "
    f"UNIFORME de {N_PHI} columnas en AMBOS modos (NO se usa set_phi_slices: en modo "
    f"directo mis-bina; con incidencia circular el patron es azimutalmente simetrico "
    f"y se promedian las columnas en el analisis). La UNICA diferencia entre modos es "
    f"el flag de scoring del sensor. ESCALERA de fotones (total por punto): "
    f"estimator {N_LADDER['estimator']}, analog {N_LADDER['analog']} (tope duro 1e9). "
    f"Cada punto se corre como {N_REPLICAS} trozos INDEPENDIENTES de N_total/"
    f"{N_REPLICAS}: la suma de trozos recupera la estadistica de N_total y su "
    f"dispersion da la banda sin costo extra (unica varianza valida con next-event). "
    f"Los puntos de la escalera son independientes entre si (no anidados). NOTA de "
    f"costo: el estimator fuerza deteccion sobre N_THETA*N_PHI={N_THETA * N_PHI} bins "
    f"por evento, asi que su tiempo por foton es ~lineal en N_THETA mientras que el "
    f"analog es insensible; N_THETA se fija una vez para toda la campana y cualquier "
    f"comparacion de costo es condicional a este valor. Semillas "
    f"SEED_BASE={SEED_BASE} + 1e6*modo + 1e3*idx + rep. n_threads={N_THREADS} FIJO en "
    f"todas las corridas (comparabilidad de tiempos); se registran runtime_s (wall) y "
    f"cpu_time_s (CPU sumado sobre threads) por replica, mas timings.csv en la raiz de "
    f"la campana. Analisis previsto: (a) eta(q) de cada N contra la referencia a N_max "
    f"del mismo modo y contra el otro modo; (b) sigma de los escalares vs N (esperado "
    f"N^-1/2); (c) T vs N (esperado lineal) y FOM = 1/(sigma^2 T)."
)

timings = []
run_counter = 0
t_campaign = time.perf_counter()

for idx in range(max(len(N_LADDER[m]) for m in RUN_MODES)):
    for mode in RUN_MODES:
        ladder = N_LADDER[mode]
        if idx >= len(ladder):
            continue
        n_total = ladder[idx]
        n_chunk = n_total // N_REPLICAS
        if n_chunk < 1:
            print(f"  [skip] {mode} idx {idx}: N_total={n_total} < N_REPLICAS")
            continue
        for rep in range(N_REPLICAS):
            name = f"layers_{mode}_N{n_tag(n_total)}_rep{rep:02d}"
            print(f"\n=== Corrida: {name}  (chunk N={n_chunk:,}) ===")
            sweep.run(run_counter, name,
                      lambda exp, mode=mode, n_chunk=n_chunk, idx=idx, rep=rep:
                          run_point(exp, mode, n_chunk, idx, rep, timings))
            run_counter += 1

# ---------------------------------------------------------------------------
# Log plano de tiempos (conveniencia; la fuente de verdad sigue siendo el HDF5)
# ---------------------------------------------------------------------------
csv_path = os.path.join(BASE_DIR, EXP_NAME, "timings.csv")
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
with open(csv_path, "w") as fh:
    fh.write("mode,ladder_index,replica,n_photons_chunk,n_photons_point_total,"
             "wall_s,cpu_s,photons_per_s,hits,seed\n")
    for row in timings:
        fh.write(",".join(str(v) for v in row) + "\n")

print(f"\n==== Campana terminada: {run_counter} corridas en "
      f"{(time.perf_counter() - t_campaign) / 3600.0:.2f} h ====")
print(f"  tiempos: {csv_path}")