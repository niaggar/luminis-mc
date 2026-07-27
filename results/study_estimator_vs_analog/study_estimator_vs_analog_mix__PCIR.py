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
    derived_quantities, derived_quantities_mixture,
    set_log_level, LogLevel, LaserSource,
    MixtureLayer,
)

set_log_level(LogLevel.info)

# ===========================================================================
# QUE ES ESTE ESTUDIO
# ---------------------------------------------------------------------------
#   GEMELO de study_convergence_layers para el medio de MEZCLA. Misma escalera,
#   mismas replicas, mismo detector unico, mismo protocolo de tiempos: lo unico
#   que cambia es la ARQUITECTURA del medio. Las dos MISMAS especies que en las
#   capas (r=0.035 y r=0.075 um) pero ahora CO-LOCALIZADAS en una MixtureLayer
#   semi-infinita a composicion unica x=0.5 (pesada por scattering).
#
#   Por que importa el par capas/mezcla: el scoring del estimador ejercita en la
#   mezcla rutas que no aparecen en un medio homogeneo NI en las capas -- sobre
#   todo la normalizacion I_norm cacheada por (medium, k), que en una mezcla se
#   invalida vertice a vertice porque active_medium cambia en cada evento. Si
#   ese camino tuviera un sesgo o un costo anomalo, aparece aqui y no alla.
#
#   Observables (identicos al gemelo de capas):
#     (a) convergencia: eta(q) a cada N vs la referencia a N_max -> mismo limite
#         en ambos modos;
#     (b) ruido: sigma de los escalares vs N -> N^-1/2;
#     (c) costo: wall/CPU vs N -> lineal en N; FOM = 1/(sigma^2 T).
#   El unico grado de libertad entre corridas es (modo, N, replica).
# ===========================================================================

# ===========================================================================
# Salida
# ===========================================================================
BASE_DIR = "/home/niaggar/Developer/luminis-mc/temporal_results"

# ===========================================================================
# Parametros fisicos
#   Las especies son las MISMAS que en la tanda de capas, con la convencion de
#   la §5.2: especie 1 = la de fraccion x. Se toma 1 = bot (0.075) y 2 = top
#   (0.035) para heredar el anclaje de mu_s_total del barrido de composicion.
# ===========================================================================
RADIUS_1 = 0.075                 # um  (= RADIUS_BOT del gemelo de capas)
RADIUS_2 = 0.035                 # um  (= RADIUS_TOP del gemelo de capas)
FRACTION_X = 0.5                 # composicion UNICA (pesada por scattering)

VOLUME_FRACTION = 0.10           # solo fija la ESCALA de mu_s_total (ancla en x=1)
N_PARTICLE = 1.59
N_MEDIUM = 1.33
WAVELENGTH = 0.514               # um
MU_A_PERCENT = 0.0               # sin absorcion (gate de reciprocidad limpio)

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
EXP_NAME = f"study_convergence_mixture__{POLARIZATION.upper()}__beam{LASER_RADIUS}"

# Funcion de fase
PHASEF_THETA_MIN = 0.0
PHASEF_THETA_MAX = np.pi
PHASEF_NDIV = 10_000

# ---------------------------------------------------------------------------
# Grilla angular: UN SOLO detector, grilla UNIFORME en theta desde el pico
# hasta la cola, q = k*l*_mix*theta in [0, Q_MAX]. Un unico sensor elimina el
# solape/stitching y la posibilidad de que dos detectores puntuen distinto en
# modo directo. FIJA para TODA la campana -> los perfiles de distintos N y
# distintos modos comparten bins y se restan bin a bin.
#
# El ancla es l*_mix a x=0.5 (no min(l*_1, l*_2)): la mezcla tiene su propio
# transporte y el gemelo de capas tiene el suyo, asi que las dos campanas NO
# comparten grilla en radianes -- solo en unidades reducidas q.
#
# COSTO: el estimator fuerza deteccion sobre N_THETA*n_columnas_phi bins EN
# CADA evento -> su tiempo por foton es ~lineal en N_THETA; el analog es
# insensible. N_THETA se fija una vez y no se toca (y debe ser el MISMO que en
# el gemelo de capas para que las dos FOM sean comparables).
# ---------------------------------------------------------------------------
Q_MAX = 40.0
N_THETA = 2000                   # dq = 0.01 -> ~750 bins dentro del cono (q<7.5)
N_PHI = 1                        # ancho azimutal NOMINAL (ver PHI_SLICES)
PHI_MAX = 2 * np.pi
k = 2 * np.pi * N_MEDIUM / WAVELENGTH

T_MAX = 0.0                      # integrado en tiempo
DT = 0.0

# ---------------------------------------------------------------------------
# Escalera de fotones (TOTAL por punto; cada punto se parte en N_REPLICAS).
# IDENTICA a la del gemelo de capas: si cambia una, cambia la otra, o las FOM
# dejan de ser comparables. Tope duro del analog: 1e9.
# ---------------------------------------------------------------------------
N_LADDER = {
    "estimator": [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 500_000],
    "analog":    [1_000_000, 3_000_000, 10_000_000, 30_000_000,
                  100_000_000, 500_000_000, 1_000_000_000],
}
RUN_MODES = ("estimator", "analog")

N_REPLICAS = 5                   # t_{0.975,4} = 2.78 en el analisis
N_THREADS = 46
SEED_BASE = 20260728             # distinto del gemelo de capas (contabilidad)

# --- Pilotos de costo: mide fotones/s por modo y proyecta la campana ---------
RUN_PILOT = True
PILOT_N = {"estimator": 2_000, "analog": 2_000_000}
MAX_WALL_HOURS = 16.0            # aborta si la proyeccion excede este techo
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


def set_albedo(medium, mu_a_percent):
    """Albedo bien definido para que la agregacion mu_a^(i) no caiga en 0/0.
    SIN mfp: en la mezcla el camino libre lo fijan las densidades (mfp_total de
    la MixtureLayer), no el medio individual."""
    medium.set_absorption_coefficient(mu_a_percent)
    medium.set_scattering_coefficient(1.0 - mu_a_percent)


def make_sensors(estimator):
    """UN solo detector far-field (pico -> cola, grilla uniforme) + estadistica,
    integrado en tiempo. Grilla IDENTICA en ambos modos: lo unico que cambia es
    el flag de scoring."""
    d_theta = THETA_MAX / N_THETA
    d_phi = PHI_MAX / N_PHI

    sens = SensorsGroup()
    det = sens.add_detector(
        FarFieldCBSSensor(THETA_MAX, PHI_MAX, T_MAX, d_theta, d_phi, DT, estimator))
    det.set_theta_limit(0, THETA_MAX)

    stats = sens.add_detector(StatisticsSensor(z=0, absorb=True))
    stats.set_theta_limit(0, THETA_MAX)
    return sens, det, stats, d_theta, d_phi


def build_sample():
    """MixtureLayer semi-infinita: las dos especies co-localizadas, elegidas por
    evento segun mu_s^(i). Densidades FIJAS (composicion unica x)."""
    sample = Sample(N_MEDIUM)
    sample.add_mixture_layer(SPECIES, DENSITIES, 0.0, float("inf"))
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
# Setup de la mezcla (una sola vez; los medios viven a nivel de modulo por la
# keep-alive gotcha de los raw pointers)
# ===========================================================================
SPECIES_1 = build_species(RADIUS_1)
SPECIES_2 = build_species(RADIUS_2)
set_albedo(SPECIES_1, MU_A_PERCENT)
set_albedo(SPECIES_2, MU_A_PERCENT)
SPECIES = [SPECIES_1, SPECIES_2]

DQ_1 = derived_quantities(SPECIES_1, VOLUME_FRACTION)
DQ_2 = derived_quantities(SPECIES_2, VOLUME_FRACTION)

# secciones eficaces (independientes de la densidad)
SIGMA_1 = DQ_1['scattering_efficiency'] * np.pi * RADIUS_1 ** 2
SIGMA_2 = DQ_2['scattering_efficiency'] * np.pi * RADIUS_2 ** 2

# invariante mu_s_total constante, anclado al medio PURO de especie 1 a f=0.10
MU_S_TOTAL = number_density(RADIUS_1, VOLUME_FRACTION) * SIGMA_1
ND_1 = FRACTION_X * MU_S_TOTAL / SIGMA_1
ND_2 = (1.0 - FRACTION_X) * MU_S_TOTAL / SIGMA_2
DENSITIES = [ND_1, ND_2]
MU_S_SHARE = [ND_1 * SIGMA_1 / MU_S_TOTAL, ND_2 * SIGMA_2 / MU_S_TOTAL]

DQ_MIX = derived_quantities_mixture(SPECIES, DENSITIES)
LSTAR_MIX = DQ_MIX['transport_mean_free_path']
THETA_MAX = Q_MAX / (k * LSTAR_MIX)
D_Q = Q_MAX / N_THETA                                      # resolucion en q


# ===========================================================================
# Corrida unitaria
# ===========================================================================
def run_point(exp, mode, n_photons, idx, rep, timings):
    """Una replica de un punto de la escalera. exp=None -> piloto (no persiste)."""
    estimator = (mode == "estimator")
    seed = make_seed(mode, idx, rep)

    sample = build_sample()
    layer = sample.layers[0]
    mfp_layer = float(layer.mfp_total) if isinstance(layer, MixtureLayer) else 0.0

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
        **DQ_MIX,
        "study": "convergence_timing",
        "medium_kind": "mixture",
        "layer_kind": "mixture",
        "n_species": len(SPECIES),
        "mode": mode,
        "ladder_index": idx,
        "replica": rep,
        "n_photons": n_photons,              # de ESTA replica
        "n_photons_point_total": n_total,    # del punto completo (suma de replicas)
        "n_replicas": N_REPLICAS,
        "seed": seed,
        "fraction_x": FRACTION_X,
        "radius": [RADIUS_1, RADIUS_2],
        "volume_fraction": VOLUME_FRACTION,
        "number_densities": DENSITIES,
        "mu_s_total": MU_S_TOTAL,
        "mu_s_share": MU_S_SHARE,
        "mfp_total_layer": mfp_layer,
        "mean_free_paths": [DQ_1['mean_free_path'], DQ_2['mean_free_path']],
        "transport_mean_free_paths": [DQ_1['transport_mean_free_path'],
                                      DQ_2['transport_mean_free_path']],
        "anisotropy_g_species": [DQ_1['anisotropy_g'], DQ_2['anisotropy_g']],
        "lstar_mix": LSTAR_MIX,
        "lstar_angle_anchor": LSTAR_MIX,
        "laser_radius_um": LASER_RADIUS,
        "polarization": POLARIZATION,
        "theta_max": THETA_MAX,
        "d_theta": d_theta, "d_phi": d_phi,
        "n_theta": N_THETA, "n_phi_nominal": N_PHI,
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

print("\n==== Setup (convergencia y costo: estimator vs analog, mezcla) ====")
print(f"  polarizacion: {POLARIZATION} (derivada de m,n) | haz: {LASER_RADIUS} um | "
      f"threads: {N_THREADS}")
print(f"  esp.1 r={RADIUS_1} (l*={DQ_1['transport_mean_free_path']:.2f}, "
      f"g={DQ_1['anisotropy_g']:.4f}) | esp.2 r={RADIUS_2} "
      f"(l*={DQ_2['transport_mean_free_path']:.2f}, g={DQ_2['anisotropy_g']:.4f})")
print(f"  x={FRACTION_X} | mu_s_total={MU_S_TOTAL:.4e} 1/um (l_s={1.0/MU_S_TOTAL:.3f} um) | "
      f"share=[{MU_S_SHARE[0]:.3f}, {MU_S_SHARE[1]:.3f}] | l*_mix={LSTAR_MIX:.3f} um "
      f"| w/l*={LASER_RADIUS/LSTAR_MIX:.1f}")
print(f"  detector unico: theta_max={np.rad2deg(THETA_MAX):.4f} deg (q={Q_MAX}) | "
      f"{N_THETA} bins uniformes (dq={D_Q:.4f}) ")
for mode in RUN_MODES:
    print(f"  escalera {mode:9s}: {[n_tag(n) for n in N_LADDER[mode]]} "
          f"x {N_REPLICAS} replicas de N/{N_REPLICAS}")

sweep.log_readme(
    f"CONVERGENCIA y COSTO COMPUTACIONAL de ESTIMATOR (next-event) vs ANALOG en el "
    f"medio de MEZCLA. GEMELO de study_convergence_layers: misma escalera, mismas "
    f"replicas, mismo detector unico y mismo protocolo de tiempos; lo unico que "
    f"cambia es la arquitectura del medio. MixtureLayer semi-infinita con las MISMAS "
    f"dos especies que las capas (r1={RADIUS_1} um, l*={DQ_1['transport_mean_free_path']:.3f}; "
    f"r2={RADIUS_2} um, l*={DQ_2['transport_mean_free_path']:.3f}) co-localizadas a "
    f"composicion UNICA x={FRACTION_X} (pesada por scattering), invariante "
    f"mu_s_total constante {MU_S_TOTAL:.4e} 1/um anclado al medio puro de especie 1 a "
    f"f={VOLUME_FRACTION} (l_s={1.0/MU_S_TOTAL:.3f} um), share=[{MU_S_SHARE[0]:.3f}, "
    f"{MU_S_SHARE[1]:.3f}], l*_mix={LSTAR_MIX:.3f} um. Motivo del gemelo: el scoring "
    f"del estimador en mezcla ejercita rutas ausentes del caso homogeneo y de las "
    f"capas -- sobre todo la normalizacion I_norm cacheada por (medium,k), que se "
    f"invalida vertice a vertice porque active_medium cambia en cada evento. "
    f"Polarizacion {POLARIZATION.upper()} (m={LASER_M}, n={LASER_N}; etiqueta derivada "
    f"de m,n), haz FIJO w={LASER_RADIUS} um (onda plana, w/l*={LASER_RADIUS/LSTAR_MIX:.1f}), "
    f"sin absorcion, INTEGRADO en tiempo (t_max=dt=0). UN SOLO detector far-field con "
    f"grilla UNIFORME desde el pico hasta la cola (sin ventanas ni stitching), FIJA "
    f"para toda la campana: q=k*l*_mix*theta in [0,{Q_MAX}] anclado a l*_mix "
    f"({LSTAR_MIX:.3f} um), {N_THETA} bins uniformes (dq={D_Q:.4f}, "
    f"theta_max={np.rad2deg(THETA_MAX):.4f} deg);"
    f"OJO al comparar con capas: las dos campanas comparten la grilla en unidades "
    f"REDUCIDAS q, no en radianes (anclas de l* distintas). La UNICA diferencia entre "
    f"modos es el flag de scoring del sensor; los cortes phi explicitos son "
    f"estimator-only, en modo directo el binning phi->columna asume grilla uniforme "
    f"(§12 code_description), asi que la comparacion entre modos se hace sobre eta "
    f"promediado en phi (con incidencia circular el patron es azimutalmente simetrico) "
    f"y NO columna a columna en valor absoluto. ESCALERA de fotones (total por punto): "
    f"estimator {N_LADDER['estimator']}, analog {N_LADDER['analog']} (tope duro 1e9); "
    f"IDENTICA a la del gemelo de capas -- si cambia una, cambia la otra o las FOM "
    f"dejan de ser comparables. Cada punto se corre como {N_REPLICAS} trozos "
    f"INDEPENDIENTES de N_total/{N_REPLICAS}: la suma de trozos recupera la "
    f"estadistica de N_total y su dispersion da la banda sin costo extra (unica "
    f"varianza valida con next-event). Los puntos de la escalera son independientes "
    f"entre si (no anidados). NOTA de costo: el estimator fuerza deteccion sobre "
    f"N_THETA*n_columnas_phi bins por evento -> tiempo por foton ~lineal en N_THETA, "
    f"el analog es insensible; N_THETA={N_THETA} fijo en ambas campanas. Semillas "
    f"SEED_BASE={SEED_BASE} + 1e6*modo + 1e3*idx + rep (distinto del gemelo de capas). "
    f"n_threads={N_THREADS} FIJO en todas las corridas (comparabilidad de tiempos); se "
    f"registran runtime_s (wall) y cpu_time_s (CPU sumado sobre threads) por replica, "
    f"mas timings.csv en la raiz de la campana. Analisis previsto: (a) eta(q) de cada N "
    f"contra la referencia a N_max del mismo modo y contra el otro modo; (b) sigma de "
    f"los escalares vs N (esperado N^-1/2); (c) T vs N (esperado lineal) y "
    f"FOM = 1/(sigma^2 T), comparada con la del gemelo de capas."
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
            name = f"mixture_x{FRACTION_X:.2f}_{mode}_N{n_tag(n_total)}_rep{rep:02d}"
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