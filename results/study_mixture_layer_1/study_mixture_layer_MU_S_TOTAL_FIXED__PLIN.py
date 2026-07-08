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

from utils.time import build_time_grid

set_log_level(LogLevel.info)

# ===========================================================================
# Salida
# ===========================================================================
EXP_NAME = "study_mixture_layer__PLIN"
BASE_DIR = "/home/niaggar/Developer/luminis-mc/temporal_results"

sweep = SweepManager(EXP_NAME, BASE_DIR, timestamped=False)
sweep.snapshot_master_script(__main__.__file__)
sweep.log_readme(
    "CBS mixture composition sweep -- dos especies RGD DISTINTAS "
    "(r1=0.175 um, r2=0.035 um), polarizacion LINEAL, estimador. "
    "Invariante mu_s_total constante: l_s fijo; ancho de cono y factor de realce "
    "como funciones limpias de la composicion x. Rejilla temporal UNICA compartida, "
    "anclada a la composicion mas lenta (x=1, especie 1 pura)."
)

# ===========================================================================
# Parametros fisicos
# ===========================================================================
RADIUS_1 = 0.075                 # um  (especie de fraccion x)
RADIUS_2 = 0.035                 # um  (especie de fraccion 1-x)

VOLUME_FRACTION = 0.10           # solo fija la ESCALA de mu_s_total (ancla en x=1)
N_PARTICLE = 1.59
N_MEDIUM = 1.33
WAVELENGTH = 0.514               # um
MU_A_PERCENT = 0.0               # sin absorcion (gate de reciprocidad limpio)

# ---------------------------------------------------------------------------
# Polarizacion: LINEAL.
# ---------------------------------------------------------------------------
LASER_M = 1
LASER_N = 0
LASER_RADIUS_MFP = 4.0           # radio del haz en unidades de l* (FIJO en el barrido)
LASER_TYPE = LaserSource.Gaussian

# Funcion de fase
PHASEF_THETA_MIN = 0.0
PHASEF_THETA_MAX = np.pi
PHASEF_NDIV = 10_000

# ---------------------------------------------------------------------------
# Grilla angular
#   NOTA: N_THETA=100 (0.01 deg/celda) resuelve de sobra un cono de ~pocos mrad.
#   Con 1000 se reparten los pocos hits del cono 10x mas fino -> ruido, y peor
#   ahora que los bins temporales multiplican el numero de celdas (theta x t).
# ---------------------------------------------------------------------------
N_THETA = 1000
N_PHI = 36
PHI_MAX = 2 * np.pi
THETA_MAX = np.deg2rad(1)

# ---------------------------------------------------------------------------
# Grilla temporal (COMPARTIDA por todas las composiciones)
#   Se ancla a la composicion mas lenta (l* mayor). Con mu_s_total fijo, l_s es
#   constante y g_eff = x*g1 + (1-x)*g2 es lineal en x; como g1 > g2, el l* maximo
#   ocurre en x=1 (especie 1 pura). Anclar ahi garantiza que la ventana CONTIENE
#   la dinamica de toda composicion (nada se corta). Las composiciones rapidas
#   sobre-extienden la cola: costo aceptable de tener un eje temporal comun.
# ---------------------------------------------------------------------------
TIME_NBINS = 30                  # ~1.2 tau*/bin sobre la especie lenta; >30 fs
TIME_TMAX_TAUSTAR = 30           # 1-30 tau* es donde ocurre el estrechamiento

# ---------------------------------------------------------------------------
# Muestreo
# ---------------------------------------------------------------------------
N_PHOTONS = 100_000
N_THREADS = 44

# ===========================================================================
# Barrido
# ===========================================================================
fraction_list = np.linspace(0.0, 1.0, 11)   # fraccion (pesada por scattering) de esp. 1


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
    La escala absoluta de mu_s NO afecta el transporte (la total la fija n_i sigma_i)."""
    medium.set_absorption_coefficient(mu_a_percent)
    medium.set_scattering_coefficient(1.0 - mu_a_percent)


# ===========================================================================
# Setup COMPARTIDO (una sola vez)
#   Las especies dependen solo del radio, no de la composicion -> se construyen
#   una vez. Vivir en scope de modulo las mantiene vivas para toda la corrida
#   (el binding guarda punteros crudos): keep-alive resuelto de raiz.
# ===========================================================================
SPECIES_1 = build_species(RADIUS_1)
SPECIES_2 = build_species(RADIUS_2)
set_albedo(SPECIES_1, MU_A_PERCENT)
set_albedo(SPECIES_2, MU_A_PERCENT)

DQ_1 = derived_quantities(SPECIES_1, VOLUME_FRACTION)
DQ_2 = derived_quantities(SPECIES_2, VOLUME_FRACTION)

# secciones eficaces (independientes de la densidad)
SIGMA_1 = DQ_1['scattering_efficiency'] * np.pi * RADIUS_1 ** 2
SIGMA_2 = DQ_2['scattering_efficiency'] * np.pi * RADIUS_2 ** 2

# ancla mu_s_total al medio PURO de especie 1 a f=0.10 (asi x=1 == homogeneo esp.1)
MU_S_TOTAL = number_density(RADIUS_1, VOLUME_FRACTION) * SIGMA_1

# rejilla temporal unica + radio de haz fijo, ambos anclados a la composicion lenta
GRID = build_time_grid(
    DQ_1['transport_mean_free_path'], N_MEDIUM,
    n_bins=TIME_NBINS, t_max_taustar=TIME_TMAX_TAUSTAR, binning="geometric",
)
LASER_RADIUS_UM = LASER_RADIUS_MFP * DQ_1['transport_mean_free_path']

print("==== Setup compartido ====")
print(f"l* esp.1 (ancla): {DQ_1['transport_mean_free_path']:.4f} um   g1={DQ_1['anisotropy_g']:.4f}")
print(f"l* esp.2         : {DQ_2['transport_mean_free_path']:.4f} um   g2={DQ_2['anisotropy_g']:.4f}")
print(f"mu_s_total (fijo): {MU_S_TOTAL:.4e} 1/um")
print(f"radio de haz     : {LASER_RADIUS_UM:.2f} um  (= {LASER_RADIUS_MFP} l*)")
print(f"GRID: {GRID}")


# ===========================================================================
# Corrida
# ===========================================================================
def run_mixture_simulation(exp: Experiment, fraction: float):
    """MixtureLayer de dos especies a composicion x; corre CBS y persiste."""
    x = float(fraction)

    # densidades: mu_s^(1)=x*mu_s_total, mu_s^(2)=(1-x)*mu_s_total -> suma fija
    nd_1 = x * MU_S_TOTAL / SIGMA_1
    nd_2 = (1.0 - x) * MU_S_TOTAL / SIGMA_2
    densities = [nd_1, nd_2]

    mu1, mu2 = nd_1 * SIGMA_1, nd_2 * SIGMA_2
    print(f"\nx={x:.2f}  share mu_s: sp1={mu1 / MU_S_TOTAL:.3f}  sp2={mu2 / MU_S_TOTAL:.3f}"
          f"  mu_s_total={mu1 + mu2:.4e}")

    species = [SPECIES_1, SPECIES_2]

    sample = Sample(N_MEDIUM)
    sample.add_mixture_layer(species, densities, 0.0, float("inf"))

    layer = sample.layers[0]
    mfp_layer = float(layer.mfp_total) if isinstance(layer, MixtureLayer) else 0.0
    print(f"mfp_total (l_s capa): {mfp_layer:.4f} um   theta_c1={DQ_1['theta_coherent']*1e3:.3f} mrad"
          f"   theta_c2={DQ_2['theta_coherent']*1e3:.3f} mrad")

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

    # --- params: derivados de mezcla + grilla ---
    dq_mix = derived_quantities_mixture(species, densities)
    extra = {
        **dq_mix,
        "fraction_x": x,
        "radius": [RADIUS_1, RADIUS_2],
        "volume_fraction": VOLUME_FRACTION,
        "number_densities": densities,
        "mu_s_share": [mu1 / MU_S_TOTAL, mu2 / MU_S_TOTAL],
        "mu_s_total": MU_S_TOTAL,
        "mean_free_paths": [DQ_1['mean_free_path'], DQ_2['mean_free_path']],
        "transport_mean_free_paths": [DQ_1['transport_mean_free_path'], DQ_2['transport_mean_free_path']],
        "theta_coherents": [DQ_1['theta_coherent'], DQ_2['theta_coherent']],
        "mfp_total_layer": mfp_layer,
        "laser_radius_um": LASER_RADIUS_UM,
        "theta_max": THETA_MAX,
        "d_theta": d_theta,
        "d_phi": d_phi,
        "n_theta": N_THETA,
        "n_phi": N_PHI,
        "t_max": t_max,
        "d_time": dt,
        "time_grid": GRID,
        "polarization": "circular",
        "layer_kind": "mixture",
        "n_species": len(species),
    }
    exp.save_params(config, extra=extra)

    # --- run ---
    t0 = time.time()
    run_simulation_parallel(config)
    print("runtime_s:", time.time() - t0, "| hits:", det.hits)

    # --- guardar RAW + procesados ---
    exp.save_sensors({"farfield_cbs": det, "statistics": stats})
    exp.save_processed("farfield_cbs", postprocess_farfield_cbs(det, N_PHOTONS), sensor=det)

    # keep-alive del sample (las especies viven en scope de modulo)
    _keep_alive = (sample,)
    del _keep_alive


# ===========================================================================
# Loop  (default-binding x=fraction: evita el bug de late-binding del closure)
# ===========================================================================
for index, fraction in enumerate(fraction_list):
    run_name = f"fraction_{fraction:.2f}"
    print(f"\n\n=== Corrida: {run_name} ===")
    sweep.run(index, run_name, lambda exp, x=fraction: run_mixture_simulation(exp, x))