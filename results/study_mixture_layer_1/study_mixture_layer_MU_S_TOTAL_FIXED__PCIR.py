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
EXP_NAME = "study_mixture_layer__PCIR__beam2500"
BASE_DIR = "/home/niaggar/Developer/luminis-mc/temporal_results"

sweep = SweepManager(EXP_NAME, BASE_DIR, timestamped=False)
sweep.snapshot_master_script(__main__.__file__)

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
# Polarizacion: CIRCULAR
# ---------------------------------------------------------------------------
LASER_M = 1 / np.sqrt(2)
LASER_N = 1j / np.sqrt(2)
LASER_TYPE = LaserSource.Gaussian
POLARIZATION = "linear" if (LASER_M, LASER_N) == (1, 0) else "circular"

# ---------------------------------------------------------------------------
# Haz FIJO en um (aparato fijo = fidelidad experimental). w/l* >> 1 en toda
# composicion (l* de 18-23 um -> w/l* de ~108 a ~136): regimen de onda plana.
# ---------------------------------------------------------------------------
LASER_RADIUS = 2500              # um

# Funcion de fase
PHASEF_THETA_MIN = 0.0
PHASEF_THETA_MAX = np.pi
PHASEF_NDIV = 10_000

# ---------------------------------------------------------------------------
# Grilla angular COMPARTIDA (identica en las 11 composiciones).
#   F12 y el test x(1-x) restan perfiles de corridas distintas -> exigen la
#   MISMA grilla en todas, sin interpolacion. Se ancla al cono MAS ANCHO del
#   barrido = x=0 (l* minimo, especie 2 pura), asi la ventana fina contiene
#   q in [0, Q_FINE] para la composicion mas ancha y sobra para el resto.
#   Doble ventana en unidades reducidas q = k*l*_ancla*theta.
# ---------------------------------------------------------------------------
N_THETA_1 = 500                  # ventana fina (cono)
N_THETA_2 = 200                  # ventana cola (fondo/baseline)
Q_FINE, Q_TAIL = 7.5, 40.0
N_PHI = 36
PHI_MAX = 2 * np.pi
k = 2 * np.pi * N_MEDIUM / WAVELENGTH

# ---------------------------------------------------------------------------
# Grilla temporal COMPARTIDA (identica en las 11 composiciones).
#   OJO: ancla OPUESTA a la angular. La ventana temporal debe CONTENER la
#   dinamica mas LENTA = x=1 (l* maximo -> tau* mayor); anclar a x=0 cortaria
#   la cola de x=1 y perderia el estrechamiento tardio Delta_theta ~ t^-1/2,
#   que es justo el observable time-resolved. Las composiciones rapidas
#   sobre-extienden la cola: costo aceptable de un eje temporal comun.
#   Bin 0 del sensor = senal INTEGRADA en tiempo (analisis estacionario §5.3);
#   bins 1..N = ventanas temporales (material extra si da el tiempo).
# ---------------------------------------------------------------------------
TIME_NBINS = 30
TIME_TMAX_TAUSTAR = 30

# ---------------------------------------------------------------------------
# Muestreo
# ---------------------------------------------------------------------------
N_PHOTONS = 100_000
N_THREADS = 46
N_REPLICAS = 5
SEED_BASE_MIX = 20260711         # distinto del barrido homogeneo (contabilidad)

RUN_MATCHED_REF = False           # corrida homogenea r2 emparejada -> gate x=0

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


def set_albedo(medium, mu_a_percent, mfp=None):
    """Albedo bien definido para que la agregacion mu_a^(i) no caiga en 0/0.
    La escala absoluta de mu_s NO afecta el transporte (la total la fija n_i sigma_i).
    mfp opcional: solo lo necesita la corrida homogenea de referencia."""
    medium.set_absorption_coefficient(mu_a_percent)
    medium.set_scattering_coefficient(1.0 - mu_a_percent)
    if mfp is not None:
        medium.set_mean_free_path(mfp)


def make_sensors(theta_1, theta_2):
    """Doble detector far-field (fino + cola) + estadistica. Grilla COMPARTIDA
    (angular anclada a x=0, temporal a x=1)."""
    d_theta_1 = theta_1 / N_THETA_1
    d_theta_2 = theta_2 / N_THETA_2
    d_phi = PHI_MAX / N_PHI
    t_max = GRID["t_max_sim"]        # ancla temporal a x=1 (dinamica mas lenta)
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

# --- Ancla ANGULAR: cono mas ancho = x=0 (especie 2 pura, l* minimo) ---
#   l*(x=0) = 1 / (mu_s_total * (1 - g2))
LSTAR_ANCHOR = 1.0 / (MU_S_TOTAL * (1.0 - DQ_2['anisotropy_g']))
THETA_1 = Q_FINE / (k * LSTAR_ANCHOR)
THETA_2 = Q_TAIL / (k * LSTAR_ANCHOR)

# --- Ancla TEMPORAL: dinamica mas lenta = x=1 (especie 1 pura, l* maximo) ---
#   Anclar aqui garantiza que la ventana temporal contiene la cola de TODA
#   composicion (la mas lenta es la cota superior). Ancla OPUESTA a la angular.
GRID = build_time_grid(
    DQ_1['transport_mean_free_path'], N_MEDIUM,
    n_bins=TIME_NBINS, t_max_taustar=TIME_TMAX_TAUSTAR, binning="geometric",
)

# sanity: el ancla debe reproducir l*(x=0) que reporta derived_quantities_mixture
_nd2_pure = MU_S_TOTAL / SIGMA_2
_lstar_x0_check = derived_quantities_mixture([SPECIES_2], [_nd2_pure])['transport_mean_free_path']

print("==== Setup compartido ====")
print(f"l* esp.1 (x=1): {DQ_1['transport_mean_free_path']:.4f} um   g1={DQ_1['anisotropy_g']:.4f}")
print(f"l* esp.2 (x=0): {DQ_2['transport_mean_free_path']:.4f} um   g2={DQ_2['anisotropy_g']:.4f}")
print(f"mu_s_total (fijo): {MU_S_TOTAL:.4e} 1/um   l_s = {1.0/MU_S_TOTAL:.4f} um")
print(f"LSTAR_ANCHOR (x=0): {LSTAR_ANCHOR:.4f} um   check via dq_mixture: {_lstar_x0_check:.4f} um")
print(f"ventana: theta_1={np.rad2deg(THETA_1):.3f} deg   theta_2={np.rad2deg(THETA_2):.3f} deg")
print(f"haz: {LASER_RADIUS} um   w/l*(x=0)={LASER_RADIUS/LSTAR_ANCHOR:.1f}  w/l*(x=1)={LASER_RADIUS/DQ_1['transport_mean_free_path']:.1f}")
print(f"tiempo (ancla x=1): dt={GRID['dt_fs']:.2f} fs  t_max={GRID['t_max_fs']:.1f} fs  "
      f"tau*={GRID['tau_star_fs']:.2f} fs  n_bins={TIME_NBINS}")
if not np.isclose(LSTAR_ANCHOR, _lstar_x0_check, rtol=1e-6):
    print("[WARN] LSTAR_ANCHOR no coincide con dq_mixture(x=0): revisar despeje del ancla")


# ===========================================================================
# Corrida de mezcla
# ===========================================================================
def run_mixture_simulation(exp: Experiment, fraction: float, frac_index: int, rep: int):
    """MixtureLayer de dos especies a composicion x; corre CBS y persiste."""
    x = float(fraction)

    # densidades: mu_s^(1)=x*mu_s_total, mu_s^(2)=(1-x)*mu_s_total -> suma fija
    nd_1 = x * MU_S_TOTAL / SIGMA_1
    nd_2 = (1.0 - x) * MU_S_TOTAL / SIGMA_2
    densities = [nd_1, nd_2]

    mu1, mu2 = nd_1 * SIGMA_1, nd_2 * SIGMA_2
    print(f"\nx={x:.2f} rep={rep}  share mu_s: sp1={mu1 / MU_S_TOTAL:.3f}  sp2={mu2 / MU_S_TOTAL:.3f}")

    species = [SPECIES_1, SPECIES_2]
    sample = Sample(N_MEDIUM)
    sample.add_mixture_layer(species, densities, 0.0, float("inf"))

    layer = sample.layers[0]
    mfp_layer = float(layer.mfp_total) if isinstance(layer, MixtureLayer) else 0.0

    laser = Laser(LASER_M, LASER_N, WAVELENGTH, LASER_RADIUS, LASER_TYPE)
    sens, det_1, det_2, stats, d_theta_1, d_theta_2, d_phi = make_sensors(THETA_1, THETA_2)
    config = base_config(sample, laser, sens, SEED_BASE_MIX + frac_index * 1000 + rep)

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
        "laser_radius_um": LASER_RADIUS,
        "lstar_anchor": LSTAR_ANCHOR,
        "angular_anchor": "x=0",
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
        "time_anchor": "x=1",
        "polarization": POLARIZATION,
        "seed": config.seed,
        "replica": rep,
        "layer_kind": "mixture",
        "n_species": len(species),
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
# Corrida de referencia EMPAREJADA (gate x=0)
#   Medio homogeneo de especie 2 a la MISMA densidad emparejada (l_s = l_s mezcla),
#   mismo haz y misma ventana. Debe coincidir con x=0: mezcla degenerada ==
#   homogeneo al mismo mu_s. Cierra el gate del extremo x=0.
# ===========================================================================
def run_matched_reference(exp: Experiment, frac_index: int, rep: int):
    print(f"\n=== matched ref (homogeneo r2={RADIUS_2}, l_s emparejado) rep={rep} ===")
    especie = build_species(RADIUS_2)
    set_albedo(especie, MU_A_PERCENT, mfp=1.0 / MU_S_TOTAL)   # fuerza l_s de la mezcla
    dq = derived_quantities(especie, VOLUME_FRACTION)

    sample = Sample(N_MEDIUM)
    sample.add_layer(especie, 0.0, float("inf"))

    laser = Laser(LASER_M, LASER_N, WAVELENGTH, LASER_RADIUS, LASER_TYPE)
    sens, det_1, det_2, stats, d_theta_1, d_theta_2, d_phi = make_sensors(THETA_1, THETA_2)
    config = base_config(sample, laser, sens, SEED_BASE_MIX + frac_index * 1000 + rep)

    extra = {
        **dq,
        "radius": RADIUS_2,
        "volume_fraction": VOLUME_FRACTION,
        "mu_s_total": MU_S_TOTAL,
        "laser_radius_um": LASER_RADIUS,
        "lstar_anchor": LSTAR_ANCHOR,
        "angular_anchor": "x=0",
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
        "time_anchor": "x=1",
        "polarization": POLARIZATION,
        "seed": config.seed,
        "replica": rep,
        "layer_kind": "homogeneous_matched_ref",
    }
    exp.save_params(config, extra=extra)

    t0 = time.time()
    run_simulation_parallel(config)
    print("runtime_s:", time.time() - t0, "| hits:", det_1.hits + det_2.hits)

    exp.save_sensors({"farfield_cbs_1": det_1, "farfield_cbs_2": det_2, "statistics": stats})
    exp.save_processed("farfield_cbs_1", postprocess_farfield_cbs(det_1, N_PHOTONS), sensor=det_1)
    exp.save_processed("farfield_cbs_2", postprocess_farfield_cbs(det_2, N_PHOTONS), sensor=det_2)

    _keep_alive = (sample, especie)
    del _keep_alive


# ===========================================================================
# README (con f-string: no se desactualiza en lo que importa)
# ===========================================================================
sweep.log_readme(
    f"CBS mixture composition sweep -- dos especies RGD DISTINTAS "
    f"(r1={RADIUS_1} um, r2={RADIUS_2} um), polarizacion {POLARIZATION.upper()} "
    f"(m={LASER_M}, n={LASER_N}), estimador next-event. Invariante mu_s_total "
    f"constante ({MU_S_TOTAL:.4e} 1/um, l_s={1.0/MU_S_TOTAL:.3f} um): l_s fijo, "
    f"ancho de cono y realce como funciones limpias de x. Haz gaussiano FIJO "
    f"w={LASER_RADIUS} um (onda plana, w/l* ~108-136). Doble ventana angular "
    f"COMPARTIDA en unidades reducidas anclada a x=0 (cono mas ancho, "
    f"l*_ancla={LSTAR_ANCHOR:.3f} um): fina q in [0,{Q_FINE}] ({N_THETA_1} bins), "
    f"cola q in [{Q_FINE},{Q_TAIL}] ({N_THETA_2} bins), solape [0.9,1.0]*theta_1. "
    f"Grilla TEMPORAL compartida anclada a x=1 (dinamica mas lenta, ancla OPUESTA "
    f"a la angular): {TIME_NBINS} bins geometricos hasta {TIME_TMAX_TAUSTAR} tau* "
    f"(bin 0 = integrado = analisis estacionario; 1..N = time-resolved). "
    f"{N_REPLICAS} replicas/composicion, "
    f"semillas SEED_BASE_MIX={SEED_BASE_MIX} + 1000*frac_index + rep, en params. "
    f"Incluye corrida de referencia homogenea r2 a l_s emparejado (gate x=0): "
    f"{RUN_MATCHED_REF}. Reemplaza a study_mixture_layer__PLIN (haz 4*l*, "
    f"ventana unica 1deg, time-resolved): NO comparar entre tandas con haz distinto."
)


# ===========================================================================
# Loop
# ===========================================================================
run_counter = 0
for index, fraction in enumerate(fraction_list):
    for rep in range(N_REPLICAS):
        name = f"fraction_{fraction:.2f}__rep{rep}"
        print(f"\n\n=== Corrida: {name} ===")
        sweep.run(run_counter, name,
                  lambda exp, x=fraction, i=index, rep=rep: run_mixture_simulation(exp, x, i, rep))
        run_counter += 1

# referencia emparejada (gate x=0): frac_index distinto para semilla unica
if RUN_MATCHED_REF:
    REF_INDEX = len(fraction_list)      # = 11, no colisiona con frac_index 0..10
    for rep in range(N_REPLICAS):
        name = f"matched_ref_r{RADIUS_2:.3f}__rep{rep}"
        print(f"\n\n=== Corrida: {name} ===")
        sweep.run(run_counter, name,
                  lambda exp, i=REF_INDEX, rep=rep: run_matched_reference(exp, i, rep))
        run_counter += 1