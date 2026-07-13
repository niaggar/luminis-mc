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

from utils.time import build_time_grid, depth_report

set_log_level(LogLevel.info)

# ===========================================================================
# Salida
# ===========================================================================
EXP_NAME = "study_homog_timeresolved__PCIR__beam2500"
BASE_DIR = "/home/niaggar/Developer/luminis-mc/temporal_results"

sweep = SweepManager(EXP_NAME, BASE_DIR, timestamped=False)
sweep.snapshot_master_script(__main__.__file__)

# ===========================================================================
# Parametros fisicos  (identicos a §5.1: mismas curvas maestras homogeneas,
# ahora RESUELTAS EN TIEMPO -> sirven de referencia para mezcla/estratificado)
# ===========================================================================
radius_values = [0.020, 0.035, 0.055, 0.075, 0.100, 0.175]

VOLUME_FRACTION = 0.10
N_PARTICLE = 1.59
N_MEDIUM = 1.33
WAVELENGTH = 0.514             # um
MU_A_PERCENT = 0.0             # sin absorcion (gate de reciprocidad limpio)

# Laser: polarizacion CIRCULAR (m=1/sqrt(2), n=i/sqrt(2)), incidencia normal.
LASER_M = 1 / np.sqrt(2)
LASER_N = 1j / np.sqrt(2)
LASER_RADIUS = 2500           # um  (regimen de onda plana)
LASER_TYPE = LaserSource.Gaussian
POLARIZATION = "circular"

# Funcion de fase
PHASEF_THETA_MIN = 0.0
PHASEF_THETA_MAX = np.pi
PHASEF_NDIV = 10_000

# ---------------------------------------------------------------------------
# Grilla angular POR ESPECIE, en unidades reducidas q = k*l*_teorico*theta.
#   N_THETA reducido vs §5.1 (500/200): el sensor 3D (theta x t) reparte hits,
#   sobre-resolver theta diluye los bins temporales tardios. Sigue el criterio
#   del estratificado (§5.4). Ventana ancha (q_fine=7.5) para dar cabeza al
#   ensanchamiento del cono a tiempos tempranos (q_w(t) ~ sqrt(3 tau*/t)).
# ---------------------------------------------------------------------------
N_THETA_1 = 400               # ventana fina (cono)
N_THETA_2 = 100               # ventana cola (fondo/baseline)
Q_FINE, Q_TAIL = 7.5, 40.0
N_PHI = 36
PHI_MAX = 2 * np.pi
k = 2 * np.pi * N_MEDIUM / WAVELENGTH

# ---------------------------------------------------------------------------
# Grilla temporal REDUCIDA POR ESPECIE (reescalado por construccion).
#   Mismo (n_bins, t_max_taustar, binning) para TODOS los radios -> el bin i de
#   cada radio corresponde al MISMO t/tau*. "Comparar cada instante" = comparar
#   bin contra bin, sin interpolacion. El colapso q_w(t/tau*) sobre sqrt(3 tau*/t)
#   es el analogo temporal del colapso en q de §5.1.
#   binning="geometric" = binear por camino GEOMETRICO (convencion correcta del
#   reloj de simulacion), NO espaciado logaritmico: los bins son UNIFORMES en dt.
#   30 bins / 30 tau* -> dt = 1 tau*/bin. La dinamica vive en 1-30 tau*.
# ---------------------------------------------------------------------------
TIME_NBINS = 100
TIME_TMAX_TAUSTAR = 40

# ---------------------------------------------------------------------------
# Muestreo
#   300k (vs 100k de §5.1): el sensor 3D reparte hits; los bins temporales
#   tardios necesitan mas fotones para no diluirse. 5 replicas = barras por
#   batch-splitting (varianza empirica entre replicas, ahora por bin (theta,t)).
# ---------------------------------------------------------------------------
N_THREADS = 46
N_PHOTONS = 300_000
N_REPLICAS = 5
SEED_BASE = 20260713          # distinto de homogeneo-integrado (...10),
                              # mezcla (...11) y estratificado (...12)


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
    """Albedo bien definido para que la agregacion mu_a^(i) no caiga en 0/0.
    La escala absoluta de mu_s no afecta el transporte; solo el cociente
    mu_a/mu_s. Absorcion primero, luego scattering (recomputa mu_t)."""
    medium.set_absorption_coefficient(mu_a_percent)
    medium.set_scattering_coefficient(1.0 - mu_a_percent)
    medium.set_mean_free_path(mfp)


# ===========================================================================
# Setup COMPARTIDO por radio (una sola vez: evita reconstruir la funcion de
# fase en cada replica; las refs en los dicts mantienen keep-alive de pybind).
# ===========================================================================
SPECIES, DQ, GRID_R, THETA = {}, {}, {}, {}

print("==== Setup (homogeneo time-resolved) ====")
print(f"  malla temporal COMPARTIDA en t/tau*: {TIME_NBINS} bins / "
      f"{TIME_TMAX_TAUSTAR} tau*  ->  dt = "
      f"{TIME_TMAX_TAUSTAR/TIME_NBINS:.2f} tau*/bin")
print("  r[um] |  l*[um] | tau*[fs] |  dt[fs] |  M  | z_diff[um] (=z_diff/l*)")
for rad in radius_values:
    esp = build_species(rad)
    dq = derived_quantities(esp, VOLUME_FRACTION)
    set_albedo(esp, MU_A_PERCENT, dq['mean_free_path'])

    grid = build_time_grid(
        dq['transport_mean_free_path'], N_MEDIUM,
        n_bins=TIME_NBINS, t_max_taustar=TIME_TMAX_TAUSTAR,
    )
    l_star = dq['transport_mean_free_path']
    theta_1 = Q_FINE / (k * l_star)
    theta_2 = Q_TAIL / (k * l_star)

    SPECIES[rad], DQ[rad], GRID_R[rad], THETA[rad] = esp, dq, grid, (theta_1, theta_2)

    M = grid["t_max_fs"] / grid["tau_star_fs"]
    z_diff = (2 * M / 3) ** 0.5 * l_star
    print(f"  {rad:5.3f} | {l_star:7.2f} | {grid['tau_star_fs']:8.2f} | "
          f"{grid['dt_fs']:7.1f} | {M:3.0f} | {z_diff:8.1f}  ({z_diff/l_star:.1f} l*)")


# ===========================================================================
# Corrida
# ===========================================================================
def run_cbs(exp, radius, rad_index, rep):
    especie = SPECIES[radius]
    dq = DQ[radius]
    grid = GRID_R[radius]
    theta_1, theta_2 = THETA[radius]

    sample = Sample(N_MEDIUM)
    sample.add_layer(especie, 0, float('inf'))

    laser = Laser(LASER_M, LASER_N, WAVELENGTH, LASER_RADIUS, LASER_TYPE)

    d_theta_1 = theta_1 / N_THETA_1
    d_theta_2 = theta_2 / N_THETA_2
    d_phi = PHI_MAX / N_PHI

    # --- EJE TEMPORAL ACTIVO: bin 0 = integrado; 1..N = time-resolved ---
    t_max = grid["t_max_sim"]
    dt = grid["dt_sim"]

    sens = SensorsGroup()
    det_1 = sens.add_detector(FarFieldCBSSensor(theta_1, PHI_MAX, t_max, d_theta_1, d_phi, dt, True))
    det_1.set_theta_limit(0, theta_1)
    det_1.set_phi_slices([0, np.pi/4, np.pi/2])

    det_2 = sens.add_detector(FarFieldCBSSensor(theta_2, PHI_MAX, t_max, d_theta_2, d_phi, dt, True))
    det_2.set_theta_limit(theta_1 * 0.9, theta_2)          # solape para stitching
    det_2.set_phi_slices([0, np.pi/4, np.pi/2])

    stats = sens.add_detector(StatisticsSensor(z=0, absorb=True))
    stats.set_theta_limit(0, theta_2)

    config = SimConfig()
    config.n_photons = N_PHOTONS
    config.sample = sample
    config.detector = sens
    config.laser = laser
    config.track_reverse_paths = True                      # imprescindible para CBS
    config.pin_threads_to_cores = False
    config.n_threads = N_THREADS
    config.show_progress = True
    config.seed = SEED_BASE + rad_index * 1000 + rep

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
        # --- eje temporal (SIM units para el sensor; fisica para el analisis) ---
        "t_max": grid["t_max_sim"],
        "d_time": grid["dt_sim"],
        "time_grid": grid,
        "tau_star_fs": grid["tau_star_fs"],
        "time_nbins": TIME_NBINS,
        "time_tmax_taustar": TIME_TMAX_TAUSTAR,
        "time_anchor": "per_species_lstar (reducido t/tau*)",
        "polarization": POLARIZATION,
        "seed": config.seed,
        "replica": rep,
        "laser_radius_um": LASER_RADIUS,
        "q_fine": Q_FINE,
        "q_tail": Q_TAIL,
    }
    exp.save_params(config, extra=extra)

    t0 = time.time()
    run_simulation_parallel(config)
    print("n_photons:", N_PHOTONS, "| runtime_s:", time.time() - t0,
          "| hits:", det_1.hits + det_2.hits)

    exp.save_sensors({"farfield_cbs_1": det_1, "farfield_cbs_2": det_2, "statistics": stats})
    exp.save_processed("farfield_cbs_1", postprocess_farfield_cbs(det_1, N_PHOTONS), sensor=det_1)
    exp.save_processed("farfield_cbs_2", postprocess_farfield_cbs(det_2, N_PHOTONS), sensor=det_2)

    _keep_alive = (sample, especie); del _keep_alive


# ===========================================================================
# README (f-string: no se desactualiza en lo que importa)
# ===========================================================================
sweep.log_readme(
    f"CBS homogeneo TIME-RESOLVED -- una especie RGD por corrida, polarizacion "
    f"{POLARIZATION.upper()} (m={LASER_M:.0f}, n={LASER_N:.0f}), estimador "
    f"next-event. Radios {radius_values} um, f={VOLUME_FRACTION}, "
    f"n_p={N_PARTICLE}, n_med={N_MEDIUM}, lambda={WAVELENGTH} um, sin absorcion. "
    f"Haz gaussiano FIJO w={LASER_RADIUS} um (onda plana). Doble ventana angular "
    f"POR ESPECIE en unidades reducidas q=k*l*_teorico*theta: fina q in "
    f"[0,{Q_FINE}] ({N_THETA_1} bins), cola q in [{Q_FINE},{Q_TAIL}] "
    f"({N_THETA_2} bins), solape [0.9,1.0]*theta_1 para stitching. "
    f"EJE TEMPORAL ACTIVO: grilla REDUCIDA por especie, IDENTICA en t/tau* para "
    f"todos los radios ({TIME_NBINS} bins hasta {TIME_TMAX_TAUSTAR} tau*, "
    f"dt={TIME_TMAX_TAUSTAR/TIME_NBINS:.2f} tau*/bin) = "
    f"camino geometrico, bins uniformes). bin 0 = integrado; 1..N = "
    f"time-resolved. El bin i de cada radio corresponde al MISMO t/tau* -> "
    f"comparacion bin-a-bin sin interpolar; colapso esperado q_w(t/tau*) ~ "
    f"sqrt(3 tau*/t). Estas curvas maestras homogeneas son la referencia contra "
    f"la que se leen mezcla (§5.3) y estratificado (§5.4). {N_REPLICAS} "
    f"replicas/radio, semillas SEED_BASE={SEED_BASE} + 1000*rad_index + rep, en "
    f"params. tau*_fs y time_grid guardados -> t[fs] = (t/tau*)*tau*_fs a "
    f"posteriori. Varianza VALIDA = empirica entre replicas por bin (theta,t); "
    f"las estimaciones internas Poisson/chi2 del next-event son invalidas."
)


# ===========================================================================
# Loop
# ===========================================================================
for index, rad in enumerate(radius_values):
    for rep in range(N_REPLICAS):
        name = f"radius_{rad:.3f}__rep{rep}"
        print(f"\n\n=== Corrida: {name} ===")
        sweep.run(index * N_REPLICAS + rep, name,
                  lambda exp, rad=rad, i=index, rep=rep: run_cbs(exp, rad, i, rep))