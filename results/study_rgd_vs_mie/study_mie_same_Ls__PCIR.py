import __main__
import time
import numpy as np

from luminis_mc import (
    Experiment,
    SweepManager,
    Laser, RGDMedium, MieMedium, Sample, FarFieldCBSSensor, StatisticsSensor, SensorsGroup,
    SimConfig, RayleighDebyeEMCPhaseFunction, MiePhaseFunction,
    run_simulation_parallel, postprocess_farfield_cbs,
    derived_quantities,
    set_log_level, LogLevel, LaserSource,
)

from utils.time import build_time_grid, depth_report

set_log_level(LogLevel.info)

# ===========================================================================
# Salida
# ===========================================================================
EXP_NAME = "study_mie_same_Ls__PCIR__beam2500"
BASE_DIR = "/home/niaggar/Developer/luminis-mc/temporal_results"

sweep = SweepManager(EXP_NAME, BASE_DIR, timestamped=False)
sweep.snapshot_master_script(__main__.__file__)

# ===========================================================================
# QUE ES ESTE ESTUDIO
# ---------------------------------------------------------------------------
#   Medio MIE, TIME-RESOLVED, con INVARIANTE l_s FIJO = l_s(RGD).
#   Comparacion a LABORATORIO FIJO, uno-a-uno contra la corrida RGD gemela.
#
#   * Se impone la MISMA tasa de eventos de scattering (l_s) que en RGD.
#   * La funcion de fase COMPLETA (g, routing de polarizacion, S34, orden bajo)
#     varia libremente RGD -> Mie: eso es lo intrinseco de la particula que
#     queremos observar.
#   * Consecuencia DELIBERADA: l*_sim = l_s/(1 - g_Mie) != l*_RGD. Las grillas
#     angular y temporal quedan ancladas a l*_RGD (aparato fijo) e IDENTICAS a
#     la corrida RGD  ->  los bins (theta, t) coinciden byte a byte  ->  resta
#     uno-a-uno. El corrimiento del cono en q_RGD y del quiebre en t/tau*_RGD
#     ES la firma de Mie, no un artefacto.
#   * Diagnostico clave guardado: lstar_ratio_sim_over_ref = (1-g_RGD)/(1-g_Mie).
#     Dividir por el separa el corrimiento de l*(g) del residual de
#     polarizacion pura.
# ===========================================================================

# ===========================================================================
# Parametros fisicos  (identicos a la corrida RGD gemela)
# ===========================================================================
radius_values = [0.020, 0.035, 0.055, 0.075, 0.100, 0.175]

VOLUME_FRACTION = 0.10
N_PARTICLE = 1.59
N_MEDIUM = 1.33
WAVELENGTH = 0.514             # um
MU_A_PERCENT = 0.0             # sin absorcion (gate de reciprocidad limpio)

C0 = 0.299792458              # um/fs

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
# GRILLA ANGULAR CANONICA -- DEBE COINCIDIR CON LA CORRIDA RGD
#   Anclada POR ESPECIE a l*_RGD (laboratorio fijo), en unidades reducidas
#   q = k*l*_RGD*theta. N_THETA reducido vs §5.1 por el sensor 3D (theta x t).
# ---------------------------------------------------------------------------
N_THETA_1 = 400               # ventana fina (cono)
N_THETA_2 = 100               # ventana cola (fondo/baseline)
Q_FINE, Q_TAIL = 7.5, 40.0
N_PHI = 36
PHI_MAX = 2 * np.pi
k = 2 * np.pi * N_MEDIUM / WAVELENGTH

# ---------------------------------------------------------------------------
# GRILLA TEMPORAL CANONICA -- DEBE COINCIDIR CON LA CORRIDA RGD
#   Anclada a l*_RGD (=> tau*_RGD). Mismo (n_bins, t_max_taustar, binning) que
#   RGD -> el bin i coincide en ambas corridas. binning="geometric" = binear por
#   camino GEOMETRICO (convencion del reloj de simulacion), bins UNIFORMES.
#   OJO: si cambias estos tres numeros, cambialos TAMBIEN en el script RGD, o la
#   comparacion bin-a-bin deja de ser valida.
# ---------------------------------------------------------------------------
TIME_NBINS = 100
TIME_TMAX_TAUSTAR = 40

# ---------------------------------------------------------------------------
# Muestreo  (identico a la corrida RGD gemela)
# ---------------------------------------------------------------------------
N_THREADS = 46
N_PHOTONS = 300_000
N_REPLICAS = 5
SEED_BASE = 20260713          # MISMO que RGD -> misma secuencia de semillas por
                              # (radio, rep): la unica diferencia es RGD vs Mie.


# ===========================================================================
# Helpers
# ===========================================================================
def build_rgd_species(rad):
    """Especie RGD de REFERENCIA (fija l_s y ancla las grillas)."""
    phase = RayleighDebyeEMCPhaseFunction(
        WAVELENGTH, rad, N_PARTICLE, N_MEDIUM,
        PHASEF_NDIV, PHASEF_THETA_MIN, PHASEF_THETA_MAX,
    )
    return RGDMedium(phase, rad, N_PARTICLE, N_MEDIUM, WAVELENGTH)


def build_mie_species(rad):
    """Especie MIE: misma particula, funcion de fase COMPLETA."""
    phase = MiePhaseFunction(
        WAVELENGTH, rad, N_PARTICLE, N_MEDIUM,
        PHASEF_NDIV, PHASEF_THETA_MIN, PHASEF_THETA_MAX,
    )
    return MieMedium(phase, rad, N_PARTICLE, N_MEDIUM, WAVELENGTH)


def set_albedo(medium, mu_a_percent, mfp):
    """Albedo bien definido para que la agregacion mu_a^(i) no caiga en 0/0.
    La escala absoluta de mu_s no afecta el transporte; solo el cociente
    mu_a/mu_s. Absorcion primero, luego scattering (recomputa mu_t)."""
    medium.set_absorption_coefficient(mu_a_percent)
    medium.set_scattering_coefficient(1.0 - mu_a_percent)
    medium.set_mean_free_path(mfp)


# ===========================================================================
# Setup COMPARTIDO por radio (una sola vez).
#   Construye la referencia RGD (para l_s y las anclas de grilla) y el medio
#   Mie (cacheado, con l_s IMPUESTO). Las refs en los dicts mantienen keep-alive
#   de pybind. Imprime la tabla de offsets l*_sim/l*_RGD para chequeo previo.
# ===========================================================================
DQ_RGD, MIE_SPECIES, MIE_DQ, GRID_R, THETA, PROV = {}, {}, {}, {}, {}, {}

print("==== Setup (Mie, l_s FIJO = l_s(RGD), laboratorio fijo) ====")
print(f"  grilla temporal CANONICA (== RGD): {TIME_NBINS} bins / "
      f"{TIME_TMAX_TAUSTAR} tau*  ->  dt = "
      f"{TIME_TMAX_TAUSTAR/TIME_NBINS:.2f} tau*/bin")
print("  r[um] | l*_RGD | l*_sim | ratio | g_RGD  | g_Mie  | tau*RGD[fs] | dt[fs]")
for rad in radius_values:
    # --- referencia RGD: fija l_s y ancla grillas ---
    rgd = build_rgd_species(rad)
    dq_rgd = derived_quantities(rgd, VOLUME_FRACTION)
    set_albedo(rgd, MU_A_PERCENT, dq_rgd['mean_free_path'])

    ls_rgd        = dq_rgd['mean_free_path']                 # scattering MFP fijo
    lstar_rgd_ref = dq_rgd['transport_mean_free_path']       # ancla de grilla
    g_rgd         = dq_rgd['anisotropy_g']

    # --- medio Mie con l_s IMPUESTO ---
    mie = build_mie_species(rad)
    mie_dq = derived_quantities(mie, VOLUME_FRACTION)         # cantidades NATIVAS Mie (f=0.10)
    set_albedo(mie, MU_A_PERCENT, ls_rgd)                     # impone l_s = l_s(RGD)

    g_mie          = mie_dq['anisotropy_g']
    lstar_sim      = ls_rgd / (1.0 - g_mie)                   # l* efectivo REAL del medio Mie
    lstar_ratio    = lstar_sim / lstar_rgd_ref               # = (1-g_RGD)/(1-g_Mie)
    taustar_sim_fs = lstar_sim * N_MEDIUM / C0

    # --- grillas ancladas a l*_RGD (laboratorio fijo, == corrida RGD) ---
    grid = build_time_grid(
        lstar_rgd_ref, N_MEDIUM,
        n_bins=TIME_NBINS, t_max_taustar=TIME_TMAX_TAUSTAR,
    )
    theta_1 = Q_FINE / (k * lstar_rgd_ref)
    theta_2 = Q_TAIL / (k * lstar_rgd_ref)

    DQ_RGD[rad]      = dq_rgd
    MIE_SPECIES[rad] = mie
    MIE_DQ[rad]      = mie_dq
    GRID_R[rad]      = grid
    THETA[rad]       = (theta_1, theta_2)
    PROV[rad] = {
        "ls_fixed_from_rgd":         ls_rgd,
        "g_rgd":                     g_rgd,
        "g_mie":                     g_mie,
        "lstar_rgd_ref":             lstar_rgd_ref,   # ancla de grilla
        "lstar_sim":                 lstar_sim,       # l* efectivo REAL Mie
        "taustar_rgd_ref_fs":        grid["tau_star_fs"],
        "taustar_sim_fs":            taustar_sim_fs,
        "lstar_ratio_sim_over_ref":  lstar_ratio,
    }

    print(f"  {rad:5.3f} | {lstar_rgd_ref:6.2f} | {lstar_sim:6.2f} | "
          f"{lstar_ratio:5.3f} | {g_rgd:6.4f} | {g_mie:6.4f} | "
          f"{grid['tau_star_fs']:11.2f} | {grid['dt_fs']:6.1f}")


# ===========================================================================
# Corrida
# ===========================================================================
def run_cbs(exp, radius, rad_index, rep):
    dq_rgd = DQ_RGD[radius]
    mie_specie = MIE_SPECIES[radius]
    mie_dq = MIE_DQ[radius]
    grid = GRID_R[radius]
    theta_1, theta_2 = THETA[radius]
    prov = PROV[radius]

    sample = Sample(N_MEDIUM)
    sample.add_layer(mie_specie, 0, float('inf'))

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
        # --- dqs anidados (patron del estratificado; sin spread ambiguo) ---
        "mie_dq_native": mie_dq,       # cantidades NATIVAS Mie a f=0.10 (referencia)
        "rgd_dq_ref":    dq_rgd,       # RGD que fija l_s y ancla las grillas
        # --- provenance del invariante l_s fijo (laboratorio fijo) ---
        **prov,                        # ls_fixed_from_rgd, g_rgd, g_mie, lstar_rgd_ref,
                                       # lstar_sim, taustar_rgd_ref_fs, taustar_sim_fs,
                                       # lstar_ratio_sim_over_ref
        "scattering_model": "mie",
        "study_invariant":  "ls_fixed",
        "grid_anchor":      "lstar_rgd (fixed-lab)",
        # --- angular (anclado a l*_RGD) ---
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
        # --- temporal (SIM units para el sensor; fisica en prov/grid) ---
        "t_max": grid["t_max_sim"],
        "d_time": grid["dt_sim"],
        "time_grid": grid,
        "tau_star_fs": grid["tau_star_fs"],   # == taustar_rgd_ref_fs (ancla)
        "time_nbins": TIME_NBINS,
        "time_tmax_taustar": TIME_TMAX_TAUSTAR,
        "time_anchor": "lstar_rgd (fixed-lab; t/tau*_RGD)",
        # --- meta ---
        "polarization": POLARIZATION,
        "seed": config.seed,
        "replica": rep,
        "laser_radius_um": LASER_RADIUS,
    }
    exp.save_params(config, extra=extra)

    t0 = time.time()
    run_simulation_parallel(config)
    print(f"[{EXP_NAME}] r={radius:.3f} rep={rep} | n_photons: {N_PHOTONS} | "
          f"runtime_s: {time.time() - t0:.1f} | hits: {det_1.hits + det_2.hits} | "
          f"l*_sim/l*_RGD={prov['lstar_ratio_sim_over_ref']:.3f}")

    exp.save_sensors({"farfield_cbs_1": det_1, "farfield_cbs_2": det_2, "statistics": stats})
    exp.save_processed("farfield_cbs_1", postprocess_farfield_cbs(det_1, N_PHOTONS), sensor=det_1)
    exp.save_processed("farfield_cbs_2", postprocess_farfield_cbs(det_2, N_PHOTONS), sensor=det_2)

    _keep_alive = (sample, mie_specie); del _keep_alive


# ===========================================================================
# README (f-string: no se desactualiza en lo que importa)
# ===========================================================================
sweep.log_readme(
    f"CBS homogeneo TIME-RESOLVED -- medio MIE con INVARIANTE l_s FIJO = l_s(RGD), "
    f"comparacion a LABORATORIO FIJO uno-a-uno contra la corrida RGD gemela. "
    f"Una especie Mie por corrida, polarizacion {POLARIZATION.upper()} "
    f"(m={LASER_M:.0f}, n={LASER_N:.0f}), estimador next-event. Radios "
    f"{radius_values} um, f={VOLUME_FRACTION}, n_p={N_PARTICLE}, n_med={N_MEDIUM}, "
    f"lambda={WAVELENGTH} um, sin absorcion. Haz gaussiano FIJO w={LASER_RADIUS} um "
    f"(onda plana). "
    f"INVARIANTE: se impone l_s = l_s(RGD) via set_mean_free_path -> misma tasa de "
    f"eventos de scattering; la funcion de fase COMPLETA (g, routing de "
    f"polarizacion S2(180)=-S1(180), S34, orden bajo) varia libremente RGD->Mie. "
    f"CONSECUENCIA DELIBERADA: l*_sim = l_s/(1-g_Mie) != l*_RGD. Las grillas "
    f"angular y temporal quedan ancladas a l*_RGD (aparato fijo) e IDENTICAS a la "
    f"corrida RGD: angular POR ESPECIE en q=k*l*_RGD*theta, fina q in [0,{Q_FINE}] "
    f"({N_THETA_1} bins), cola q in [{Q_FINE},{Q_TAIL}] ({N_THETA_2} bins), solape "
    f"[0.9,1.0]*theta_1 para stitching; temporal {TIME_NBINS} bins hasta "
    f"{TIME_TMAX_TAUSTAR} tau*_RGD (dt={TIME_TMAX_TAUSTAR/TIME_NBINS:.2f} tau*/bin, "
    f"bins uniformes; bin 0 = "
    f"integrado, 1..N = time-resolved). Los bins (theta,t) COINCIDEN con RGD -> "
    f"resta uno-a-uno. El corrimiento del cono en q_RGD y del quiebre en "
    f"t/tau*_RGD ES la firma de Mie, NO artefacto (a diferencia de anclar a "
    f"l*_Mie nativo, donde ese corrimiento se divide fuera). DIAGNOSTICO guardado "
    f"lstar_ratio_sim_over_ref = l*_sim/l*_RGD = (1-g_RGD)/(1-g_Mie): dividir por "
    f"el separa el corrimiento de l*(g) del residual de polarizacion pura. "
    f"Provenance en params: mie_dq_native (Mie f=0.10), rgd_dq_ref, ls_fixed_from_rgd, "
    f"g_rgd, g_mie, lstar_rgd_ref (ancla), lstar_sim (l* REAL), taustar_rgd_ref_fs, "
    f"taustar_sim_fs. IMPORTANTE: N_THETA, Q, TIME_NBINS, TIME_TMAX_TAUSTAR y "
    f"binning DEBEN coincidir con el script RGD o la comparacion bin-a-bin no es "
    f"valida. {N_REPLICAS} replicas/radio, semillas SEED_BASE={SEED_BASE} + "
    f"1000*rad_index + rep (MISMA base que RGD: unica diferencia = modelo de "
    f"scattering). Varianza VALIDA = empirica entre replicas por bin (theta,t); "
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