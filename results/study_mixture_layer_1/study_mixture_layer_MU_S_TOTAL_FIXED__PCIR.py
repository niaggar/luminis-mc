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
    MixtureLayer
)


set_log_level(LogLevel.info)

# ===========================================================================
# Salida
# ===========================================================================
EXP_NAME = "study_mixture_layer_MU_S_TOTAL_FIXED__PCIR"
BASE_DIR = "/Users/niaggar/Documents/Thesis/tests"

sweep = SweepManager(EXP_NAME, BASE_DIR, timestamped=False)
sweep.snapshot_master_script(__main__.__file__)
sweep.log_readme(
    "CBS validation -- MIXTURE of two IDENTICAL species (linear pol, estimator). "
    "Equivalence test against sim_homogeneous.py: mixture(n/2,n/2) == homogeneous(n)."
)

# ===========================================================================
# Parametros fisicos
#   >>> IDENTICOS A sim_homogeneous.py <<<
# ===========================================================================
RADIUS_1 = 0.175                 # um
RADIUS_2 = 0.035                 # um

VOLUME_FRACTION = 0.10
N_PARTICLE = 1.59
N_MEDIUM = 1.33
WAVELENGTH = 0.514             # um
MU_A_PERCENT = 0.0             # sin absorcion (gate de reciprocidad limpio)

LASER_M = 1 / np.sqrt(2)
LASER_N = -1j / np.sqrt(2)
LASER_RADIUS_MFP = 4.0         # en unidades de l_s
LASER_TYPE = LaserSource.Gaussian

# Funcion de fase
PHASEF_THETA_MIN = 0.0
PHASEF_THETA_MAX = np.pi
PHASEF_NDIV = 10_000

# ---------------------------------------------------------------------------
# Grilla angular   >>> IDENTICA A sim_homogeneous.py <<<
# ---------------------------------------------------------------------------
N_THETA = 1000
N_PHI = 36
PHI_MAX = 2 * np.pi
THETA_MAX = np.deg2rad(1)

T_MAX = 0
D_TIME = 0

# ---------------------------------------------------------------------------
# Muestreo   >>> IDENTICO A sim_homogeneous.py <<<
# ---------------------------------------------------------------------------
N_PHOTONS = 100_000
N_THREADS = 44


# ===========================================================================
# Helpers
# ===========================================================================
def number_density(radius, volume_fraction):
    """n = f / ((4/3) pi r^3)   [particulas / um^3].

    Misma convencion que usa derived_quantities internamente para n; garantiza
    que sum_i n_i sigma_i (mezcla) == n sigma (homogeneo).
    """
    return volume_fraction / ((4.0 / 3.0) * np.pi * radius ** 3)


def build_species(rad):
    """Una especie RGD con funcion de fase EMC (seccion eficaz NO nula)."""
    phase = RayleighDebyeEMCPhaseFunction(
        WAVELENGTH, rad, N_PARTICLE, N_MEDIUM,
        PHASEF_NDIV, PHASEF_THETA_MIN, PHASEF_THETA_MAX,
    )
    return RGDMedium(phase, rad, N_PARTICLE, N_MEDIUM, WAVELENGTH)


def set_albedo(medium, mu_a_percent):
    """Fija un albedo bien definido para que la agregacion mu_a^(i) no caiga en 0/0.

    La escala absoluta de mu_s aqui NO afecta el transporte (la mu_s total la fija
    n_i sigma_i); solo importa el cociente mu_a/mu_s. Absorcion primero, luego
    scattering (recomputa mu_t).
    """
    mu_s = 1.0 - mu_a_percent
    mu_a = mu_a_percent
    medium.set_absorption_coefficient(mu_a)
    medium.set_scattering_coefficient(mu_s)


# ===========================================================================
# Corrida
# ===========================================================================
def run_mixture_simulation(exp: Experiment, fraction: float):
    """Construye una MixtureLayer de dos especies, corre CBS y persiste."""

    ref_1 = build_species(RADIUS_1)
    ref_2 = build_species(RADIUS_2)
    dq_1 = derived_quantities(ref_1, VOLUME_FRACTION)
    dq_2 = derived_quantities(ref_2, VOLUME_FRACTION)

    # n_total = number_density(RADIUS_1, VOLUME_FRACTION)
    # nd_1 = n_total * fraction
    # nd_2 = n_total - nd_1
    # densities = [nd_1, nd_2]

    # secciones eficaces y anisotropias (independientes de la densidad)
    sigma_1 = dq_1['scattering_efficiency'] * np.pi * RADIUS_1**2
    sigma_2 = dq_2['scattering_efficiency'] * np.pi * RADIUS_2**2

    # ancla mu_s_total al medio PURO de especie 1 a f=0.10:
    # asi x=1 coincide con tu punto fraction=1 actual
    MU_S_TOTAL = number_density(RADIUS_1, VOLUME_FRACTION) * sigma_1

    x = fraction                               # fraccion pesada por scattering de esp. 1
    nd_1 = x * MU_S_TOTAL / sigma_1
    nd_2 = (1.0 - x) * MU_S_TOTAL / sigma_2
    densities = [nd_1, nd_2]

    # diagnostico: cuanto aporta cada especie al scattering (esto es lo que "ve" el CBS)
    mu1, mu2 = nd_1 * sigma_1, nd_2 * sigma_2
    print(f"share mu_s: sp1={mu1/(mu1+mu2):.3f}  sp2={mu2/(mu1+mu2):.3f}  mu_s_total={mu1+mu2:.4e}")


    species = [build_species(RADIUS_1), build_species(RADIUS_2)]
    for med in species:
        set_albedo(med, MU_A_PERCENT)

    sample = Sample(N_MEDIUM)
    sample.add_mixture_layer(species, densities, 0.0, float("inf"))

    layer = sample.layers[0]
    mfp_layer = 0.0
    if isinstance(layer, MixtureLayer):
        mfp_layer = float(layer.mfp_total)
        print("---- MixtureLayer ----")
        print(f"mu_s_total: {layer.mu_s_total:.6e}")
        print(f"mu_a_total: {layer.mu_a_total:.6e}")
        print(f"mfp_total (capa): {mfp_layer:.6f} um")


    print("---- Parametros de la simulacion (MEZCLA) ----")
    print(f"radio 1: {RADIUS_1:.3f} um   radio 2: {RADIUS_2:.3f} um")
    # print(f"n_total: {n_total:.4e} 1/um^3   nd_1: {nd_1:.4e} 1/um^3   nd_2: {nd_2:.4e} 1/um^3")
    print(f"eficiencia de scattering 1: {dq_1['scattering_efficiency']:.3f}")
    print(f"eficiencia de scattering 2: {dq_2['scattering_efficiency']:.3f}")

    print(f"camino libre medio l_s 1: {dq_1['mean_free_path']:.4f} um")
    print(f"camino libre medio l_s 2: {dq_2['mean_free_path']:.4f} um")
    
    print(f"camino de transporte l* 1: {dq_1['transport_mean_free_path']:.4f} um")
    print(f"camino de transporte l* 2: {dq_2['transport_mean_free_path']:.4f} um")

    print(f"factor de anisotropia g 1: {dq_1['anisotropy_g']:.4f}")
    print(f"factor de anisotropia g 2: {dq_2['anisotropy_g']:.4f}")
    
    print(f"theta_coherent 1: {dq_1['theta_coherent'] * 1e3:.4f} mrad")
    print(f"theta_coherent 2: {dq_2['theta_coherent'] * 1e3:.4f} mrad")

    laser = Laser(
        LASER_M, LASER_N, WAVELENGTH,
        LASER_RADIUS_MFP * mfp_layer, LASER_TYPE,
    )

    # --- grilla angular (misma receta y mismos numeros que la homogenea) ---
    d_theta = THETA_MAX / N_THETA
    d_phi = PHI_MAX / N_PHI

    # --- sensores ---
    sens = SensorsGroup()
    det = sens.add_detector(
        FarFieldCBSSensor(THETA_MAX, PHI_MAX, T_MAX, d_theta, d_phi, D_TIME, True)
    )
    det.set_theta_limit(0, THETA_MAX)
    det.set_phi_slices([0, np.pi/4, np.pi/2, 3*np.pi/4])

    stats = sens.add_detector(StatisticsSensor(z=0, absorb=True))
    stats.set_theta_limit(0, THETA_MAX)

    # --- config ---
    config = SimConfig()
    config.n_photons = N_PHOTONS
    config.sample = sample
    config.detector = sens
    config.laser = laser
    config.track_reverse_paths = True          # <-- imprescindible para CBS
    config.pin_threads_to_cores = False
    config.n_threads = N_THREADS
    config.show_progress = True

    # --- params: derivados de mezcla + info de grilla ---
    dq_mix = derived_quantities_mixture(species, densities)

    extra = {
        **dq_mix,
        "radius": [RADIUS_1, RADIUS_2],
        "volume_fraction": VOLUME_FRACTION,
        "number_densities": densities,
        "mean_free_paths": [dq_1['mean_free_path'], dq_2['mean_free_path']],                       # canonico
        "transport_mean_free_paths": [dq_1['transport_mean_free_path'], dq_2['transport_mean_free_path']],
        "theta_coherents": [dq_1['theta_coherent'], dq_2['theta_coherent']],
        "mfp_total_layer": mfp_layer,
        "theta_max": THETA_MAX,
        "d_theta": d_theta,
        "d_phi": d_phi,
        "n_theta": N_THETA,
        "n_phi": N_PHI,
        "polarization": "linear",
        "layer_kind": "mixture",
        "n_species": len(species),
    }

    # NOTA: save_params auto-captura la MixtureLayer via capture_params (§14).
    # Si esto lanzara sobre la mezcla, el problema esta en ese path (vars(medium)
    # sobre MixtureLayer), no en la fisica de la simulacion.
    exp.save_params(config, extra=extra)

    # --- run ---
    t0 = time.time()
    run_simulation_parallel(config)
    print("runtime_s:", time.time() - t0)
    print("hits:", det.hits)

    # --- guardar RAW + procesados ---
    sensors_to_save = {"farfield_cbs": det, "statistics": stats}
    exp.save_sensors(sensors_to_save)

    cbs = postprocess_farfield_cbs(det, N_PHOTONS)
    exp.save_processed("farfield_cbs", cbs, sensor=det)


    # keep-alive explicito: referenciar species/sample al final del scope asegura
    # que sobreviven a toda la corrida (el binding guarda punteros crudos).
    _keep_alive = (species, ref_1, ref_2, sample)
    del _keep_alive


# ===========================================================================
# Una sola corrida (sweep de un elemento -> load_sweep lo lee igual)
# ===========================================================================

fraction_list = np.linspace(0.0, 1.0, 11)  # fraccion de la especie 1 (la otra es 1-frac)


for index, fraction in enumerate(fraction_list):
    run_name = f"fraction_{fraction:.2f}"

    print(f"\n\n=== Corrida: {run_name} ===\n")
    sweep.run(index, run_name, lambda exp: run_mixture_simulation(exp, fraction=fraction))