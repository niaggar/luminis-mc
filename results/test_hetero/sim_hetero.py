"""
Validacion CBS -- corrida de MEZCLA (MixtureLayer con DOS especies IDENTICAS).

Contraparte de sim_homogeneous.py. Divide la poblacion homogenea de densidad n
en dos especies FISICAMENTE IDENTICAS con densidad n/2 cada una:

    HomogeneousLayer(densidad n)   ==   MixtureLayer([esp, esp], [n/2, n/2])

Como las dos especies comparten funcion de fase, matriz de amplitud y seccion
eficaz, el transporte queda fijado por mu_s_total = (n/2 + n/2) sigma = n sigma,
IDENTICO al homogeneo. Si el MixtureLayer esta bien, las curvas de realce co-pol
deben coincidir dentro del ruido Monte Carlo.

Robustez de la construccion:
  - La GRILLA angular se calcula desde derived_quantities(especie, VOLUME_FRACTION)
    con la fraccion de volumen COMPLETA. Para especies identicas esto es
    exactamente el agregado de la mezcla (mfp_total, l*, theta_coh), asi que la
    grilla resulta bit-identica a la de sim_homogeneous.py SIN depender de las
    claves de derived_quantities_mixture.
  - Se hace un CROSS-CHECK: MixtureLayer.mfp_total (salida real del C++) vs el
    mfp derivado. Deben coincidir a precision de maquina; si no, la seleccion de
    especie o la agregacion de mu_s esta mal.
  - Guard anti-NaN: cada especie necesita un albedo BIEN DEFINIDO (mu_s>0, mu_a)
    porque mu_a_total = sum_i mu_s^(i) (mu_a/mu_s)_i cae en 0/0 si mu_s^(i)=0.
    Se setea absorcion y luego scattering en CADA especie ANTES de la mezcla.
  - keep-alive: la lista `species` se mantiene viva toda la corrida (el binding
    guarda punteros crudos a las especies).

Notas de API (§15.13 de code_description):
  - MixtureLayer necesita seccion eficaz NO NULA por especie -> funcion de fase
    EMC/Mie (aqui RayleighDebyeEMCPhaseFunction). Otras devuelven 0 y el ctor
    lanza (mu_s_total <= 0).
  - La mu_s de la mezcla sale de n_i sigma^(i), NO de set_scattering_coefficient;
    pero set_scattering/absorption por especie SI fijan el albedo agregado.
  - set_mean_free_path por especie es IRRELEVANTE en la mezcla (la capa muestrea
    desde mfp_total).

Semilla, numero de fotones y grilla FIJOS e IDENTICOS a sim_homogeneous.py.
"""

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
)

# MixtureLayer es opcional para el cross-check (solo diagnostico).
try:
    from luminis_mc import MixtureLayer
except Exception:
    MixtureLayer = None


set_log_level(LogLevel.info)

# ===========================================================================
# Salida
# ===========================================================================
EXP_NAME = "cbs_validation_mixture"
BASE_DIR = "/Users/niaggar/Documents/Thesis/tests"   # <-- ajustar a tu entorno

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
RADIUS = 0.175                 # um  (fuera de RGD estricto; test de implementacion)
VOLUME_FRACTION = 0.10
N_PARTICLE = 1.59
N_MEDIUM = 1.33
WAVELENGTH = 0.514             # um
MU_A_PERCENT = 0.0             # sin absorcion (gate de reciprocidad limpio)

# Laser: polarizacion LINEAL a lo largo de m (X), incidencia normal.
LASER_M = 1.0
LASER_N = 0.0
LASER_RADIUS_MFP = 4.0         # en unidades de l_s
LASER_TYPE = LaserSource.Gaussian

# Funcion de fase
PHASEF_THETA_MIN = 0.0
PHASEF_THETA_MAX = np.pi
PHASEF_NDIV = 1_000

# ---------------------------------------------------------------------------
# Grilla angular   >>> IDENTICA A sim_homogeneous.py <<<
# ---------------------------------------------------------------------------
N_THETA = 500
N_PHI = 1
PHI_MAX = 2 * np.pi
THETA_CONE_FACTOR = 10.0       # theta_max = K * theta_coherent

T_MAX = 0
D_TIME = 0

# ---------------------------------------------------------------------------
# Muestreo   >>> IDENTICO A sim_homogeneous.py <<<
# ---------------------------------------------------------------------------
N_PHOTONS = 100_000
N_THREADS = 7
SEED = 12345                   # misma semilla que la corrida homogenea

RUN_ANALOG_CHECK = False


# ===========================================================================
# Helpers
# ===========================================================================
def number_density(radius, volume_fraction):
    """n = f / ((4/3) pi r^3)   [particulas / um^3].

    Misma convencion que usa derived_quantities internamente para n; garantiza
    que sum_i n_i sigma_i (mezcla) == n sigma (homogeneo).
    """
    return volume_fraction / ((4.0 / 3.0) * np.pi * radius ** 3)


def build_species():
    """Una especie RGD con funcion de fase EMC (seccion eficaz NO nula)."""
    phase = RayleighDebyeEMCPhaseFunction(
        WAVELENGTH, RADIUS, N_PARTICLE, N_MEDIUM,
        PHASEF_NDIV, PHASEF_THETA_MIN, PHASEF_THETA_MAX,
    )
    return RGDMedium(phase, RADIUS, N_PARTICLE, N_MEDIUM, WAVELENGTH)


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
def run_mixture_simulation(exp: Experiment):
    """Construye una MixtureLayer de dos especies identicas, corre CBS y persiste."""
    # --- especie de referencia (SIN tocar albedo) para la fisica derivada ---
    # Para especies identicas, derived_quantities con la fraccion COMPLETA == el
    # agregado de la mezcla. De aqui sale la grilla, identica a la homogenea.
    ref = build_species()
    dq = derived_quantities(ref, VOLUME_FRACTION)
    mean_free_path = dq["mean_free_path"]
    theta_coherent = dq["theta_coherent"]

    # --- dos especies IDENTICAS, densidades que SUMAN la del homogeneo ---
    n_total = number_density(RADIUS, VOLUME_FRACTION)
    nd_each = n_total / 2.0
    densities = [nd_each, nd_each]

    species = [build_species(), build_species()]   # <-- mantener vivo toda la corrida
    for med in species:
        set_albedo(med, MU_A_PERCENT)              # <-- guard anti-NaN en AMBAS

    sample = Sample(N_MEDIUM)
    sample.add_mixture_layer(species, densities, 0.0, float("inf"))

    # --- CROSS-CHECK: mfp reportado por la mezcla vs mfp derivado ---
    mfp_layer = None
    if MixtureLayer is not None:
        try:
            layer = sample.layers[0]
            if isinstance(layer, MixtureLayer):
                mfp_layer = float(layer.mfp_total)
                print("---- Cross-check de la MixtureLayer ----")
                print(f"mu_s_total: {layer.mu_s_total:.6e}")
                print(f"mu_a_total: {layer.mu_a_total:.6e}")
                print(f"mfp_total (capa): {mfp_layer:.6f} um")
        except Exception as e:
            print(f"[warn] no pude leer MixtureLayer.mfp_total: {e}")

    print("---- Parametros de la simulacion (MEZCLA) ----")
    print(f"radio: {RADIUS:.3f} um   (2 especies identicas, n/2 c/u)")
    print(f"n_total: {n_total:.4e} 1/um^3   nd_each: {nd_each:.4e} 1/um^3")
    print(f"eficiencia de scattering: {dq['scattering_efficiency']:.3f}")
    print(f"mfp derivado (== homogeneo): {mean_free_path:.6f} um")
    if mfp_layer is not None:
        rel = abs(mfp_layer - mean_free_path) / mean_free_path
        flag = "OK" if rel < 1e-6 else "!! REVISAR"
        print(f"mfp mezcla vs derivado: dif rel = {rel:.2e}   [{flag}]")
    print(f"camino de transporte l*: {dq['transport_mean_free_path']:.4f} um")
    print(f"factor de anisotropia g: {dq['anisotropy_g']:.4f}")
    print(f"theta_coherent: {theta_coherent * 1e3:.4f} mrad")

    laser = Laser(
        LASER_M, LASER_N, WAVELENGTH,
        LASER_RADIUS_MFP * mean_free_path, LASER_TYPE,
    )

    # --- grilla angular (misma receta y mismos numeros que la homogenea) ---
    theta_max = THETA_CONE_FACTOR * theta_coherent
    d_theta = theta_max / N_THETA
    d_phi = PHI_MAX / N_PHI
    print(f"theta_max (K={THETA_CONE_FACTOR:g}): {theta_max * 1e3:.4f} mrad")
    print(f"d_theta: {d_theta * 1e3:.5f} mrad")

    # --- sensores ---
    sens = SensorsGroup()
    det = sens.add_detector(
        FarFieldCBSSensor(theta_max, PHI_MAX, T_MAX, d_theta, d_phi, D_TIME, True)
    )
    det.set_theta_limit(0, theta_max)

    det_analog = None
    if RUN_ANALOG_CHECK:
        det_analog = sens.add_detector(
            FarFieldCBSSensor(theta_max, PHI_MAX, T_MAX, d_theta, d_phi, D_TIME, False)
        )
        det_analog.set_theta_limit(0, theta_max)

    stats = sens.add_detector(StatisticsSensor(z=0, absorb=True))
    stats.set_theta_limit(0, theta_max)

    # --- config ---
    config = SimConfig()
    config.n_photons = N_PHOTONS
    config.sample = sample
    config.detector = sens
    config.laser = laser
    config.track_reverse_paths = True          # <-- imprescindible para CBS
    config.pin_threads_to_cores = False
    config.n_threads = N_THREADS
    config.seed = SEED
    config.show_progress = True

    # --- params: derivados de mezcla + info de grilla ---
    # Se hace spread SEGURO de derived_quantities_mixture (sin indexar sus claves)
    # y luego se FUERZAN las claves canonicas que fig_validation.py necesita.
    dq_mix = {}
    try:
        dq_mix = dict(derived_quantities_mixture(species, densities))
        print("derived_quantities_mixture:", dq_mix)
    except Exception as e:
        print(f"[warn] derived_quantities_mixture fallo: {e}")

    extra = {
        **dq_mix,
        "radius": RADIUS,
        "volume_fraction": VOLUME_FRACTION,
        "number_densities": densities,
        "mean_free_path": mean_free_path,                       # canonico
        "transport_mean_free_path": dq["transport_mean_free_path"],
        "theta_coherent": theta_coherent,
        "mfp_total_layer": mfp_layer,
        "theta_max": theta_max,
        "d_theta": d_theta,
        "d_phi": d_phi,
        "n_theta": N_THETA,
        "n_phi": N_PHI,
        "theta_cone_factor": THETA_CONE_FACTOR,
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
    run_simulation_parallel(config)clear
    print("runtime_s:", time.time() - t0)
    print("hits:", det.hits)

    # --- guardar RAW + procesados ---
    sensors_to_save = {"farfield_cbs": det, "statistics": stats}
    if det_analog is not None:
        sensors_to_save["farfield_cbs_analog"] = det_analog
    exp.save_sensors(sensors_to_save)

    cbs = postprocess_farfield_cbs(det, N_PHOTONS)
    exp.save_processed("farfield_cbs", cbs, sensor=det)

    if det_analog is not None:
        cbs_analog = postprocess_farfield_cbs(det_analog, N_PHOTONS)
        exp.save_processed("farfield_cbs_analog", cbs_analog, sensor=det_analog)

    # keep-alive explicito: referenciar species/sample al final del scope asegura
    # que sobreviven a toda la corrida (el binding guarda punteros crudos).
    _keep_alive = (species, ref, sample)
    del _keep_alive


# ===========================================================================
# Una sola corrida (sweep de un elemento -> load_sweep lo lee igual)
# ===========================================================================
run_name = f"mixture_r_{RADIUS:.3f}"
print(f"Running MIXTURE CBS validation for radius={RADIUS:.3f} (2 identical species)")
sweep.run(0, run_name, lambda exp: run_mixture_simulation(exp))