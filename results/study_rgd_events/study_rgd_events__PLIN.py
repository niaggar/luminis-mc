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
EXP_NAME = "study_rgd_events__PLIN__beam2500"
BASE_DIR = "/Users/niaggar/Tests"

sweep = SweepManager(EXP_NAME, BASE_DIR, timestamped=False)
sweep.snapshot_master_script(__main__.__file__)
sweep.log_readme(
    "CBS homogeneous radius sweep -- single RGD species per run, polarizacion "
    "LINEAL (m=1, n=0), estimador next-event. Radios [0.020, 0.035, 0.055, "
    "0.075, 0.175] um, f=0.10, n_p=1.59, n_med=1.33, lambda=0.514 um, sin "
    "absorcion. Haz gaussiano FIJO w=2500 um (regimen de onda plana, w/l* de "
    "~4.5 a ~110 segun radio). Doble ventana angular en unidades reducidas "
    "q=k*l*_teorico*theta: fina q in [0, 7.5] (500 bins, cono), cola q in "
    "[7.5, 40] (200 bins, fondo/baseline), solape en [0.9, 1.0]*theta_1 para "
    "chequeo de stitching. Sensores integrados en tiempo (t_max=0). "
)

# ===========================================================================
# Parametros fisicos
# ===========================================================================
radius_values = [0.020, 0.035, 0.055, 0.075, 0.100, 0.175]
events_study = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50, 100, 200, 500, 1000]


VOLUME_FRACTION = 0.10
N_PARTICLE = 1.59
N_MEDIUM = 1.33
WAVELENGTH = 0.514             # um
MU_A_PERCENT = 0.0             # sin absorcion (gate de reciprocidad limpio)

# Laser: polarizacion LINEAL a lo largo de m (X), incidencia normal.
LASER_M = 1.0
LASER_N = 0.0
LASER_RADIUS = 2500          # um
LASER_TYPE = LaserSource.Gaussian

# Funcion de fase
PHASEF_THETA_MIN = 0.0
PHASEF_THETA_MAX = np.pi
PHASEF_NDIV = 10_000

# ---------------------------------------------------------------------------
# Grilla angular
# ---------------------------------------------------------------------------
N_PHI = 36
PHI_MAX = 2 * np.pi

# ---------------------------------------------------------------------------
# Muestreo
# ---------------------------------------------------------------------------
N_THREADS = 46
N_PHOTONS = 100_000
N_THETA_1 = 500
N_THETA_2 = 200
Q_FINE, Q_TAIL = 7.5, 40.0
k = 2 * np.pi * N_MEDIUM / WAVELENGTH

SEED_BASE = 20260710   # fija una vez por tanda; cualquier entero documentado

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
    """Fija un albedo bien definido para que la agregacion mu_a^(i) no caiga en 0/0.

    La escala absoluta de mu_s aqui NO afecta el transporte (la mu_s total la fija
    n_i sigma_i); solo importa el cociente mu_a/mu_s. Absorcion primero, luego
    scattering (recomputa mu_t).
    """
    mu_s = 1.0 - mu_a_percent
    mu_a = mu_a_percent
    medium.set_absorption_coefficient(mu_a)
    medium.set_scattering_coefficient(mu_s)
    medium.set_mean_free_path(mfp)



# ===========================================================================
# Corrida
# ===========================================================================
def run_cbs(exp, radius, rad_index, rep):
    especie = build_species(radius)
    dq = derived_quantities(especie, VOLUME_FRACTION)
    set_albedo(especie, MU_A_PERCENT, dq['mean_free_path'])

    time_grid = build_time_grid(dq['transport_mean_free_path'], N_MEDIUM, n_bins=25, t_max_taustar=30, binning="geometric")

    sample = Sample(N_MEDIUM)
    sample.add_layer(especie, 0, float('inf'))

    laser = Laser(LASER_M, LASER_N, WAVELENGTH, LASER_RADIUS, LASER_TYPE)


    theta_1 = Q_FINE / (k * dq['transport_mean_free_path'])
    theta_2 = Q_TAIL / (k * dq['transport_mean_free_path'])
    d_theta_1 = theta_1 / N_THETA_1
    d_theta_2 = theta_2 / N_THETA_2
    d_phi = PHI_MAX / N_PHI
    t_max = 0
    dt = 0

    sens = SensorsGroup()
    detectors = {}

    for ev in events_study:
        det_1 = sens.add_detector(FarFieldCBSSensor(theta_1, PHI_MAX, t_max, d_theta_1, d_phi, dt, True))
        det_1.set_theta_limit(0, theta_1)
        det_1.set_phi_slices([0, np.pi/4, np.pi/2])
        det_1.set_events_limit(ev, ev)

        det_2 = sens.add_detector(FarFieldCBSSensor(theta_2, PHI_MAX, t_max, d_theta_2, d_phi, dt, True))
        det_2.set_theta_limit(theta_1 * 0.9, theta_2)
        det_2.set_phi_slices([0, np.pi/4, np.pi/2])
        det_2.set_events_limit(ev, ev)

        detectors[f"farfield_cbs_1_events_{ev}"] = det_1
        detectors[f"farfield_cbs_2_events_{ev}"] = det_2


    det_sum010_1 = sens.add_detector(FarFieldCBSSensor(theta_1, PHI_MAX, t_max, d_theta_1, d_phi, dt, True))
    det_sum010_1.set_theta_limit(0, theta_1)
    det_sum010_1.set_phi_slices([0, np.pi/4, np.pi/2])
    det_sum010_1.set_events_limit(0, 10)

    det_sum010_2 = sens.add_detector(FarFieldCBSSensor(theta_2, PHI_MAX, t_max, d_theta_2, d_phi, dt, True))
    det_sum010_2.set_theta_limit(theta_1 * 0.9, theta_2)
    det_sum010_2.set_phi_slices([0, np.pi/4, np.pi/2])
    det_sum010_2.set_events_limit(0, 10)


    det_sum1050_1 = sens.add_detector(FarFieldCBSSensor(theta_1, PHI_MAX, t_max, d_theta_1, d_phi, dt, True))
    det_sum1050_1.set_theta_limit(0, theta_1)
    det_sum1050_1.set_phi_slices([0, np.pi/4, np.pi/2])
    det_sum1050_1.set_events_limit(10, 50)

    det_sum1050_2 = sens.add_detector(FarFieldCBSSensor(theta_2, PHI_MAX, t_max, d_theta_2, d_phi, dt, True))
    det_sum1050_2.set_theta_limit(theta_1 * 0.9, theta_2)
    det_sum1050_2.set_phi_slices([0, np.pi/4, np.pi/2])
    det_sum1050_2.set_events_limit(10, 50)

    stats = sens.add_detector(StatisticsSensor(z=0, absorb=True))
    stats.set_theta_limit(0, theta_2)

    config = SimConfig()
    config.n_photons = N_PHOTONS
    config.sample = sample
    config.detector = sens
    config.laser = laser
    config.track_reverse_paths = True
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
        "t_max": t_max,
        "d_time": dt,
        "polarization": "linear",
        "seed": config.seed,
        "replica": rep,
        "laser_radius_um": LASER_RADIUS,
        "q_fine": Q_FINE,
        "q_tail": Q_TAIL,
    }
    exp.save_params(config, extra=extra)

    t0 = time.time()
    run_simulation_parallel(config)
    print("n_photons:", N_PHOTONS, "| runtime_s:", time.time() - t0)

    sensors_save = {
        **detectors,
        "farfield_cbs_1_sum010": det_sum010_1,
        "farfield_cbs_2_sum010": det_sum010_2,
        "farfield_cbs_1_sum1050": det_sum1050_1,
        "farfield_cbs_2_sum1050": det_sum1050_2,
        "statistics": stats}
    exp.save_sensors(sensors_save)
    for ev in events_study:
        exp.save_processed(f"farfield_cbs_1_events_{ev}", postprocess_farfield_cbs(detectors[f"farfield_cbs_1_events_{ev}"], N_PHOTONS), sensor=detectors[f"farfield_cbs_1_events_{ev}"])
        exp.save_processed(f"farfield_cbs_2_events_{ev}", postprocess_farfield_cbs(detectors[f"farfield_cbs_2_events_{ev}"], N_PHOTONS), sensor=detectors[f"farfield_cbs_2_events_{ev}"])
    exp.save_processed("farfield_cbs_1_sum010", postprocess_farfield_cbs(det_sum010_1, N_PHOTONS), sensor=det_sum010_1)
    exp.save_processed("farfield_cbs_2_sum010", postprocess_farfield_cbs(det_sum010_2, N_PHOTONS), sensor=det_sum010_2)
    exp.save_processed("farfield_cbs_1_sum1050", postprocess_farfield_cbs(det_sum1050_1, N_PHOTONS), sensor=det_sum1050_1)
    exp.save_processed("farfield_cbs_2_sum1050", postprocess_farfield_cbs(det_sum1050_2, N_PHOTONS), sensor=det_sum1050_2)

    _keep_alive = (sample, especie); del _keep_alive


N_REPLICAS = 5
for index, rad in enumerate(radius_values):
    for rep in range(N_REPLICAS):
        name = f"radius_{rad:.3f}__rep{rep}"
        sweep.run(index * N_REPLICAS + rep, name, lambda exp, rad=rad, i=index, rep=rep: run_cbs(exp, rad, i, rep))
