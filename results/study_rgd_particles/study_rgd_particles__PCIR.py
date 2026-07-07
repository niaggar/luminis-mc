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
EXP_NAME = "study_rgd_particles__PCIR"
BASE_DIR = "/home/niaggar/Developer/luminis-mc/temporal_results"

sweep = SweepManager(EXP_NAME, BASE_DIR, timestamped=False)
sweep.snapshot_master_script(__main__.__file__)
sweep.log_readme(
    "CBS validation -- MIXTURE of two IDENTICAL species (linear pol, estimator). "
    "Equivalence test against sim_homogeneous.py: mixture(n/2,n/2) == homogeneous(n)."
)

# ===========================================================================
# Parametros fisicos
# ===========================================================================
radius_values = [0.020, 0.035, 0.055, 0.075, 0.175]
VOLUME_FRACTION = 0.10
N_PARTICLE = 1.59
N_MEDIUM = 1.33
WAVELENGTH = 0.514             # um
MU_A_PERCENT = 0.0             # sin absorcion (gate de reciprocidad limpio)

# Laser: polarizacion CIRCULAR
LASER_M = 1 / np.sqrt(2)
LASER_N = -1j / np.sqrt(2)
LASER_RADIUS_MFP = 4.0         # en unidades de l_s
LASER_TYPE = LaserSource.Gaussian

# Funcion de fase
PHASEF_THETA_MIN = 0.0
PHASEF_THETA_MAX = np.pi
PHASEF_NDIV = 10_000

# ---------------------------------------------------------------------------
# Grilla angular
# ---------------------------------------------------------------------------
N_THETA = 1000
N_PHI = 36
PHI_MAX = 2 * np.pi
THETA_MAX = np.deg2rad(1)

# ---------------------------------------------------------------------------
# Muestreo
# ---------------------------------------------------------------------------
N_THREADS = 46
N_PHOTONS = 50_000

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
def run_cbs(exp, radius):
    especie = build_species(radius)
    dq = derived_quantities(especie, VOLUME_FRACTION)
    set_albedo(especie, MU_A_PERCENT, dq['mean_free_path'])

    time_grid = build_time_grid(dq['transport_mean_free_path'], N_MEDIUM, n_bins=25, t_max_taustar=30, binning="geometric")
    print(time_grid)

    sample = Sample(N_MEDIUM)
    sample.add_layer(especie, 0, float('inf'))

    laser = Laser(LASER_M, LASER_N, WAVELENGTH, LASER_RADIUS_MFP * dq['transport_mean_free_path'], LASER_TYPE)

    d_theta = THETA_MAX / N_THETA
    d_phi = PHI_MAX / N_PHI
    t_max = 0
    dt = 0

    sens = SensorsGroup()
    det = sens.add_detector(
        FarFieldCBSSensor(THETA_MAX, PHI_MAX, t_max, d_theta, d_phi, dt, True)
    )
    det.set_theta_limit(0, THETA_MAX)
    stats = sens.add_detector(StatisticsSensor(z=0, absorb=True))
    stats.set_theta_limit(0, THETA_MAX)

    config = SimConfig()
    config.n_photons = N_PHOTONS
    config.sample = sample
    config.detector = sens
    config.laser = laser
    config.track_reverse_paths = True
    config.pin_threads_to_cores = False
    config.n_threads = N_THREADS
    config.show_progress = True

    extra = {
        **dq,
        "theta_max": THETA_MAX,
        "d_theta": d_theta,
        "d_phi": d_phi,
        "n_theta": N_THETA,
        "n_phi": N_PHI,
        "t_max": t_max,
        "d_time": dt,
        "polarization": "linear",
    }
    exp.save_params(config, extra=extra)

    t0 = time.time()
    run_simulation_parallel(config)
    print("n_photons:", N_PHOTONS, "| runtime_s:", time.time() - t0, "| hits:", det.hits)

    exp.save_sensors({"farfield_cbs": det, "statistics": stats})
    exp.save_processed("farfield_cbs", postprocess_farfield_cbs(det, N_PHOTONS), sensor=det)

    _keep_alive = (sample, especie); del _keep_alive


for index, rad in enumerate(radius_values):
    name = f"radius_{rad:.2f}"
    sweep.run(index, name, lambda exp, rad=rad: run_cbs(exp, rad))


