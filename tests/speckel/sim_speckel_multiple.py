"""
Speckle Parameter Sweep Study
==============================
Two orthogonal sweeps to characterize speckle simulation quality and runtime:

  Sweep A: Fix radius=0.5 µm (g≈0.85), vary n_photons
           → How does quality and runtime scale with photon count?

  Sweep B: Fix n_photons=1_000_000, vary radius_real (→ vary anisotropy g)
           → How does quality and runtime change with scattering regime?

Quality metric: speckle contrast  C = std(I_inst) / mean(I_avg)
                should → 1.0 for a fully developed speckle pattern.
                Also saves KL divergence from the theoretical exponential.
"""

import __main__
import time
import numpy as np
from scipy.stats import entropy

from luminis_mc import (
    SweepManager,
    Laser, MieMedium, Sample,
    PlanarFluenceSensor, PlanarFieldSensor, StatisticsSensor, SensorsGroup,
    SimConfig, MiePhaseFunction,
    run_simulation_parallel,
    set_log_level, LogLevel, LaserSource
)

set_log_level(LogLevel.info)

base_dir = "/Users/niaggar/Documents/Thesis/Progress/02Mar26"

# ─────────────────────────────────────────────────────────────────────────────
# Fixed physical parameters (shared across both sweeps)
# ─────────────────────────────────────────────────────────────────────────────
mean_free_path_sim  = 1.0
mean_free_path_real = 2.8
n_particle_real     = 1.58984
n_medium_real       = 1.33
wavelength_real     = 0.52       # µm

mu_absortion_sim    = 0.0
mu_scattering_sim   = 1.0 / mean_free_path_sim - mu_absortion_sim

# Circularly polarized laser (same as speckel.py reference)
laser_m_polarization_state = 1 / np.sqrt(2)
laser_n_polarization_state = -1j / np.sqrt(2)
laser_radius = 5 * mean_free_path_sim
laser_type   = LaserSource.Gaussian

# Phase-function discretization
phasef_ndiv      = 1000
phasef_theta_min = 0.0
phasef_theta_max = np.pi

# Sensor geometry
sensor_z     = 0
sensor_len   = 40 * mean_free_path_sim   # same for x and y
sensor_dx    = 0.1 * mean_free_path_sim
sensor_absorb   = True
sensor_estimate = False


# ─────────────────────────────────────────────────────────────────────────────
# Quality-metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def speckle_contrast(I_inst, I_avg):
    """
    C = std(I_inst) / mean(I_avg)  over pixels where I_avg > 1% of peak.
    Should converge to 1 for a fully developed speckle pattern.
    """
    mask = I_avg > 0.01 * np.max(I_avg)
    return float(np.std(I_inst[mask]) / np.mean(I_avg[mask]))


def kl_from_exponential(I_inst, I_avg, n_bins=80):
    """
    KL divergence  D_KL( p_sim || p_theory )  where p_theory = e^{-η}.
    Smaller → better match to theory.
    η = I_inst / I_avg, restricted to illuminated region.
    """
    mask = I_avg > 0.01 * np.max(I_avg)
    eta = I_inst[mask] / I_avg[mask]

    # Histogram over [0, 8] (captures ~99.97% of exponential distribution)
    hist, edges = np.histogram(eta, bins=n_bins, range=(0, 8), density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    bin_w   = edges[1] - edges[0]

    p_sim    = hist * bin_w                    # empirical probabilities
    p_theory = np.exp(-centers) * bin_w        # theoretical exponential

    # Clip to avoid log(0)
    p_sim    = np.clip(p_sim,    1e-10, None)
    p_theory = np.clip(p_theory, 1e-10, None)

    return float(entropy(p_sim, p_theory))     # scipy: sum p log(p/q)


# ─────────────────────────────────────────────────────────────────────────────
# Core simulation function
# ─────────────────────────────────────────────────────────────────────────────

def run_speckle(exp, radius_real, n_photons):
    """
    Build and run one speckle simulation; save raw sensor data + quality metrics.
    Returns (runtime_s, contrast_x, kl_x, anisotropy_g).
    """
    laser  = Laser(laser_m_polarization_state, laser_n_polarization_state,
                   wavelength_real, laser_radius, laser_type)
    phase  = MiePhaseFunction(wavelength_real, radius_real,
                              n_particle_real, n_medium_real,
                              phasef_ndiv, phasef_theta_min, phasef_theta_max)
    medium = MieMedium(mu_absortion_sim, mu_scattering_sim, phase,
                       mean_free_path_sim, radius_real,
                       n_particle_real, n_medium_real, wavelength_real)
    sample = Sample(n_medium=n_medium_real)
    sample.add_layer(medium, 0.0, float('inf'))

    anisotropy_g = float(phase.get_anisotropy_factor()[0])
    print(f"  g = {anisotropy_g:.4f}   n_photons = {n_photons:,}")

    # Sensors
    sens = SensorsGroup()
    pfield   = sens.add_detector(PlanarFieldSensor(
        sensor_z, sensor_len, sensor_len, sensor_dx, sensor_dx,
        sensor_absorb, sensor_estimate))
    pfluence = sens.add_detector(PlanarFluenceSensor(
        sensor_z, sensor_len, sensor_len, 0.0,
        sensor_dx, sensor_dx, 0.0,
        sensor_absorb, sensor_estimate))
    stats = sens.add_detector(StatisticsSensor(sensor_z, sensor_absorb))

    stats.set_events_histogram_bins(500)
    stats.set_position_limit(-sensor_len/2, sensor_len/2,
                             -sensor_len/2, sensor_len/2)

    config = SimConfig(n_photons=n_photons, sample=sample,
                       detector=sens, laser=laser, track_reverse_paths=True)
    config.n_threads = 8

    # Log parameters
    exp.log_params(
        radius_real=radius_real,
        anisotropy_g=anisotropy_g,
        n_photons=n_photons,
        mean_free_path_sim=mean_free_path_sim,
        mean_free_path_real=mean_free_path_real,
        n_particle_real=n_particle_real,
        n_medium_real=n_medium_real,
        mu_absortion_sim=mu_absortion_sim,
        mu_scattering_sim=mu_scattering_sim,
        wavelength_real=wavelength_real,
        laser_radius=laser_radius,
        laser_type=laser_type,
        sensor_len=sensor_len,
        sensor_dx=sensor_dx,
    )

    # Run
    t0 = time.time()
    run_simulation_parallel(config)
    runtime_s = time.time() - t0
    print(f"  runtime: {runtime_s:.1f} s")

    # Save raw sensor data
    exp.save_sensor(pfield,   "planarfield")
    exp.save_sensor(pfluence, "planarfluence")
    exp.save_sensor(stats,    "statistics")

    # ── Inline quality metrics (no need to reload from disk) ─────────────────
    # Retrieve Stokes from fluence sensor and E-fields from field sensor
    S0_t = np.asarray(pfluence.S0_t)[0]   # time-averaged total intensity
    S1_t = np.asarray(pfluence.S1_t)[0]
    Ex   = np.asarray(pfield.Ex)
    Ey   = np.asarray(pfield.Ey)

    I_inst_x = np.abs(Ex)**2
    I_avg_x  = (S0_t + S1_t) / 2.0

    I_inst_y = np.abs(Ey)**2
    I_avg_y  = (S0_t - S1_t) / 2.0

    contrast_x = speckle_contrast(I_inst_x, I_avg_x)
    contrast_y = speckle_contrast(I_inst_y, I_avg_y)
    kl_x       = kl_from_exponential(I_inst_x, I_avg_x)
    kl_y       = kl_from_exponential(I_inst_y, I_avg_y)

    print(f"  contrast_x={contrast_x:.4f}  contrast_y={contrast_y:.4f}")
    print(f"  KL_x={kl_x:.4f}  KL_y={kl_y:.4f}")

    # Save scalar quality metrics as derived data so they appear in the sweep table
    exp.save_derived("quality/runtime_s",   np.array([runtime_s]))
    exp.save_derived("quality/anisotropy_g", np.array([anisotropy_g]))
    exp.save_derived("quality/contrast_x",  np.array([contrast_x]))
    exp.save_derived("quality/contrast_y",  np.array([contrast_y]))
    exp.save_derived("quality/kl_x",        np.array([kl_x]))
    exp.save_derived("quality/kl_y",        np.array([kl_y]))

    return runtime_s, contrast_x, kl_x, anisotropy_g


# ─────────────────────────────────────────────────────────────────────────────
# Sweep A  —  vary n_photons, fixed radius
# ─────────────────────────────────────────────────────────────────────────────
FIXED_RADIUS_FOR_A = 0.5     # µm  → g ≈ 0.85 (mid forward-scattering regime)

n_photons_sweep = [
    10_000,
    50_000,
    100_000,
    500_000,
    1_000_000,
    5_000_000,
    10_000_000,
    20_000_000,
    50_000_000,
    100_000_000,
    500_000_000,
    1_000_000_000,
]

sweep_A = SweepManager("speckle_sweep_A_nphotons", base_dir, timestamped=True)
sweep_A.snapshot_master_script(__main__.__file__)
sweep_A.log_readme(
    f"Speckle quality vs n_photons — fixed radius={FIXED_RADIUS_FOR_A} µm"
)

for i, n in enumerate(n_photons_sweep):
    run_name = f"n_{n}"
    sweep_A.run(i, run_name,
                lambda exp, n=n: run_speckle(exp, FIXED_RADIUS_FOR_A, n))


# ─────────────────────────────────────────────────────────────────────────────
# Sweep B  —  vary radius (→ anisotropy g), fixed n_photons
# ─────────────────────────────────────────────────────────────────────────────
FIXED_N_FOR_B = 500_000_000

radius_sweep = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]   # µm

sweep_B = SweepManager("speckle_sweep_B_anisotropy", base_dir, timestamped=True)
sweep_B.snapshot_master_script(__main__.__file__)
sweep_B.log_readme(
    f"Speckle quality vs anisotropy g — fixed n_photons={FIXED_N_FOR_B:,}"
)

for i, r in enumerate(radius_sweep):
    run_name = f"radius_{r:.3f}"
    sweep_B.run(i, run_name, lambda exp, r=r: run_speckle(exp, r, FIXED_N_FOR_B))