import __main__
import time
import numpy as np
from datetime import datetime

from luminis_mc import (
    SweepManager, ProgressMonitor, on_progress,
    Laser, RGDMedium, Sample, FarFieldCBSSensor, StatisticsSensor, SensorsGroup, SimConfig, RayleighDebyeEMCPhaseFunction,
    run_simulation_parallel, postprocess_farfield_cbs,
    set_log_level, LogLevel, LaserSource
)



params_sweep = [
    {
        "radius": 0.070 / 2,
        "mean_free_path": 121.3,
        "mu_scattering": 1.0 / 121.3,
    },
    {
        "radius": 0.110 / 2,
        "mean_free_path": 34.6,
        "mu_scattering": 1.0 / 34.6,
    },
    {
        "radius": 0.350 / 2,
        "mean_free_path": 4.9,
        "mu_scattering": 1.0 / 4.9,
    }
]

n_particle = 1.59
n_medium = 1.33
mu_absortion = 0.0
wavelength = 0.514

# Laser parameters
laser_m_polarization_state = 1
laser_n_polarization_state = 0
laser_radius = 0.0
laser_type = LaserSource.Point

# Phase function parameters
phasef_theta_min = 0.0
phasef_theta_max = np.pi
phasef_ndiv = 100_000


volume_fraction = 0.1



for params in params_sweep:
    radius = params["radius"]
    mean_free_path = params["mean_free_path"]
    mu_scattering = params["mu_scattering"]

    print(f"Running simulation for radius={radius}, mean_free_path={mean_free_path}, mu_scattering={mu_scattering}")

    phase = RayleighDebyeEMCPhaseFunction(wavelength, radius, n_particle, n_medium, phasef_ndiv, phasef_theta_min, phasef_theta_max)
    medium = RGDMedium(mu_absortion, mu_scattering, phase, mean_free_path, radius, n_particle, n_medium, wavelength)

    anysotropy = phase.get_anisotropy_factor()
    print(f"Anisotropy factor g: {anysotropy[0]}")

    scattering_efficiency = medium.scattering_efficiency()
    print(f"Scattering efficiency Q_sca: {scattering_efficiency}")

    l_s_nm = (4.0 * radius) / (3.0 * volume_fraction * scattering_efficiency)
    print(f"Calculated mean free path: {l_s_nm}")
    
