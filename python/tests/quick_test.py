# %%

from luminis_mc import (
    Laser,
    SimpleMedium,
    Detector,
    SimConfig,
    HenyeyGreensteinPhaseFunction,
)
from luminis_mc import LogLevel, LaserSource
from luminis_mc import run_simulation, set_log_level
import numpy as np

set_log_level(LogLevel.debug)

# %%

laser_source = Laser([0, 0, 0], [0, 0, 1], [1, 0], 1.0, 5.0, LaserSource.Gaussian)
detector = Detector(origin=(0, 0, 0), normal=(0, 0, 1))
phase_function = HenyeyGreensteinPhaseFunction(g=0.9)
medium = SimpleMedium(
    scattering=0.1, absorption=0.01, phase_func=phase_function, mfp=1, radius=10
)
config = SimConfig(seed=42, n_photons=10)

# %%

run_simulation(config, medium, detector, laser_source)
