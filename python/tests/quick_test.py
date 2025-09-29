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

laser_source = Laser([0, 0, 0], [0, 0, 1], [1, 0], 532.0, 1.0, LaserSource.Gaussian)
detector = Detector(origin=(0, 0, 0), normal=(0, 0, 1))
phase_function = HenyeyGreensteinPhaseFunction(g=0.9)
medium = SimpleMedium(0.01, 0.1, phase_function, 1.33, 0.1)
config = SimConfig(seed=42, n_photons=1000)

# %%

run_simulation(config, medium, detector, laser_source)
