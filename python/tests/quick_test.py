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
import matplotlib.pyplot as plt
import numpy as np

set_log_level(LogLevel.debug)

# %%

n_photons = 10000
laser_source = Laser([0, 0, 0], [0, 0, 1], [1, 0], 532.0, 1.0, LaserSource.Gaussian)
detector = Detector(origin=(0, 0, 0), normal=(0, 0, 1))
phase_function = HenyeyGreensteinPhaseFunction(g=0.9)
medium = SimpleMedium(0.01, 0.1, phase_function, 1.33, 0.1)
config = SimConfig(seed=42, n_photons=n_photons)

# %%

run_simulation(config, medium, detector, laser_source)

# %%
print(len(detector.recorded_photons))
emited_positions = []
for i in range(len(detector.recorded_photons)):
    photon = detector.recorded_photons[i]
    emited_positions.append((photon.pos[0], photon.pos[1]))

plt.figure(figsize=(10, 6))
plt.plot(
    np.array(emited_positions)[:, 0],
    np.array(emited_positions)[:, 1],
    "o",
    markersize=1,
    alpha=0.5,
)
plt.title("Photon Emission Positions")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.axis("equal")
plt.grid(True)
plt.show()
