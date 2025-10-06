# %%

from luminis_mc import (
    Laser,
    SimpleMedium,
    Detector,
    SimConfig,
    HenyeyGreensteinPhaseFunction,
    Absorption,
)
from luminis_mc import LogLevel, LaserSource
from luminis_mc import run_simulation, set_log_level
import matplotlib.pyplot as plt
import numpy as np

# set_log_level(LogLevel.Info)

# %%

mu_absortion = 0.07
mu_scattering = 0.05
mean_free_path = 1 / (mu_absortion + mu_scattering)
radius = 0.1 * mean_free_path

print(f"Mean free path: {mean_free_path}")
print(f"Medium radius: {radius}")

n_photons = 4000000
laser_source = Laser(
    [0, 0, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [1, 0],
    532.0,
    mean_free_path * 1,
    LaserSource.Gaussian,
)
detector = Detector(origin=(0, 0, 0), normal=(0, 0, 1))
phase_function = HenyeyGreensteinPhaseFunction(0.999)
medium = SimpleMedium(
    mu_absortion, mu_scattering, phase_function, mean_free_path, radius
)
absorption = Absorption(mean_free_path * 20, mean_free_path * 40, 0.5, 0.5)
medium.absorption = absorption
config = SimConfig(n_photons=n_photons)

# %%

run_simulation(config, medium, detector, laser_source)

# %%
print(len(detector.recorded_photons))
# emited_positions = []
# for i in range(len(detector.recorded_photons)):
#     photon = detector.recorded_photons[i]
#     emited_positions.append((photon.pos[0], photon.pos[1], photon.pos[2]))


# emited_positions = np.array(emited_positions)
# plt.figure(figsize=(8, 8))
# plt.scatter(emited_positions[:, 0], emited_positions[:, 1], s=1, alpha=0.5)
# # plt.xlim(-30, 30)
# # plt.ylim(-30, 30)
# plt.xlabel("X position")
# plt.ylabel("Y position")
# plt.title("Photon Emission Positions")
# plt.grid(True)
# plt.show()


absortion_image = medium.absorption.get_absorption_image(n_photons)
plt.figure(figsize=(8, 8))
plt.imshow(absortion_image, cmap="viridis", origin="lower")
plt.colorbar(label="Absorption Values")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Absorption Values Grid")
plt.show()
