# %%

from luminis_mc import (
    Detector,
)
from luminis_mc import LogLevel
from luminis_mc import set_log_level
import matplotlib.pyplot as plt
import numpy as np

set_log_level(LogLevel.debug)

mean_free_path = 2.8
light_speed = 299792458e-6
t_ref = mean_free_path / light_speed
dt = 1 * t_ref
max_time = 10 * t_ref

detector = Detector(0)
detector.load_recorded_photons("test-data-phi-conditional.dat")

spatial_intensity = detector.compute_time_resolved_spatial_intensity(
    x_len=10.0,
    y_len=10.0,
    max_theta=np.pi / 2,
    t_max=max_time,
    dt=dt,
    n_x=500,
    n_y=500,
)

def study_spatial_intensity(I_array, title):
    plt.figure(figsize=(8, 8))
    plt.imshow(I_array, cmap='inferno', origin='lower')
    plt.colorbar(label="Intensity")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.imshow(np.log(I_array + 1e-20), cmap='inferno', origin='lower')
    plt.colorbar(label="Log Intensity")
    plt.title(f"Logarithmic {title}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

for it, time_bin in enumerate(spatial_intensity):
    I_total_array = np.array(time_bin.Icros, copy=False)
    study_spatial_intensity(I_total_array, f"Total Intensity at time bin {it}")