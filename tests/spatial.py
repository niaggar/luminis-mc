# %%

from luminis_mc import (
    Detector,
)
from luminis_mc import LogLevel
from luminis_mc import set_log_level
import matplotlib.pyplot as plt
import numpy as np

set_log_level(LogLevel.debug)

detector = Detector(0)
detector.load_recorded_photons("speckle_photons.dat")

spatial_intensity = detector.compute_spatial_intensity(x_len=20, y_len=20, max_theta=np.pi/12, n_x=1125, n_y=1125)
I_total_array = np.array(spatial_intensity.I_total, copy=False)
I_x_array = np.array(spatial_intensity.Ix, copy=False)
I_y_array = np.array(spatial_intensity.Iy, copy=False)
I_z_array = np.array(spatial_intensity.Iz, copy=False)

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

study_spatial_intensity(I_total_array, "Total Intensity")
study_spatial_intensity(I_x_array, "X Polarization Intensity")
study_spatial_intensity(I_y_array, "Y Polarization Intensity")
study_spatial_intensity(I_z_array, "Z Polarization Intensity")