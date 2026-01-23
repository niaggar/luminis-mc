# %%

from luminis_mc import (
    Detector,
    compute_speckle,
    load_recorded_photons,
)
from luminis_mc import LogLevel
from luminis_mc import set_log_level
import matplotlib.pyplot as plt
import numpy as np

set_log_level(LogLevel.debug)

detector = Detector(0)
load_recorded_photons("test-data-phi-random.dat", detector)

speckle = compute_speckle(detector, 1125, 360)
I_total_array = np.array(speckle.I_total, copy=False)
I_x_array = np.array(speckle.Ix, copy=False)
I_y_array = np.array(speckle.Iy, copy=False)
I_z_array = np.array(speckle.Iz, copy=False)

def study_speckle(I_array, title):
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

    mean_I_total = np.mean(I_array)
    print(f"Mean intensity <I> = {mean_I_total:.6e}")
    if mean_I_total == 0:
        print("Mean intensity is zero, skipping histogram.")
        return
    
    eta_total = I_array / mean_I_total
    bins = np.linspace(0, 10, 60)
    hist_total, edges_total = np.histogram(eta_total, bins=bins, density=True)
    centers_total = 0.5 * (edges_total[1:] + edges_total[:-1])

    eta_theory = np.linspace(0, 10, 500)
    p_theory = np.exp(-eta_theory)
    C_total = np.std(I_array) / np.mean(I_array)
    print(f"Speckle contrast C (total) = {C_total:.3f}")

    plt.figure(figsize=(6, 5))
    plt.semilogy(centers_total, hist_total, 'o', label='Speckle Histogram')
    plt.semilogy(eta_theory, p_theory, '-', label=r'$e^{-\eta}$')
    plt.xlabel(r'$\eta = I / \langle I \rangle$')
    plt.ylabel('Probability Density (log scale)')
    plt.title(f'Speckle Intensity Distribution - {title}')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.show()

study_speckle(I_total_array, "Total Intensity")
study_speckle(I_x_array, "X Polarization Intensity")
study_speckle(I_y_array, "Y Polarization Intensity")
study_speckle(I_z_array, "Z Polarization Intensity")