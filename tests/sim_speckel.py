from luminis_mc import (
    Laser,
    SimpleMedium,
    MultiDetector,
    AngleDetector,
    HistogramDetector,
    ThetaHistogramDetector,
    SimConfig,
    RayleighDebyeEMCPhaseFunction,
    CVec2,
    Vec3,
    compute_speckle_angledetector,
    Rng,
    save_angle_detector_fields,
    load_angle_detector_fields,
    make_theta_condition,
)
from luminis_mc import LogLevel, LaserSource
from luminis_mc import run_simulation_parallel, set_log_level
import numpy as np
import matplotlib.pyplot as plt
import time

set_log_level(LogLevel.info)


start_time = time.time()

# Global frame of reference
m_global = Vec3(1, 0, 0)
n_global = Vec3(0, 1, 0)
s_global = Vec3(0, 0, 1)
light_speed = 1

# Medium parameters in micrometers
radius = 0.46
mean_free_path = 2.8
wavelength = 0.525
inv_mfp = 1 / mean_free_path
mu_absortion = 0.0003 * inv_mfp
mu_scattering = inv_mfp - mu_absortion
n_particle = 1.59
n_medium = 1.33

size_parameter = 2 * np.pi * radius / wavelength
condition_1 = np.abs(n_particle / n_medium - 1)
condition_2 = size_parameter * condition_1


# Time parameters
t_ref = mean_free_path / light_speed
dt = 0
max_time = 50 * t_ref

# Phase function parameters
thetaMin = 0.00001
thetaMax = np.pi
nDiv = 1000
n_photons = 10_000_000

# Laser parameters
origin = Vec3(0, 0, 0)
polarization = CVec2(1, 0)
laser_radius = 0.1 * mean_free_path
laser_type = LaserSource.Gaussian


rng_test = Rng()
laser_source = Laser(origin, polarization, wavelength, laser_radius, laser_type)
phase_function = RayleighDebyeEMCPhaseFunction(wavelength, radius, n_particle, n_medium, nDiv, thetaMin, thetaMax)
medium = SimpleMedium(mu_absortion, mu_scattering, phase_function, mean_free_path, radius, n_particle, n_medium)
anysotropy = phase_function.get_anisotropy_factor(rng_test)
print(f"Transport mean free path: {1 / (1 - anysotropy[0])}")

detectors_container = MultiDetector()
speckle_detector = detectors_container.add_detector(AngleDetector(0, 1125, 360))
histogram_detector_1 = detectors_container.add_detector(HistogramDetector(0, 500))
histogram_detector_2 = detectors_container.add_detector(HistogramDetector(0, 500))
histogram_detector_3 = detectors_container.add_detector(HistogramDetector(0, 500))
histogram_detector_4 = detectors_container.add_detector(HistogramDetector(0, 500))
histogram_detector_5 = detectors_container.add_detector(HistogramDetector(0, 500))
histogram_detector_6 = detectors_container.add_detector(HistogramDetector(0, 500))
theta_histogram_detector = detectors_container.add_detector(ThetaHistogramDetector(0, 500))

histogram_detector_1.set_theta_limit(0, np.pi)      # 180 degrees
histogram_detector_2.set_theta_limit(0, np.pi/2)    # 90 degrees
histogram_detector_3.set_theta_limit(0, np.pi/4)    # 45 degrees
histogram_detector_4.set_theta_limit(0, np.pi/8)    # 22.5 degrees
histogram_detector_5.set_theta_limit(0, 0.2)        # ~11.5 degrees
histogram_detector_6.set_theta_limit(0, 0.1)        # ~5.7 degrees



print("CONDITIONS FOR RAYLEIGH-DEBYE APPROXIMATION:")
print(f"Size parameter: {size_parameter}")
print(f"|n_particle / n_medium - 1|: {condition_1}")
print(f"Size parameter * |n_particle / n_medium - 1|: {condition_2}")

print("MEDIUM PARAMETERS:")
print(f"Mean free path: {mean_free_path}")
print(f"Particle radius: {radius}")
print(f"Wavelength: {wavelength}")
print(f"Scattering coefficient: {mu_scattering}")
print(f"Absorption coefficient: {mu_absortion}")
print(f"Albedo: {mu_scattering / (mu_scattering + mu_absortion)}")
print(f"Refractive index particle: {n_particle}")
print(f"Refractive index medium: {n_medium}")
print(f"Relative refractive index: {n_particle / n_medium}")
print(f"Anisotropy: {anysotropy}")


config = SimConfig(
    n_photons=n_photons,
    medium=medium,
    detector=detectors_container,
    laser=laser_source,
    track_reverse_paths=False,
)
config.n_threads = 8

run_simulation_parallel(config)

end_time = time.time()
print(f"---- Simulation time: {end_time - start_time:.2f} seconds")


save_angle_detector_fields("speckle_fields.dat", speckle_detector)


speckle_pattern = compute_speckle_angledetector(speckle_detector)
I_total_array = np.array(speckle_pattern.I_total, copy=False)
I_x_array = np.array(speckle_pattern.Ix, copy=False)
I_y_array = np.array(speckle_pattern.Iy, copy=False)
I_z_array = np.array(speckle_pattern.Iz, copy=False)

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


# plot histogram from hitogram_detector the data is just ints that count hits
def plot_histogram(detector, title):
    hits_data = np.array(detector.histogram)
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(hits_data)), hits_data, linewidth=2)
    plt.xlabel("Event Index")
    plt.ylabel("Counts")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

plot_histogram(histogram_detector_1, "Histogram Detector 1 (theta 0 to 180 degrees)")
plot_histogram(histogram_detector_2, "Histogram Detector 2 (theta 0 to 90 degrees)")
plot_histogram(histogram_detector_3, "Histogram Detector 3 (theta 0 to 45 degrees)")
plot_histogram(histogram_detector_4, "Histogram Detector 4 (theta 0 to 22.5 degrees)")
plot_histogram(histogram_detector_5, "Histogram Detector 5 (theta 0 to ~11.5 degrees)")
plot_histogram(histogram_detector_6, "Histogram Detector 6 (theta 0 to ~5.7 degrees)")


print(f"Number of photons detected by speckle detector: {speckle_detector.hits}")
print(f"Number of photons detected by histogram detector 1: {histogram_detector_1.hits}")
print(f"Number of photons detected by histogram detector 2: {histogram_detector_2.hits}")
print(f"Number of photons detected by histogram detector 3: {histogram_detector_3.hits}")
print(f"Number of photons detected by histogram detector 4: {histogram_detector_4.hits}")
print(f"Number of photons detected by histogram detector 5: {histogram_detector_5.hits}")
print(f"Number of photons detected by histogram detector 6: {histogram_detector_6.hits}")


theta_bins = np.linspace(0, np.pi / 2, len(theta_histogram_detector.histogram) + 1)
theta_centers = (theta_bins[:-1] + theta_bins[1:]) / 2
plt.figure(figsize=(8, 5))
plt.plot(theta_centers, theta_histogram_detector.histogram, linewidth=2)
plt.xlabel('Theta (rad)')
plt.ylabel('Count')
plt.title('Theta Histogram')
plt.grid(True, alpha=0.3)
plt.show()