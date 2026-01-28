from luminis_mc import (
    Laser,
    SimpleMedium,
    MultiDetector,
    HistogramDetector,
    SpatialCoherentDetector,
    SimConfig,
    RayleighDebyeEMCPhaseFunction,
    CVec2,
    Vec3,
    Rng,
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
mu_absortion = 0.1 * inv_mfp
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

detectors_container = MultiDetector()
cbs_detector = detectors_container.add_detector(SpatialCoherentDetector(0, 280.0, 280.0, 1000, 1000))
histogram_detector_1 = detectors_container.add_detector(HistogramDetector(0, 500))
histogram_detector_2 = detectors_container.add_detector(HistogramDetector(0, 500))
histogram_detector_3 = detectors_container.add_detector(HistogramDetector(0, 500))
histogram_detector_4 = detectors_container.add_detector(HistogramDetector(0, 500))

histogram_detector_1.set_theta_limit(0, np.pi/2)    # 90 degrees
histogram_detector_2.set_theta_limit(0, np.pi/8)    # 22.5 degrees
histogram_detector_3.set_theta_limit(0, 0.2)        # ~11.5 degrees
histogram_detector_4.set_theta_limit(0, 0.1)        # ~5.7 degrees

cbs_detector.set_theta_limit(0, 0.05)


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
    track_reverse_paths=True,
)
config.n_threads = 8

run_simulation_parallel(config)

end_time = time.time()
print(f"---- Simulation time: {end_time - start_time:.2f} seconds")





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

plot_histogram(histogram_detector_1, "Histogram Detector 1 (theta 0 to 90 degrees)")
plot_histogram(histogram_detector_2, "Histogram Detector 2 (theta 0 to 22.5 degrees)")
plot_histogram(histogram_detector_3, "Histogram Detector 3 (theta 0 to ~11.5 degrees)")
plot_histogram(histogram_detector_4, "Histogram Detector 4 (theta 0 to ~5.7 degrees)")
print(f"Number of photons detected by histogram detector 1: {histogram_detector_1.hits}")
print(f"Number of photons detected by histogram detector 2: {histogram_detector_2.hits}")
print(f"Number of photons detected by histogram detector 3: {histogram_detector_3.hits}")
print(f"Number of photons detected by histogram detector 4: {histogram_detector_4.hits}")
print(f"Number of photons detected by CBS detector: {cbs_detector.hits}")



I_x_array = np.array(cbs_detector.I_x, copy=False)
I_y_array = np.array(cbs_detector.I_y, copy=False)
I_z_array = np.array(cbs_detector.I_z, copy=False)

I_inco_x_array = np.array(cbs_detector.I_inco_x, copy=False)
I_inco_y_array = np.array(cbs_detector.I_inco_y, copy=False)
I_inco_z_array = np.array(cbs_detector.I_inco_z, copy=False)

print("Plotting CBS Detector Spatial Intensities...")
print("Coherent Intensities:")

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

study_spatial_intensity(I_x_array, "Coherent X Polarization Intensity")
study_spatial_intensity(I_y_array, "Coherent Y Polarization Intensity")
study_spatial_intensity(I_z_array, "Coherent Z Polarization Intensity")
study_spatial_intensity(I_inco_x_array, "Incoherent X Polarization Intensity")
study_spatial_intensity(I_inco_y_array, "Incoherent Y Polarization Intensity")
study_spatial_intensity(I_inco_z_array, "Incoherent Z Polarization Intensity")


I_x_theta_array = np.array(cbs_detector.I_x_theta)
I_y_theta_array = np.array(cbs_detector.I_y_theta)
I_z_theta_array = np.array(cbs_detector.I_z_theta)

I_inco_x_theta_array = np.array(cbs_detector.I_inco_x_theta)
I_inco_y_theta_array = np.array(cbs_detector.I_inco_y_theta)
I_inco_z_theta_array = np.array(cbs_detector.I_inco_z_theta)

def plot_angular_intensity(I_theta_array, title):
    theta_values = np.linspace(0, 0.1, len(I_theta_array))
    plt.figure(figsize=(8, 5))
    plt.plot(theta_values, I_theta_array, linewidth=2)
    plt.xlabel("Theta (radians)")
    plt.ylabel("Intensity")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

print("Angular Intensities:")
plot_angular_intensity(I_x_theta_array, "Angular Coherent X Polarization Intensity")
plot_angular_intensity(I_y_theta_array, "Angular Coherent Y Polarization Intensity")
plot_angular_intensity(I_z_theta_array, "Angular Coherent Z Polarization Intensity")
plot_angular_intensity(I_inco_x_theta_array, "Angular Incoherent X Polarization Intensity")
plot_angular_intensity(I_inco_y_theta_array, "Angular Incoherent Y Polarization Intensity")
plot_angular_intensity(I_inco_z_theta_array, "Angular Incoherent Z Polarization Intensity")
