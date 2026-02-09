from luminis_mc import (
    Laser,
    SimpleMedium,
    MultiDetector,
    HistogramDetector,
    SpatialDetector,
    AbsorptionTimeDependent,
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
radius = 0.3
mean_free_path = 2.8
wavelength = 0.525
inv_mfp = 1 / mean_free_path
mu_absortion = 0.001 * inv_mfp
mu_scattering = inv_mfp - mu_absortion
n_particle = 1.31
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
n_photons = 1_000_000

# Laser parameters
origin = Vec3(0, 0, 0)
polarization = CVec2(1/np.sqrt(2), 1j/np.sqrt(2))  # Circular polarization
laser_radius = 0.1 * mean_free_path
laser_type = LaserSource.Gaussian


rng_test = Rng()
laser_source = Laser(origin, polarization, wavelength, laser_radius, laser_type)
phase_function = RayleighDebyeEMCPhaseFunction(wavelength, radius, n_particle, n_medium, nDiv, thetaMin, thetaMax)
medium = SimpleMedium(mu_absortion, mu_scattering, phase_function, mean_free_path, radius, n_particle, n_medium)
absorption = AbsorptionTimeDependent(
    radius=10 * mean_free_path,
    depth=100 * mean_free_path,
    d_r=mean_free_path / 50,
    d_z=mean_free_path / 100,
    d_t=0,
    t_max=0,
)
anysotropy = phase_function.get_anisotropy_factor(rng_test)

max_cbs_theta = np.pi / 2 # radians

detectors_container = MultiDetector()
spatial_detector = detectors_container.add_detector(SpatialDetector(0, mean_free_path*5, mean_free_path*5, 500, 500))
histogram_detector_1 = detectors_container.add_detector(HistogramDetector(0, 500))

# histogram_detector_1.set_theta_limit(0, np.pi/2)
# spatial_detector.set_theta_limit(0, max_cbs_theta)


print("CONDITIONS FOR RAYLEIGH-DEBYE APPROXIMATION:")
print(f"Size parameter: {size_parameter}")
print(f"|n_particle / n_medium - 1|: {condition_1}")
print(f"Size parameter * |n_particle / n_medium - 1|: {condition_2}")

print("MEDIUM PARAMETERS:")
print(f"Mean free path: {mean_free_path}")
print(f"Transport mean free path: {mean_free_path / (1 - anysotropy[0])}")
print(f"Max cbs theory theta: {wavelength / (np.sqrt(2) * (mean_free_path / (1 - anysotropy[0])))}")
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
    absorption=absorption,
    track_reverse_paths=False,
)
config.n_threads = 8

run_simulation_parallel(config)

end_time = time.time()
print(f"---- Simulation time: {end_time - start_time:.2f} seconds")





# plot histogram from hitogram_detector the data is just ints that count hits
def plot_histogram(detector, title):
    array_data = np.array(detector.histogram, int)
    # array_data = array_data / np.max(array_data)
    hits_data = np.array(array_data)
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(hits_data)), hits_data, linewidth=2)
    plt.xlabel("Event Index")
    plt.ylabel("Counts")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

print(len(histogram_detector_1.histogram))

plot_histogram(histogram_detector_1, "Histogram Detector 1 (theta 0 to 90 degrees)")
print(f"Number of photons detected by histogram detector 1: {histogram_detector_1.hits}")
print(f"Number of photons detected by spatial detector: {spatial_detector.hits}")


print(spatial_detector.I_x)

I_x_array = np.array(spatial_detector.I_x, copy=False)

print(I_x_array)

I_y_array = np.array(spatial_detector.I_y, copy=False)
I_plus_array = np.array(spatial_detector.I_plus, copy=False)
I_minus_array = np.array(spatial_detector.I_minus, copy=False)

print("Plotting CBS Detector Spatial Intensities")

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
study_spatial_intensity(I_plus_array, "Copolarized Intensity")
study_spatial_intensity(I_minus_array, "Crosspolarized Intensity")





abs_image = absorption.get_absorption_image(n_photons, time_index=0)
abs_image_array = np.array(abs_image, copy=False)

plt.figure(figsize=(8, 6))
plt.imshow(abs_image_array, cmap='inferno', origin='lower')
plt.colorbar(label="Absorption")
plt.title("Absorption Spatial Distribution")
plt.xlabel("Radius (micrometers)")
plt.ylabel("Depth (micrometers)")
plt.show()
