from luminis_mc import (
    Laser,
    SimpleMedium,
    MultiDetector,
    HistogramDetector,
    SpatialCoherentDetector,
    AngularCoherentDetector,
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
anysotropy = phase_function.get_anisotropy_factor(rng_test)

max_cbs_theta = 0.05 # radians

detectors_container = MultiDetector()
cbs_spatial_detector = detectors_container.add_detector(SpatialCoherentDetector(0, mean_free_path*5, mean_free_path*5, 500, 500))
cbs_angular_detector = detectors_container.add_detector(AngularCoherentDetector(0, 100, max_cbs_theta))
histogram_detector_1 = detectors_container.add_detector(HistogramDetector(0, 500))
histogram_detector_2 = detectors_container.add_detector(HistogramDetector(0, 500))
histogram_detector_3 = detectors_container.add_detector(HistogramDetector(0, 500))
histogram_detector_4 = detectors_container.add_detector(HistogramDetector(0, 500))

histogram_detector_1.set_theta_limit(0, np.pi/2)    # 90 degrees
histogram_detector_2.set_theta_limit(0, np.pi/8)    # 22.5 degrees
histogram_detector_3.set_theta_limit(0, 0.1)        # ~11.5 degrees
histogram_detector_4.set_theta_limit(0, max_cbs_theta)        # ~5.7 degrees

cbs_spatial_detector.set_theta_limit(0, max_cbs_theta)


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
    track_reverse_paths=True,
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
print(len(histogram_detector_2.histogram))
print(len(histogram_detector_3.histogram))
print(len(histogram_detector_4.histogram))

plot_histogram(histogram_detector_1, "Histogram Detector 1 (theta 0 to 90 degrees)")
plot_histogram(histogram_detector_2, "Histogram Detector 2 (theta 0 to 22.5 degrees)")
plot_histogram(histogram_detector_3, "Histogram Detector 3 (theta 0 to ~11.5 degrees)")
plot_histogram(histogram_detector_4, f"Histogram Detector 4 (theta 0 to {max_cbs_theta} radians)")
print(f"Number of photons detected by histogram detector 1: {histogram_detector_1.hits}")
print(f"Number of photons detected by histogram detector 2: {histogram_detector_2.hits}")
print(f"Number of photons detected by histogram detector 3: {histogram_detector_3.hits}")
print(f"Number of photons detected by histogram detector 4: {histogram_detector_4.hits}")
print(f"Number of photons detected by CBS spatial detector: {cbs_spatial_detector.hits}")
print(f"Number of photons detected by CBS angular detector: {cbs_angular_detector.hits}")




I_x_array = np.array(cbs_spatial_detector.I_x, copy=False)
I_y_array = np.array(cbs_spatial_detector.I_y, copy=False)
I_inco_x_array = np.array(cbs_spatial_detector.I_inco_x, copy=False)
I_inco_y_array = np.array(cbs_spatial_detector.I_inco_y, copy=False)

I_x_enhancement = np.divide(I_x_array, I_inco_x_array, out=np.ones_like(I_x_array), where=I_inco_x_array!=0)
I_y_enhancement = np.divide(I_y_array, I_inco_y_array, out=np.ones_like(I_y_array), where=I_inco_y_array!=0)

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
study_spatial_intensity(I_inco_x_array, "Incoherent X Polarization Intensity")
study_spatial_intensity(I_inco_y_array, "Incoherent Y Polarization Intensity")
study_spatial_intensity(I_x_enhancement, "Enhancement Factor X Polarization")
study_spatial_intensity(I_y_enhancement, "Enhancement Factor Y Polarization")


I_x_theta_array = np.array(cbs_angular_detector.I_x)
I_y_theta_array = np.array(cbs_angular_detector.I_y)

I_plus_theta_array = np.array(cbs_angular_detector.I_plus)
I_minus_theta_array = np.array(cbs_angular_detector.I_minus)
I_total_array = np.array(cbs_angular_detector.I_total)

I_inco_x_theta_array = np.array(cbs_angular_detector.I_inco_x)
I_inco_y_theta_array = np.array(cbs_angular_detector.I_inco_y)

I_inco_plus_theta_array = np.array(cbs_angular_detector.I_inco_plus)
I_inco_minus_theta_array = np.array(cbs_angular_detector.I_inco_minus)
I_inco_total_array = np.array(cbs_angular_detector.I_inco_total)

def plot_angular_intensity(I_theta_array, title):
    theta_values = np.linspace(0, max_cbs_theta, len(I_theta_array))
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
plot_angular_intensity(I_plus_theta_array, "Angular Coherent Plus Polarization Intensity")
plot_angular_intensity(I_minus_theta_array, "Angular Coherent Minus Polarization Intensity")
plot_angular_intensity(I_total_array, "Angular Coherent Total Intensity")
plot_angular_intensity(I_inco_x_theta_array, "Angular Incoherent X Polarization Intensity")
plot_angular_intensity(I_inco_y_theta_array, "Angular Incoherent Y Polarization Intensity")
plot_angular_intensity(I_inco_plus_theta_array, "Angular Incoherent Plus Polarization Intensity")
plot_angular_intensity(I_inco_minus_theta_array, "Angular Incoherent Minus Polarization Intensity")
plot_angular_intensity(I_inco_total_array, "Angular Incoherent Total Intensity")


I_x_enhancement = np.divide(I_x_theta_array, I_inco_x_theta_array, out=np.ones_like(I_x_theta_array), where=I_inco_x_theta_array!=0)
I_y_enhancement = np.divide(I_y_theta_array, I_inco_y_theta_array, out=np.ones_like(I_y_theta_array), where=I_inco_y_theta_array!=0)
I_plus_enhancement = np.divide(I_plus_theta_array, I_inco_plus_theta_array, out=np.ones_like(I_plus_theta_array), where=I_inco_plus_theta_array!=0)
I_minus_enhancement = np.divide(I_minus_theta_array, I_inco_minus_theta_array, out=np.ones_like(I_minus_theta_array), where=I_inco_minus_theta_array!=0)
I_total_enhancement = np.divide(I_total_array, I_inco_total_array, out=np.ones_like(I_total_array), where=I_inco_total_array!=0)

plot_angular_intensity(I_x_enhancement, "Enhancement Factor X Polarization")
plot_angular_intensity(I_y_enhancement, "Enhancement Factor Y Polarization")
plot_angular_intensity(I_plus_enhancement, "Enhancement Factor Plus Polarization")
plot_angular_intensity(I_minus_enhancement, "Enhancement Factor Minus Polarization")
plot_angular_intensity(I_total_enhancement, "Enhancement Factor Total Intensity")
