from luminis_mc import (
    Laser,
    SimpleMedium,
    MultiDetector,
    HistogramDetector,
    SpatialDetector,
    SpatialTimeDetector,
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
mean_free_path_real = 2.8
radius_real = 0.15
wavelength_real = 0.525
n_particle_real = 1.31
n_medium_real = 1.33


mean_free_path_sim = 1.0
inv_mfp_sim = 1 / mean_free_path_sim
mu_absortion_sim = 0.05 * inv_mfp_sim
mu_scattering_sim = inv_mfp_sim - mu_absortion_sim



size_parameter = 2 * np.pi * radius_real / wavelength_real
condition_1 = np.abs(n_particle_real / n_medium_real - 1)
condition_2 = size_parameter * condition_1


# Time parameters
t_ref = mean_free_path_sim / light_speed
dt = 0.5 * t_ref
max_time = 10 * t_ref

# Phase function parameters
thetaMin = 0.00001
thetaMax = np.pi
nDiv = 1000
n_photons = 10_000_000

# Laser parameters
origin = Vec3(0, 0, 0)

polarization = CVec2(1/np.sqrt(2), -1j/np.sqrt(2))  # Circular Right polarization
# polarization = CVec2(1/np.sqrt(2), 1j/np.sqrt(2))   # Circular Left polarization
laser_radius = 0.5 * mean_free_path_sim
laser_type = LaserSource.Gaussian


rng_test = Rng()
laser_source = Laser(origin, polarization, wavelength_real, laser_radius, laser_type)
phase_function = RayleighDebyeEMCPhaseFunction(wavelength_real, radius_real, n_particle_real, n_medium_real, nDiv, thetaMin, thetaMax)
medium = SimpleMedium(mu_absortion_sim, mu_scattering_sim, phase_function, mean_free_path_sim, radius_real, n_particle_real, n_medium_real)

abs_r_len = 5 * mean_free_path_sim
abs_z_len = 10 * mean_free_path_sim
abs_d_r = mean_free_path_sim / 100
abs_d_z = mean_free_path_sim / 100

absorption = AbsorptionTimeDependent(
    radius=abs_r_len,
    depth=abs_z_len,
    d_r=abs_d_r,
    d_z=abs_d_z,
    d_t=dt,
    t_max=15*dt,
)
anysotropy = phase_function.get_anisotropy_factor(rng_test)



# Detector setup
x_len = mean_free_path_sim * 20
y_len = mean_free_path_sim * 20
r_len = mean_free_path_sim * 10
resolution = 500

detectors_container = MultiDetector()
spatial_detector = detectors_container.add_detector(SpatialDetector(0, x_len, y_len, r_len, resolution, resolution, resolution))
histogram_detector_1 = detectors_container.add_detector(HistogramDetector(0, 500))

print("CONDITIONS FOR RAYLEIGH-DEBYE APPROXIMATION:")
print(f"Size parameter: {size_parameter}")
print(f"|n_particle / n_medium - 1|: {condition_1}")
print(f"Size parameter * |n_particle / n_medium - 1|: {condition_2}")

print("MEDIUM PARAMETERS:")
print(f"Mean free path: {mean_free_path_sim}")
print(f"Transport mean free path: {mean_free_path_sim / (1 - anysotropy[0])}")
print(f"Particle radius: {radius_real}")
print(f"Wavelength: {wavelength_real}")
print(f"Scattering coefficient: {mu_scattering_sim}")
print(f"Absorption coefficient: {mu_absortion_sim}")
print(f"Albedo: {mu_scattering_sim / (mu_scattering_sim + mu_absortion_sim)}")
print(f"Refractive index particle: {n_particle_real}")
print(f"Refractive index medium: {n_medium_real}")
print(f"Relative refractive index: {n_particle_real / n_medium_real}")
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
    hits_data = np.array(array_data)
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(hits_data)), hits_data, linewidth=2)
    plt.xlabel("Event Index")
    plt.ylabel("Counts")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


plot_histogram(histogram_detector_1, "Histogram Detector 1 (theta 0 to 90 degrees)")
print(f"Number of photons detected by histogram detector 1: {histogram_detector_1.hits}")
print(f"Number of photons detected by spatial detector: {spatial_detector.hits}")


I_plus_array = np.array(spatial_detector.I_plus, copy=False) / n_photons
I_minus_array = np.array(spatial_detector.I_minus, copy=False) / n_photons

I_rad_plus_array = np.array(spatial_detector.calculate_radial_plus_intensity()) / n_photons
I_rad_minus_array = np.array(spatial_detector.calculate_radial_minus_intensity()) / n_photons

def study_spatial_intensity(I_plus_array, I_minus_array, title, detector_width, detector_height, 
                           nx_bins, ny_bins, length_unit="mfp", save_path=None, show_plot=True):
    """
    Plot spatial intensity distributions with physical units.
    
    Args:
        I_plus_array: Copolarized intensity array
        I_minus_array: Crosspolarized intensity array
        title: Plot title
        detector_width: Width of detector in physical units
        detector_height: Height of detector in physical units
        nx_bins: Number of bins in x direction
        ny_bins: Number of bins in y direction
        length_unit: String describing the length unit (default: "mfp" for mean free path)
        save_path: Optional path to save figure
        show_plot: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Physical extent of the detector
    extent = [-detector_width/2, detector_width/2, -detector_height/2, detector_height/2]
    
    # Copolarized (left)
    im1 = axes[0].imshow(I_plus_array, cmap='inferno', origin='lower', extent=extent, aspect='auto')
    axes[0].set_title("Copolarized Intensity")
    axes[0].set_xlabel(f"X ({length_unit})")
    axes[0].set_ylabel(f"Y ({length_unit})")
    plt.colorbar(im1, ax=axes[0], label="Normalized Intensity")
    
    # Crosspolarized (right)
    im2 = axes[1].imshow(I_minus_array, cmap='inferno', origin='lower', extent=extent, aspect='auto')
    axes[1].set_title("Crosspolarized Intensity")
    axes[1].set_xlabel(f"X ({length_unit})")
    axes[1].set_ylabel(f"Y ({length_unit})")
    plt.colorbar(im2, ax=axes[1], label="Normalized Intensity")
    
    fig.suptitle(title, fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Image saved to: {save_path}")
    
    if show_plot:
        plt.show()

def study_radial_intensity(I_plus_array, I_minus_array, title, max_radius, length_unit="mfp", 
                          save_path=None, show_plot=True):
    """
    Plot radial intensity distributions with physical units.
    
    Args:
        I_plus_array: Copolarized radial intensity array
        I_minus_array: Crosspolarized radial intensity array
        title: Plot title
        max_radius: Maximum radius of detector in physical units
        length_unit: String describing the length unit (default: "mfp" for mean free path)
        save_path: Optional path to save figure
        show_plot: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Create radius array in physical units
    n_bins = len(I_plus_array)
    radius_array = np.linspace(0, max_radius, n_bins)
    
    # Copolarized (left)
    axes[0].plot(radius_array, I_plus_array, marker='o', linestyle='-', color='blue', markersize=3)
    axes[0].set_title("Copolarized Intensity")
    axes[0].set_xlabel(f"Radius ({length_unit})")
    axes[0].set_ylabel("Normalized Intensity")
    axes[0].grid(True, alpha=0.3)
    
    # Crosspolarized (right)
    axes[1].plot(radius_array, I_minus_array, marker='o', linestyle='-', color='red', markersize=3)
    axes[1].set_title("Crosspolarized Intensity")
    axes[1].set_xlabel(f"Radius ({length_unit})")
    axes[1].set_ylabel("Normalized Intensity")
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Image saved to: {save_path}")
    
    if show_plot:
        plt.show()


study_spatial_intensity(I_plus_array, I_minus_array, "Spatial Intensities", 
                       x_len, y_len, resolution, resolution,
                       length_unit="mfp", show_plot=True)
study_radial_intensity(I_rad_plus_array, I_rad_minus_array, "Radial Intensities", 
                      max_radius=r_len, length_unit="mfp", show_plot=True)



def study_absorption_image(abs_image, title, abs_r_len, abs_z_len, length_unit="mfp", save_path=None, show_plot=True):
    """
    Plot absorption image with physical units.
    
    Args:
        abs_image: 2D array of absorbed photon counts
        title: Plot title
        abs_r_len: Maximum radius of absorption image in physical units
        abs_z_len: Maximum depth of absorption image in physical units
        length_unit: String describing the length unit (default: "mfp" for mean free path)
        save_path: Optional path to save figure
        show_plot: Whether to display the plot
    """    
    
    plt.figure(figsize=(8, 5))
    extent = [0.0, abs_z_len, -abs_r_len, abs_r_len]
    plt.imshow(abs_image, cmap='inferno', origin='lower', extent=extent, aspect='auto')
    plt.colorbar(label='Absorbed Photon Count')
    plt.title(title)
    plt.xlabel(f"Depth ({length_unit})")
    plt.ylabel(f"Radius ({length_unit})")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Image saved to: {save_path}")
    
    if show_plot:
        plt.show()



total_absortion_images = len(absorption.time_slices)

for i in range(total_absortion_images):
    abs_image = absorption.get_absorption_image(n_photons, i)
    save_file = f"/Users/niaggar/Downloads/abs/absortion_t{i:03d}.png"
    study_absorption_image(abs_image, f"Absorption Image at Time Slice {i+1}/{total_absortion_images}", 
                          abs_r_len, abs_z_len, length_unit="mfp", save_path=save_file, show_plot=False)

# abs_image = absorption.get_absorption_image(n_photons, 0)
# abs_image_numpy = np.array(abs_image, copy=False)
# print(f"Total absorbed photons: {np.sum(abs_image_numpy)}")
# print(abs_image_numpy)

# extent = [0.0, abs_z_len, -abs_r_len, abs_r_len]

# plt.figure(figsize=(8, 5))
# plt.imshow(abs_image_numpy, cmap='inferno', origin='lower', extent=extent, aspect='auto')
# plt.colorbar(label='Absorbed Photon Count')
# plt.title('Absorption Image (Radius vs Depth)')
# plt.ylabel('Radius (mfp)')
# plt.xlabel('Depth (mfp)')
# plt.grid(True, alpha=0.3)
# plt.show()