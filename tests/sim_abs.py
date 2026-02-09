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


set_log_level(LogLevel.info)

radius_study = np.array([0.15, 0.20, 0.30, 0.50])
start_time = time.time()

# Global frame of reference
m_global = Vec3(1, 0, 0)
n_global = Vec3(0, 1, 0)
s_global = Vec3(0, 0, 1)
light_speed = 1

# Medium parameters in micrometers
mean_free_path_real = 2.8
wavelength_real = 0.525
n_particle_real = 1.31
n_medium_real = 1.33

mean_free_path_sim = 1.0
inv_mfp_sim = 1 / mean_free_path_sim
mu_absortion_sim = 0.001 * inv_mfp_sim
mu_scattering_sim = inv_mfp_sim - mu_absortion_sim

# Time parameters
t_ref = mean_free_path_sim / light_speed
dt = 0.5 * t_ref
max_time = 10 * t_ref

# Phase function parameters
thetaMin = 0.00001
thetaMax = np.pi
nDiv = 1000
n_photons = 1000_000

# Laser parameters
origin = Vec3(0, 0, 0)
polarization = CVec2(1/np.sqrt(2), -1j/np.sqrt(2))  # Circular Right polarization
# polarization = CVec2(1/np.sqrt(2), 1j/np.sqrt(2))   # Circular Left polarization
laser_radius = 0.1 * mean_free_path_sim
laser_type = LaserSource.Gaussian

# Detector parameters
resolution = 500
x_len = mean_free_path_sim * 20
y_len = mean_free_path_sim * 20
r_len = mean_free_path_sim * 10

abs_r_len = 5 * mean_free_path_sim
abs_z_len = 10 * mean_free_path_sim
abs_d_r = mean_free_path_sim / 100
abs_d_z = mean_free_path_sim / 100



data_storage = []

for r in radius_study:
    radius_real = r
    size_parameter = 2 * np.pi * radius_real / wavelength_real
    condition_1 = np.abs(n_particle_real / n_medium_real - 1)
    condition_2 = size_parameter * condition_1

    # Setup simulation components
    rng_test = Rng()
    laser_source = Laser(origin, polarization, wavelength_real, laser_radius, laser_type)
    phase_function = RayleighDebyeEMCPhaseFunction(wavelength_real, radius_real, n_particle_real, n_medium_real, nDiv, thetaMin, thetaMax)
    medium = SimpleMedium(mu_absortion_sim, mu_scattering_sim, phase_function, mean_free_path_sim, radius_real, n_particle_real, n_medium_real)
    anysotropy = phase_function.get_anisotropy_factor(rng_test)

    absorption = AbsorptionTimeDependent(
        radius=abs_r_len,
        depth=abs_z_len,
        d_r=abs_d_r,
        d_z=abs_d_z,
        d_t=0,
        t_max=0,
    )

    # Detector setup
    detectors_container = MultiDetector()
    spatial_detector = detectors_container.add_detector(SpatialDetector(0, x_len, y_len, r_len, resolution, resolution, resolution))

    print("CONDITIONS FOR RAYLEIGH-DEBYE APPROXIMATION:")
    print(f"Particle radius: {radius_real}")
    print(f"Size parameter: {size_parameter}")
    print(f"Size parameter * |n_particle / n_medium - 1|: {condition_2}")
    print(f"Anisotropy: {anysotropy}")

    # Run simulation
    config = SimConfig(
        n_photons=n_photons,
        medium=medium,
        detector=detectors_container,
        laser=laser_source,
        track_reverse_paths=False,
        absorption=absorption,
    )
    config.n_threads = 8
    run_simulation_parallel(config)

    # Extracting intensity data
    I_rad_plus_array = np.array(spatial_detector.calculate_radial_plus_intensity()) / n_photons
    I_rad_minus_array = np.array(spatial_detector.calculate_radial_minus_intensity()) / n_photons

    # Extracting absorption data
    absorption_data = absorption.get_absorption_image(n_photons, 0)

    data_storage.append((radius_real, size_parameter, condition_2, anysotropy[0], I_rad_plus_array, I_rad_minus_array, absorption_data))



end_time = time.time()
print(f"---- Simulation time: {end_time - start_time:.2f} seconds")

# Plot the I_rad_plus_array for each radius in a single figure
# plt.rcParams['text.usetex'] = True
# plt.figure(figsize=(10, 6))
# for (radius_real, size_parameter, condition_2, anysotropy, I_rad_plus_array, I_rad_minus_array, absorption_data) in data_storage:
#     n_bins = len(I_rad_plus_array)
#     radius_array = np.linspace(0, r_len, n_bins)
#     plt.plot(radius_array, I_rad_plus_array, label=f'$g={anysotropy:.2f}, r={radius_real:.2f}$')
# plt.xlabel('Radius (l)')
# plt.ylabel('Radial Copolarized Intensity')
# plt.title('Radial Copolarized Intensity for Different Particle Radii')
# plt.xlim(0, 4)
# plt.legend()
# plt.grid(True)
# plt.show()


# Plot the absorption data for each radius in a different figure each
for (radius_real, size_parameter, condition_2, anysotropy, I_rad_plus_array, I_rad_minus_array, absorption_data) in data_storage:
    plt.rcParams['text.usetex'] = True
    plt.figure(figsize=(6, 5))
    extent = [0.0, abs_z_len, -abs_r_len, abs_r_len]
    plt.imshow(absorption_data, cmap='inferno', origin='lower', extent=extent, aspect='auto')
    plt.colorbar(label='Absorption Probability')
    plt.title(f'Absorption Distribution for r={radius_real:.2f}, g={anysotropy:.2f}')
    plt.xlabel('Radius (l)')
    plt.ylabel('Depth (l)')
    plt.grid(False)
    plt.show()














