"""
Validacion de CBS para particulas de poliestireno.

Barre varios radios de particula y guarda, por cada caso, el perfil de CBS
(metodo del estimador, integrado en el tiempo). La construccion fisica es
imperativa; el guardado esta estandarizado con `save_params` / `save_sensors` /
`save_processed`.
"""

import __main__
import time
import numpy as np

from luminis_mc import (
    Experiment,
    SweepManager,
    Laser, RGDMedium, Sample, FarFieldCBSSensor, StatisticsSensor, SensorsGroup,
    SimConfig, RayleighDebyeEMCPhaseFunction,
    run_simulation_parallel, postprocess_farfield_cbs, derived_quantities,
    set_log_level, LogLevel, LaserSource,
)


set_log_level(LogLevel.info)

# ---------------------------------------------------------------------------
# Salida
# ---------------------------------------------------------------------------
exp_name = "cbs_poly_validation"
base_dir = "/Users/niaggar/Documents/Thesis/tests"

sweep = SweepManager(exp_name, base_dir, timestamped=False)
sweep.snapshot_master_script(__main__.__file__)
sweep.log_readme(
    "CBS validation for polystyrene particles (estimator method)."
)

# ---------------------------------------------------------------------------
# Parametros fisicos
# ---------------------------------------------------------------------------
params_sweep = [
    {"radius": 0.020, "volume_fraction": 0.10},
    {"radius": 0.035, "volume_fraction": 0.10},
    {"radius": 0.055, "volume_fraction": 0.10},
    {"radius": 0.075, "volume_fraction": 0.10},
    {"radius": 0.175, "volume_fraction": 0.10},
]

n_particle = 1.59
n_medium = 1.33
mu_absortion_percent = 0.0
wavelength = 0.514

# Laser: polarizacion circular (helicidad), incidencia normal
laser_m_polarization_state = 1 / np.sqrt(2)
laser_n_polarization_state = -1j / np.sqrt(2)
laser_radius = 4.0
laser_type = LaserSource.Gaussian

# Funcion de fase
phasef_theta_min = 0.0
phasef_theta_max = np.pi
phasef_ndiv = 1_000

# Numero de fotones
n_photons_estimator = 1_000

# ---------------------------------------------------------------------------
# Sensor de campo lejano (cono CBS)
# ---------------------------------------------------------------------------
theta_max_far_field = np.deg2rad(1)
phi_max_far_field = 2 * np.pi
n_theta_far_field = 1000
n_phi_far_field = 1
d_theta = theta_max_far_field / n_theta_far_field
d_phi = phi_max_far_field / n_phi_far_field

# Una sola ventana temporal -> integrado en el tiempo
t_max = 0
d_time = 0


def run_estimator_simulation(exp: Experiment, *, radius, volume_fraction):
    """Construye el sistema, ejecuta la simulacion CBS y guarda los resultados."""
    # --- sistema fisico (imperativo, flexible) ---
    phase = RayleighDebyeEMCPhaseFunction(
        wavelength, radius, n_particle, n_medium,
        phasef_ndiv, phasef_theta_min, phasef_theta_max,
    )
    medium = RGDMedium(phase, radius, n_particle, n_medium, wavelength)
    sample = Sample(n_medium)
    sample.add_layer(medium, 0.0, float("inf"))

    dq = derived_quantities(medium, volume_fraction)   # Q_sca, g, l_s, l*, theta_coh, ...
    mean_free_path = dq["mean_free_path"]
    inv_mfp = 1.0 / mean_free_path
    mu_absortion = mu_absortion_percent * inv_mfp
    mu_scattering = inv_mfp - mu_absortion

    medium.set_mean_free_path(mean_free_path)
    medium.set_scattering_coefficient(mu_scattering)
    medium.set_absorption_coefficient(mu_absortion)

    laser = Laser(
        laser_m_polarization_state, laser_n_polarization_state,
        wavelength, laser_radius * mean_free_path, laser_type,
    )

    print("---- Parametros de la simulacion -----")
    print(f"eficiencia de scattering: {dq['scattering_efficiency']:.3f}")
    print(f"camino libre medio l_s: {mean_free_path:.3f} um")
    print(f"camino de transporte l*: {dq['transport_mean_free_path']:.3f} um")
    print(f"factor de anisotropia g: {dq['anisotropy_g']:.4f}")
    print(f"ancho aprox del cono CBS: {np.rad2deg(dq['theta_coherent']):.3f} deg")

    # --- sensores ---
    sens = SensorsGroup()
    det = sens.add_detector(
        FarFieldCBSSensor(theta_max_far_field, phi_max_far_field, t_max,
                          d_theta, d_phi, d_time, True)
    )
    det.set_theta_limit(0, theta_max_far_field)

    stats = sens.add_detector(StatisticsSensor(z=0, absorb=True))
    stats.set_theta_limit(0, theta_max_far_field)

    # --- config ---
    config = SimConfig()
    config.n_photons = n_photons_estimator
    config.sample = sample
    config.detector = sens
    config.laser = laser
    config.track_reverse_paths = True          # <-- imprescindible para CBS
    config.pin_threads_to_cores = True
    config.n_threads = 7
    config.show_progress = True

    # --- params (auto-capturados de los objetos + derivados) ---
    exp.save_params(config, extra=dq)

    # --- run ---
    t0 = time.time()
    run_simulation_parallel(config)
    print("runtime_s:", time.time() - t0)
    print("hits:", det.hits)

    # --- guardar RAW + procesados (estandarizado) ---
    exp.save_sensors({"farfield_cbs": det, "statistics": stats})
    cbs = postprocess_farfield_cbs(det, n_photons_estimator)
    exp.save_processed("farfield_cbs", cbs, sensor=det)


for i, data in enumerate(params_sweep):
    radius = data["radius"]
    volume_fraction = data["volume_fraction"]
    run_name = f"radius_{radius:.3f}_volumefraction_{volume_fraction:.3f}"

    print(f"Running CBS test for radius={radius:.3f}")
    sweep.run(i, run_name + "_estimator",
              lambda exp, r=radius, v=volume_fraction:
                  run_estimator_simulation(exp, radius=r, volume_fraction=v))
