"""Quick smoke test for Sample layered media."""
from luminis_mc import (
    Sample, SampleLayer, RGDMedium, UniformPhaseFunction,
    SimConfig, Laser, SensorsGroup, PhotonRecordSensor,
    CVec2, LaserSource, run_simulation
)

# Create a phase function
pf = UniformPhaseFunction()

# Create two media with different scattering
m1 = RGDMedium(0.0, 1.0, pf, 1.0, 0.01, 1.5, 1.33)
m2 = RGDMedium(0.0, 0.5, pf, 2.0, 0.01, 1.5, 1.33)

# Create layered stack
stack = Sample(n_medium=1.33)
stack.add_layer(m1, 0.0, 5.0)          # Layer 1: z=[0, 5)
stack.add_layer(m2, 5.0, float('inf')) # Layer 2: z=[5, inf)

print(f"Stack size: {stack.size()}")
print(f"z_top: {stack.z_top()}")
print(f"Layer 0: z=[{stack.get_layer(0).z_min}, {stack.get_layer(0).z_max}), mu_s={stack.get_layer(0).medium.mu_s}")
print(f"Layer 1: z=[{stack.get_layer(1).z_min}, {stack.get_layer(1).z_max}), mu_s={stack.get_layer(1).medium.mu_s}")
print(f"Interfaces: {stack.interfaces}")

# Run a quick simulation with the stack
laser = Laser(CVec2(1, 0), CVec2(0, 1), 500.0, 0.0, LaserSource.Point)
sens = SensorsGroup()
det = sens.add_detector(PhotonRecordSensor(z=0, absorb=True))
config = SimConfig(n_photons=1000, sample=stack, laser=laser, detector=sens)
run_simulation(config)
print(f"Detected photons: {len(det.recorded_photons)}")

# Also test single-layer (infinite) stack
stack2 = Sample(n_medium=1.33)
stack2.add_layer(m1, 0.0, float('inf'))
sens2 = SensorsGroup()
det2 = sens2.add_detector(PhotonRecordSensor(z=0, absorb=True))
config2 = SimConfig(n_photons=1000, sample=stack2, laser=laser, detector=sens2)
run_simulation(config2)
print(f"Single-layer detected photons: {len(det2.recorded_photons)}")

print("SUCCESS: Layered media simulation works!")
