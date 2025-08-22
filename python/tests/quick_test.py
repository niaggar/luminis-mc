# %%
import luminis_mc as lm

cfg = lm.SimConfig()
mat = lm.Material(5.0)
det = lm.PlaneDetector(100.0, 10.0)
sim = lm.Simulation(cfg, mat, det)
stats = sim.run()
print("rate:", stats.rate)
