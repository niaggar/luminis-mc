# %%
from luminis_mc import Laser, Rng
from luminis_mc import set_log_level
from luminis_mc import LogLevel, LaserSource
import numpy as np

set_log_level(LogLevel.debug)

source = LaserSource.Uniform
laser_source = Laser([0,0,0], [0,0,1], [1,0], 1.0, 5.0, source)
rng = Rng(seed=42)


# %%

num_samples = 100000
emited_positions = []
for _ in range(num_samples):
    photon = laser_source.emit_photon(rng)
    emited_positions.append((photon.pos[0], photon.pos[1]))


# %%

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(np.array(emited_positions)[:,0], np.array(emited_positions)[:,1], 'o', markersize=1, alpha=0.5)
plt.title('Photon Emission Positions')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.axis('equal')
plt.grid(True)
plt.show()

# %%
