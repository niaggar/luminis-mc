# %%
from luminis_mc import Photon, Detector
from luminis_mc import set_log_level
from luminis_mc import LogLevel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

set_log_level(LogLevel.debug)

# %%

pos = (5, 1, 6)
prev_pos = (0, 5, -7.5)
middle_pos = (0, 0, 0)
dir = (0, 0, 1)

detector = Detector(origin=(0, 0, 0), normal=(0, 0, 1))
photon = Photon(position=(0, 0, -10), direction=(0, 0, 1), wavelength_nm=500)

photon.pos = pos
photon.prev_pos = prev_pos
photon.dir = dir

detector.record_hit(photon)
middle_pos = photon.pos

# %%
# %matplotlib widget

print(f"pos: {pos}, prev_pos: {prev_pos}, middle_pos: {middle_pos}")

fig = plt.figure()
ax = Axes3D(fig)
ax = fig.add_subplot(111, projection="3d")

xx, yy = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
zz = np.zeros_like(xx)  # z=0 plane since detector is at origin with normal (0,0,1)
ax.plot_surface(xx, yy, zz, alpha=0.3, color="lightblue", label="Detector Plane")

ax.scatter(*pos, color="red", s=100, label="pos")
ax.scatter(*prev_pos, color="blue", s=100, label="prev_pos")
ax.scatter(*middle_pos, color="green", s=100, label="middle_pos")

line_x = [prev_pos[0], pos[0]]
line_y = [prev_pos[1], pos[1]]
line_z = [prev_pos[2], pos[2]]
ax.plot(line_x, line_y, line_z, "k--", linewidth=2, label="pos-prev line")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Photon Path and Detector Plane")
plt.show()

# %%
