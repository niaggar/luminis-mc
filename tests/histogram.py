# %%

from luminis_mc import (
    Detector,
)
from luminis_mc import LogLevel
from luminis_mc import set_log_level
import matplotlib.pyplot as plt
import numpy as np

set_log_level(LogLevel.debug)

detector = Detector(0)
detector.load_recorded_photons("test-data-phi-random.dat")

events_hist = detector.compute_events_histogram(0, np.pi / 2)
theta_hist = detector.compute_theta_histogram(0, np.pi / 2, 100)
phi_hist = detector.compute_phi_histogram(0, 2*np.pi, 100)

# Plot histograms
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Events histogram
axes[0].plot(range(len(events_hist)), events_hist, linewidth=2)
axes[0].set_xlabel('Event Index')
axes[0].set_ylabel('Count')
axes[0].set_title('Events Histogram')
axes[0].grid(True, alpha=0.3)

# Theta histogram
theta_bins = np.linspace(0, np.pi / 2, len(theta_hist) + 1)
theta_centers = (theta_bins[:-1] + theta_bins[1:]) / 2
axes[1].plot(theta_centers, theta_hist, linewidth=2)
axes[1].set_xlabel('Theta (rad)')
axes[1].set_ylabel('Count')
axes[1].set_title('Theta Histogram')
axes[1].grid(True, alpha=0.3)

# Phi histogram
phi_bins = np.linspace(0, 2*np.pi, len(phi_hist) + 1)
phi_centers = (phi_bins[:-1] + phi_bins[1:]) / 2
axes[2].plot(phi_centers, phi_hist, linewidth=2)
axes[2].set_xlabel('Phi (rad)')
axes[2].set_ylabel('Count')
axes[2].set_title('Phi Histogram')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
