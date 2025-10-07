import numpy as np
import matplotlib.pyplot as plt


def weight_update_1(w, mua, mut):
    return w - w * (mua / mut)


def weight_update_2(w, mus, mut):
    return w * (mus) / mut


mean_free_path = 2.8 # in micrometers
inv_mfp = 1 / mean_free_path
mu_absortion = 0.003 * inv_mfp
mu_scattering = inv_mfp - mu_absortion

w = 1.0
mua = mu_absortion
mus = mu_scattering
mut = mua + mus
albedo = mus / mut
print(f"Albedo: {albedo}")

n_steps = 1000
weights_1 = np.zeros(n_steps)
weights_2 = np.zeros(n_steps)
weights_1[0] = w
weights_2[0] = w

for i in range(1, n_steps):
    weights_1[i] = weight_update_1(weights_1[i - 1], mua, mut)
    weights_2[i] = weight_update_2(weights_2[i - 1], mus, mut)

plt.plot(weights_1, label="Weight Update 1")
plt.plot(weights_2, label="Weight Update 2", linestyle="--")
plt.xlabel("Step")
plt.ylabel("Weight")
plt.title("Photon Weight Update Comparison")
plt.legend()
plt.grid()
plt.show()
