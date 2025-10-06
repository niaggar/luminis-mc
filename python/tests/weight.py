import numpy as np
import matplotlib.pyplot as plt


def weight_update_1(w, mua, mut):
    return w - w * (mua / mut)


def weight_update_2(w, mus, mut):
    return w * (mus) / mut


w = 1.0
mua = 0.01
mus = 0.05
mut = mua + mus

n_steps = 10
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
