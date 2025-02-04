import matplotlib.pyplot as plt
import numpy as np

d = np.linspace(0, 0.1, 100)
theta = np.linspace(-np.pi/4, np.pi/4, 100)
lam = 0.25

delta_phi =  2*np.pi*d*np.sin(theta)/lam

plt.ylabel("phase shift delta phi in rad")
plt.xlabel("d in meters")
plt.plot(d, delta_phi)
plt.show()