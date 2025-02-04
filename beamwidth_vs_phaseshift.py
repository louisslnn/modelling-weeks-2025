import numpy as np
import matplotlib.pyplot as plt
from radiation_pattern import *

# Define fixed parameters
L = 4.0  # Antenna array length in meters
wavelength = 0.25  # Wavelength in meters
phase_shifts = np.linspace(-90, 90, 100)  # Phase shifts in degrees
phase_shifts_rad = np.radians(phase_shifts)  # Convert to radians

# Allow user to adjust N dynamically
N_values = [40]

plt.figure(figsize=(8, 6))
for N in N_values:
    d = L / (N - 1)
    hpbw_values = [snippet_compute_beamwidth(N, d, wavelength, delta_phi_rad) for delta_phi_rad in phase_shifts_rad]
    plt.plot(phase_shifts_rad, hpbw_values, 'r', label=f"N = {N}")

plt.xlabel("Phase Shift (Rad)")
plt.ylabel("Half-Power Beamwidth (Degrees)")
plt.title("Beamwidth as a Function of Phase Shift")
plt.legend()
plt.grid(True)
plt.show()
