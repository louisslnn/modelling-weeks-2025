import numpy as np
import matplotlib.pyplot as plt

# Define fixed parameters
L = 4.0  # Antenna array length in meters
wavelength = 0.25  # Wavelength in meters
phase_shifts = np.linspace(-90, 90, 100)  # Phase shifts in degrees
phase_shifts_rad = np.radians(phase_shifts)  # Convert to radians

# Function to compute beamwidth
def compute_beamwidth(N, d, wavelength, delta_phi):
    k = 2.0 * np.pi / wavelength  # Wavenumber
    phi_deg = np.linspace(-45, 45, 1000)  # Angular range in degrees
    phi_rad = np.radians(phi_deg)  # Convert to radians

    # Compute array factor
    I = np.zeros_like(phi_rad, dtype=complex)
    for i, phi in enumerate(phi_rad):
        beta = k * d * np.sin(phi) + delta_phi
        I[i] = np.sum(np.exp(1j * np.arange(N) * beta))
    
    # Normalize pattern
    pattern = np.abs(I) ** 2
    pattern_norm = pattern / np.max(pattern)

    # Find -3 dB beamwidth
    half_max = np.max(pattern_norm) / 2
    indices = np.where(pattern_norm >= half_max)[0]
    if len(indices) > 1:
        beamwidth = phi_deg[indices[-1]] - phi_deg[indices[0]]
    else:
        beamwidth = 0  # Fallback if no clear HPBW detected
    return beamwidth

# Allow user to adjust N
N_values = [40]  # Different values of N for comparison

plt.figure(figsize=(8, 6))
for N in N_values:
    d = L / (N - 1)  # Element spacing
    hpbw_values = [compute_beamwidth(N, d, wavelength, delta_phi) for delta_phi in phase_shifts_rad]
    plt.plot(phase_shifts, hpbw_values, 'r', label=f"N = {N}")

plt.xlabel("Phase Shift (Degrees)")
plt.ylabel("Half-Power Beamwidth (Degrees)")
plt.title("Beamwidth as a Function of Phase Shift")
plt.legend()
plt.grid(True)
plt.show()
