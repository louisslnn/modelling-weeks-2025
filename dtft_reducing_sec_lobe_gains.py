import numpy as np
import matplotlib.pyplot as plt

# Constants
f = 1200e6 # Frequency (Hz)
c = 3e8 # Speed of light (m/s)
lambda_ = c / f # Wavelength (m)
k = 2 * np.pi / lambda_ # Wave number
angles = np.linspace(-90, 90, 1000) # Azimuth angles
L = 4 # Aperture size (meters)
N = 40 # Fixed number of elements

# Function to generate tapering weights
def taper_weights(N, taper_type):
    n = np.arange(N)
    if taper_type == "hamming":
        #Usual values for this window
        alpha = 0.54
        beta = 0.46
        return alpha - beta * np.cos((2 * np.pi * n) / (N - 1))
    elif taper_type == "hann":
        #Usual values for this window
        alpha = 0.5
        beta = 0.5
        return alpha - beta * np.cos((2 * np.pi * n) / (N - 1))
    elif taper_type == "uniform":  # Default: Uniform weights
        return np.ones(N)

# Function to calculate radiation intensity (I = AF^2)
def radiation_intensity(theta, N, taper_type):
    d = L / (N - 1)  # Element spacing
    theta_rad = np.radians(theta)
    big_delta_phi = k * d * np.sin(theta_rad)  # Geometric phase shift
    
    # Generate weights for the selected tapering method
    weights = np.array(taper_weights(N, taper_type))[:, np.newaxis]
    
    # Compute array factor
    element_indices = np.arange(N)
    element_phase = np.exp(1j * element_indices[:, np.newaxis] * big_delta_phi)
    numerator = np.sum(weights * element_phase, axis=0)  # Weighted sum of element contributions
    I = (np.abs(numerator) / np.max(np.abs(numerator)))**2  # Normalize gain

    return I

if __name__ == "__main__":
    # Compute intensity for all tapers
    I_uniform = radiation_intensity(angles, N, "uniform")
    I_hamming = radiation_intensity(angles, N, "hamming")
    I_hann = radiation_intensity(angles, N, "hann")

    # Plot results
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(angles, 10 * np.log10(I_uniform), label="Uniform", color="blue", linewidth=2)
    ax.plot(angles, 10 * np.log10(I_hamming), label="Hamming", color="green", linewidth=2)
    ax.plot(angles, 10 * np.log10(I_hann), label="Hann", color="red", linewidth=2)

    # Formatting
    ax.set_title("Radiation Intensity for Uniform, Hamming, and Hann Windows")
    ax.set_xlabel("Azimuth Angle (degrees)")
    ax.set_ylabel("Radiation Intensity (dB)")
    ax.set_xlim(-90, 90)
    ax.set_ylim(-50, 10)
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=10)

    # Show plot
    plt.show()