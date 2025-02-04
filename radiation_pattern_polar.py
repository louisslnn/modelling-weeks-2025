import numpy as np
import matplotlib.pyplot as plt

def plot_polar_array_factor(N, f, delta_phi, do_square, do_db, plot_beamwidth, angle_step=0.1, c=3e8):
    """
    Plot the array factor in polar coordinates to study grating lobes.
    """
    # Wavelength and wavenumber
    L = 4  # Antenna width
    lam = c / f
    k = 2.0 * np.pi / lam
    d = L / (N - 1)

    # Angle array covering FULL 360° for grating lobe detection
    phi_deg = np.arange(0, 360 + angle_step, angle_step)
    phi_rad = np.radians(phi_deg)

    # Compute Radiation intensity
    I = np.zeros_like(phi_rad, dtype=complex)
    for i, phi in enumerate(phi_rad):
        beta = k * d * np.sin(phi) + delta_phi
        I[i] = np.sum(np.exp(1j * np.arange(N) * beta))

    # Convert to magnitude or magnitude-squared
    pattern = np.abs(I) ** 2 if do_square else np.abs(I)
    pattern_norm = pattern / np.max(pattern)  # Normalize

    # Convert to dB scale if selected
    pattern_dB = 10.0 * np.log10(pattern_norm + 1e-12) if do_db else pattern_norm

    # Identify main lobe
    max_gain_idx = np.argmax(pattern_norm)
    max_gain_dir = phi_rad[max_gain_idx]

    # Compute grating lobe positions only if valid
    if lam / d <= 1:
        grating_lobe_positions = np.degrees(np.arcsin(lam / d))  # Single angle
        grating_lobe_positions = np.arange(0, 360, grating_lobe_positions)  # Repeat every grating lobe period
    else:
        grating_lobe_positions = []  # No grating lobes

    # Plot in polar coordinates
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    ax.plot(phi_rad, pattern_dB, 'b-', label="Radiation Pattern")

    # Mark main lobe
    ax.plot(max_gain_dir, pattern_dB[max_gain_idx], 'go', label=f"Main Lobe at {np.degrees(max_gain_dir):.1f}°")

    # Mark grating lobes if they exist
    for angle in grating_lobe_positions:
        if 0 <= angle <= 360:
            ax.plot(np.radians(angle), np.max(pattern_dB), 'ro', label="Grating Lobe" if angle == grating_lobe_positions[0] else "")

    ax.set_title(f"Polar Radiation Pattern (N={N}, Δϕ={delta_phi:.3f} rad)")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # Make angles increase counterclockwise
    ax.grid(True)
    ax.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    plot_polar_array_factor(N=33, f=1.2e9, delta_phi=3.14, do_square=False, do_db=True, plot_beamwidth=True)
