import numpy as np

def optimize_bins(total_azimuth=90, wavelength=0.03, d_ratio=0.5, N=40):
    # Compute wavenumber k
    L = 4
    k = L / N - 1
    d = d_ratio * wavelength  # Element spacing

    # Find the optimal number of bins
    hpbw = 102/(N*(d/wavelength))
    step = hpbw / 2  # Ensure HPBW/2 overlap
    optimal_bins = int(np.ceil(total_azimuth / step))

    # Compute beam directions
    beam_angles = np.linspace(-total_azimuth / 2, total_azimuth / 2, optimal_bins)

    # Compute phase shifts needed for each beam
    phase_shifts = -np.pi * np.sin(np.radians(beam_angles))

    return optimal_bins, beam_angles, phase_shifts

# Run optimization
optimal_bins, beam_angles, phase_shifts = optimize_bins()

# Display results
print(f"Optimal number of bins: {optimal_bins}")
print(f"Beam angles: {beam_angles}")
print(f"Phase shifts (radians): {phase_shifts}")
