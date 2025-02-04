import numpy as np
import matplotlib.pyplot as plt
from optimize_azimuthal_bins import *

def compute_phase_for_steerAngle(N, freq, c, steer_deg):
    """
    Convert a desired steering angle (deg) into the required global phase shift (deg)
    using the relationship: phase_deg = -k * d * sin(steer_deg),
    where d = 4/(N-1).
    """
    if N == 1:
        # For a single element, phase shift is irrelevant
        return 0.0

    lam = c / freq
    d = 4.0 / (N - 1)
    k = 2.0 * np.pi / lam
    phase_deg = -np.degrees(k * d * np.sin(np.radians(steer_deg)))
    return phase_deg


def tile_no_gap(N=40, freq=1.2e9, c=3e8, angle_min=-45.0, angle_max=45.0, angle_step=0.2):
    """
    Tile the angular domain [angle_min, angle_max] with lobes such that each new lobe's
    left -3 dB edge coincides with the previous lobe's right -3 dB edge (no gap at -3 dB).

    Returns a list of tuples:
      (steerAngle_deg, phaseShift_deg, HPBW, left_3db, right_3db)
    """
    if N == 1:
        # For a single element, the entire range is covered by one lobe
        return [(0.0, 0.0, 180.0, angle_min, angle_max)]

    coverage_left = angle_min
    lobes = []
    margin = 2.0

    while coverage_left < (angle_max - margin):
        new_steer = find_steering_for_leftEdge(coverage_left, N, freq, c, angle_min, angle_max, angle_step=angle_step)
        new_phase = compute_phase_for_steerAngle(N, freq, c, new_steer)
        _, _, info = compute_array_factor(
            N, freq, c, angle_min, angle_max, new_phase,
            angle_step=angle_step, steer_angle_deg=new_steer
        )
        L3 = info['left_3db']
        R3 = info['right_3db']
        W3 = info['HPBW']

        if R3 <= coverage_left + 1e-5:
            break

        lobes.append((new_steer, new_phase, W3, L3, R3))
        coverage_left = R3
        if len(lobes) > 300:
            break

    if coverage_left < angle_max:
        new_steer = find_steering_for_leftEdge(coverage_left, N, freq, c, angle_min, angle_max, angle_step=angle_step)
        new_phase = compute_phase_for_steerAngle(N, freq, c, new_steer)
        _, _, info = compute_array_factor(
            N, freq, c, angle_min, angle_max, new_phase,
            angle_step=angle_step, steer_angle_deg=new_steer
        )
        L3 = info['left_3db']
        R3 = info['right_3db']
        W3 = info['HPBW']
        if R3 > coverage_left + 1e-5:
            lobes.append((new_steer, new_phase, W3, L3, R3))

    return lobes


def calculate_bins_for_all_elements(max_elements=40, freq=1.2e9, c=3e8, array_len=4.0, angle_min=-45.0, angle_max=45.0):
    """
    Calculates the number of bins required for no-gap coverage for each N (number of elements)
    from 1 to max_elements (inclusive).

    Parameters:
    - max_elements: Maximum number of radiating elements (default: 40).
    - freq: Operating frequency (Hz).
    - c: Speed of light (m/s).
    - array_len: Total array length (meters).
    - angle_min: Minimum coverage angle (degrees).
    - angle_max: Maximum coverage angle (degrees).

    Returns:
    - bins_array: Array of the number of bins for each N.
    """
    bins_array = []  # Array to store bins for each N

    for N in range(1, max_elements + 1):
        try:
            # Calculate lobe tiling for the current N
            final_lobes = tile_no_gap(
                N=N,
                freq=freq,
                c=c,
                angle_min=angle_min,
                angle_max=angle_max
            )
            # Count the number of lobes (bins)
            bins_array.append(len(final_lobes))
        except ValueError as e:
            # Handle cases where invalid configurations occur
            bins_array.append(None)
            print(f"Error for N={N}: {e}")

    return np.array(bins_array)

def main():
    # Constants
    max_elements = 40
    freq = 1.2e9  # Frequency (Hz)
    c = 3e8  # Speed of light (m/s)
    array_len = 4.0  # Array length (meters)
    angle_min = -45.0  # Min coverage angle (degrees)
    angle_max = 45.0  # Max coverage angle (degrees)

    # Calculate bins for each N
    bins_array = calculate_bins_for_all_elements(
        max_elements=max_elements,
        freq=freq,
        c=c,
        array_len=array_len,
        angle_min=angle_min,
        angle_max=angle_max
    )

    # Print results
    print("Number of bins for each N (1 to 40):")
    for N, bins in enumerate(bins_array, start=1):
        print(f"N = {N}: bins = {bins}")

    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_elements + 1), bins_array, marker="o", label="Bins vs. Number of Elements")
    plt.xlabel("Number of Radiating Elements (N)")
    plt.ylabel("Number of Bins")
    plt.title("Number of Bins vs. Number of Elements")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
