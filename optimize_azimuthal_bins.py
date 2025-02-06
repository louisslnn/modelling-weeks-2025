import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt

################################################################################
# 1) UTILITY FUNCTION: Uniform Amplitude Window
################################################################################
def amplitude_window(num_elements, window_type="uniform"):
    """
    Return amplitude weights for the given number of elements.
    Here we use a uniform window.
    """
    if num_elements < 1:
        return np.array([])
    return np.ones(num_elements)

################################################################################
# 2) High-Resolution Beamwidth Calculation
################################################################################
def compute_pattern_full(N=40, freq_hz=1.2e9, delta_phi=0.0, array_len=4.0, do_square=True):
    """
    Compute the radiation pattern over the full angular range [-90°, 90°] using 10,000 points.
    Returns:
      - full_phi_deg: angle array (deg)
      - pattern_norm: normalized power pattern (peak=1)
      - hpbw: half-power beamwidth (deg)
      - left_edge, right_edge: angles (deg) where the pattern crosses 0.5
      - main_beam_direction: angle (deg) where the pattern is maximum
      - secondary_gain: normalized gain of the highest sidelobe (outside a 2° margin around the main lobe)
      - secondary_angle: angle (deg) corresponding to that secondary lobe gain

    The function uses the standard array factor summation for an array of length 'array_len'
    (with spacing d = array_len/(N-1)) and applies a global phase shift (delta_phi in radians).
    """
    c = 3e8
    lam = c / freq_hz
    k = 2.0 * np.pi / lam
    d = array_len / (N - 1) if N > 1 else 0.0

    # Create a high-resolution angle grid from -90° to 90°
    full_phi_deg = np.linspace(-90, 90, 4000)
    full_phi_rad = np.radians(full_phi_deg)

    # Compute the array factor for each angle
    I = np.zeros_like(full_phi_rad, dtype=complex)
    for i, phi in enumerate(full_phi_rad):
        beta = k * d * np.sin(phi) + delta_phi
        I[i] = np.sum(np.exp(1j * np.arange(N) * beta))

    # Compute and normalize the power pattern
    pattern = np.abs(I)**2
    pattern /= (np.max(pattern) + 1e-12)

    # Find the half-power points (using 0.5 as the threshold)
    half_val = 0.5
    idx = np.where(pattern >= half_val)[0]
    if len(idx) > 1:
        hpbw = full_phi_deg[idx[-1]] - full_phi_deg[idx[0]]
        left_edge = full_phi_deg[idx[0]]
        right_edge = full_phi_deg[idx[-1]]
    else:
        hpbw = 0.0
        left_edge = 0.0
        right_edge = 0.0

    # Main beam direction is the angle corresponding to the maximum pattern value.
    peak_index = np.argmax(pattern)
    main_beam_direction = full_phi_deg[peak_index]

    # For the secondary lobe, exclude a 2° margin around the main lobe region
    margin = 2.0  
    sec_mask = (full_phi_deg < (left_edge - margin)) | (full_phi_deg > (right_edge + margin))
    sec_indices = np.where(sec_mask)[0]
    if len(sec_indices) > 0:
        secondary_index = sec_indices[np.argmax(pattern[sec_indices])]
        secondary_gain = pattern[secondary_index]
        secondary_angle = full_phi_deg[secondary_index]
    else:
        secondary_gain = 0.0
        secondary_angle = 0.0

    if not do_square:
        pattern = np.sqrt(pattern)

    return (full_phi_deg, pattern, hpbw, left_edge, right_edge,
            main_beam_direction, secondary_gain, secondary_angle)

################################################################################
# 3) Compute Array Factor and Retrieve Beam Parameters
################################################################################
def compute_array_factor(N, freq, c, angle_min, angle_max, global_phase_deg,
                         angle_step=0.2, steer_angle_deg=None):
    """
    Compute the array factor (radiation pattern) for a linear array of total length 4 m
    (spacing = 4/(N-1)) with uniform amplitude and a global phase shift.
   
    The pattern is computed over the user-defined angular domain [angle_min, angle_max]
    with resolution angle_step. The global phase shift is provided in degrees (converted to radians internally).
   
    Returns:
      - phi_deg: the angle array (deg) used for computation
      - patt_norm: the normalized power pattern (peak = 1)
      - info: a dictionary containing:
           { 'HPBW'           : half-power beamwidth (deg),
             'left_3db'       : left -3 dB edge (deg),
             'right_3db'      : right -3 dB edge (deg),
             'global_phase'   : global phase shift (deg),
             'main_dir'       : main beam direction (deg),
             'secondary_gain' : secondary lobe gain (normalized),
             'secondary_angle': secondary lobe angle (deg) }
    """
    # Calculate spacing and constant parameters
    d = 4.0 / (N - 1)
    lam = c / freq
    k = 2.0 * np.pi / lam

    # Use uniform amplitude weights
    w = amplitude_window(N, "uniform")

    # Create the angle array for the computation domain
    phi_deg = np.arange(angle_min, angle_max + angle_step, angle_step)
    phi_rad = np.radians(phi_deg)

    # Convert the global phase shift to radians
    delta_phase_rad = np.radians(global_phase_deg)

    # Compute the array factor (AF) by summing contributions from each element
    AF = np.zeros_like(phi_rad, dtype=complex)
    for i, ar in enumerate(phi_rad):
        beta = k * d * np.sin(ar) + delta_phase_rad
        accum = 0+0.j
        for n in range(N):
            accum += w[n] * np.exp(1j * n * beta)
        AF[i] = accum

    patt_unnorm = np.abs(AF)**2
    peak_val = max(np.max(patt_unnorm), 1e-12)
    patt_norm = patt_unnorm / peak_val

    # Determine main beam direction from the computed pattern
    idx_peak = np.argmax(patt_norm)
    main_dir = phi_deg[idx_peak]

    # Use the high-resolution beamwidth calculation (which expects phase in radians)
    (_, _, hpbw, left_edge, right_edge,
     main_dir_new, secondary_gain, secondary_angle) = compute_pattern_full(
         N, freq, delta_phase_rad, 4.0, do_square=True
     )

    info = {
        'HPBW': hpbw,
        'left_3db': left_edge,
        'right_3db': right_edge,
        'global_phase': global_phase_deg,
        'main_dir': main_dir_new,
        'secondary_gain': secondary_gain,
        'secondary_angle': secondary_angle
    }
    return phi_deg, patt_norm, info

################################################################################
# 4) Convert Steering Angle to Global Phase Shift (deg)
################################################################################
def compute_phase_for_steerAngle(N, freq, c, steer_deg):
    """
    Convert a desired steering angle (deg) into the required global phase shift (deg)
    using the relationship: phase_deg = -k * d * sin(steer_deg),
    where d = 4/(N-1).
    """
    lam = c / freq
    d = 4.0 / (N - 1)
    k = 2.0 * np.pi / lam
    phase_deg = -np.degrees(k * d * np.sin(np.radians(steer_deg)))
    return phase_deg

################################################################################
# 5) Binary Search for Steering Angle Based on Desired Left Edge
################################################################################
def find_steering_for_leftEdge(left_edge_target, N, freq, c,
                               angle_min, angle_max, angle_step=0.2, max_iter=40):
    """
    Use binary search in [angle_min, angle_max] to find a steering angle whose main-lobe
    left -3 dB edge is approximately equal to left_edge_target.
    """
    lo = angle_min
    hi = angle_max
    best_steer = 0.5 * (lo + hi)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        phase_deg = compute_phase_for_steerAngle(N, freq, c, mid)
        _, _, info = compute_array_factor(
            N, freq, c, angle_min, angle_max, phase_deg,
            angle_step=angle_step, steer_angle_deg=mid
        )
        left_3db = info['left_3db']

        if left_3db > left_edge_target:
            hi = mid
        else:
            lo = mid

        best_steer = 0.5 * (lo + hi)
        if abs(left_3db - left_edge_target) < 1e-4 or (hi - lo) < 1e-4:
            break

    return best_steer

################################################################################
# 6) Tiling: Determine No-Gap Coverage over the Domain
################################################################################
def tile_no_gap(N=40, freq=1.2e9, c=3e8, angle_min=-45.0, angle_max=45.0, angle_step=0.2):
    """
    Tile the angular domain [angle_min, angle_max] with lobes such that each new lobe's
    left -3 dB edge coincides with the previous lobe's right -3 dB edge (no gap at -3 dB).
   
    Returns a list of tuples:
      (steerAngle_deg, phaseShift_deg, HPBW, left_3db, right_3db)
    """
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

################################################################################
# 7) MAIN: Print Results and Plot Patterns
################################################################################
def main():
    # User parameters
    N = 40
    freq = 1.2e9
    c = 3e8
    angle_min = -45.0
    angle_max = 45.0
    angle_step = 0.2

    # Determine lobe tiling (no-gap coverage)
    print("\n== Determining no-gap coverage over the domain ==")
    final_lobes = tile_no_gap(N, freq, c, angle_min, angle_max, angle_step=angle_step)
    n_lobes = len(final_lobes)

    print("SteerAngle (deg) | PhaseShift (deg) | HPBW (deg)       | Left_3dB (deg)   | Right_3dB (deg)")
    for (steerA, phaseA, w3, L3, R3) in final_lobes:
        # Print with full precision
        print(f"{steerA:9.3f}      | {phaseA:9.3f}       | {w3:14.6f}   | {L3:14.6f} | {R3:14.6f}")

    # Plot the dB patterns for each lobe
    plt.figure(figsize=(8, 5))
    for (steerDeg, phaseDeg, HPBW, L3dB, R3dB) in final_lobes:
        phi_deg, patt_norm, info = compute_array_factor(
            N, freq, c, angle_min, angle_max,
            phaseDeg, angle_step=angle_step, steer_angle_deg=steerDeg
        )
        patt_dB = 10.0 * np.log10(np.maximum(patt_norm, 1e-12))
        #labelStr = f"S={steerDeg:.1f}°, Ph={phaseDeg:.1f}°, HPBW={HPBW:.6f}°"
        plt.plot(phi_deg, patt_dB)

    plt.axhline(-3.0, color='red', linestyle='--', label='-3 dB line')
    plt.xlim(angle_min, angle_max)
    plt.ylim(-25, 2)
    plt.xlabel("Azimuth Angle (deg)")
    plt.ylabel("Normalized Power (dB)")
    plt.title(f"No-gap Tiling => {n_lobes} Lobes (Main-lobe Patterns)")
    plt.grid(True)
    plt.legend(fontsize="small")
    plt.show()

    # Plot Phase Shift vs. Steering Angle
    steerAngles = [l[0] for l in final_lobes]
    phaseShifts = [l[1] for l in final_lobes]

    plt.figure(figsize=(6, 5))
    plt.plot(steerAngles, phaseShifts, 'bo-', markersize=6, linewidth=1.5)
    plt.xlabel("Steering Angle (deg)")
    plt.ylabel("Global Phase Shift (deg)")
    plt.title("Phase Shift vs. Steering Angle")
    plt.grid(True)
    plt.show()

    print("\n** Done. Two final plots: Radiation Pattern (in dB scale) & Phase Shift vs. Steering Angle. **\n")
    print(f"\nFound {n_lobes} lobes over the domain.\n")

if __name__ == "__main__":
    main()