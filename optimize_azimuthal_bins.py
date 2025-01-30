import numpy as np
import matplotlib.pyplot as plt

def amplitude_window(N, window_type="uniform", ripple=30):
    if N < 1:
        return np.array([])

    wtype = window_type.lower()
    if wtype == "uniform":
        w = np.ones(N)
    elif wtype == "hamming":
        w = np.hamming(N)
    elif wtype == "chebyshev":
        w = np.chebwin(N, at=ripple)
    else:
        w = np.ones(N)

    return w

def compute_3db_beamwidth(phi_deg, pattern_norm):
    if len(phi_deg) < 2:
        return 0.0, 0.0, 0.0, 0.0

    max_val = np.max(pattern_norm)
    if max_val <= 1e-12:
        return 0.0, 0.0, 0.0, 0.0

    half_val = 0.5 * max_val
    # Indices where pattern >= half of peak
    above_mask = (pattern_norm >= half_val)
    idx = np.where(above_mask)[0]

    if len(idx) < 2:
        # If no contiguous region found above -3 dB,
        # fallback: everything collapses to angle_of_max
        angle_of_max = phi_deg[np.argmax(pattern_norm)]
        return 0.0, angle_of_max, angle_of_max, angle_of_max

    left_3db_edge = phi_deg[idx[0]]
    right_3db_edge = phi_deg[idx[-1]]
    bw = right_3db_edge - left_3db_edge
    angle_of_max = phi_deg[np.argmax(pattern_norm)]
    return bw, angle_of_max, left_3db_edge, right_3db_edge

def compute_array_factor_intensity(
    N, f, c,
    angle_min, angle_max, angle_step,
    delta_phi_deg,
    window_type="uniform",
    ripple_db=30.0
):
    if N < 2:
        # Degenerate case
        phi_deg = np.arange(angle_min, angle_max + angle_step, angle_step)
        pattern_norm = np.ones_like(phi_deg)
        info = {
            'max_gain': 1.0,
            'direction_of_max': 0.0,
            'HPBW': 0.0,
            'left_3db': 0.0,
            'right_3db': 0.0,
            'peak_side_lobe': 0.0
        }
        return phi_deg, pattern_norm, info

    # Spacing and wavenumber
    d = 4.0 / (N - 1)
    lam = c / f
    k = 2.0 * np.pi / lam

    delta_phi_rads = np.radians(delta_phi_deg)
    w = amplitude_window(N, window_type=window_type, ripple=ripple_db)

    phi_deg = np.arange(angle_min, angle_max + angle_step, angle_step)
    phi_rad = np.radians(phi_deg)

    AF = np.zeros_like(phi_rad, dtype=complex)
    for i, phi in enumerate(phi_rad):
        beta = k * d * np.sin(phi) + delta_phi_rads
        s = 0.0j
        for n in range(N):
            s += w[n] * np.exp(1j * n * beta)
        AF[i] = s

    pattern = np.abs(AF)**2
    peak_val = np.max(pattern) if len(pattern) > 0 else 1e-12
    if peak_val < 1e-12:
        peak_val = 1e-12
    pattern_norm = pattern / peak_val

    # Some lobe info
    max_gain = peak_val
    idx_max = np.argmax(pattern)
    direction_of_max = phi_deg[idx_max]
    HPBW, main_ang, left_3db, right_3db = compute_3db_beamwidth(phi_deg, pattern_norm)

    # Very rough side-lobe check: skip ±2 deg around main-lobe max
    side_mask = np.ones_like(phi_deg, dtype=bool)
    side_mask &= (np.abs(phi_deg - direction_of_max) > 2.0)
    pattern_side = pattern_norm[side_mask]
    peak_side_lobe = np.max(pattern_side) if len(pattern_side) else 0.0

    info = {
        'max_gain': max_gain,
        'direction_of_max': direction_of_max,
        'HPBW': HPBW,
        'left_3db': left_3db,
        'right_3db': right_3db,
        'peak_side_lobe': peak_side_lobe,
    }
    return phi_deg, pattern_norm, info

def compute_array_factor_by_angle(
    N, f, c,
    angle_min, angle_max, angle_step,
    steering_angle_deg,
    window_type="uniform",
    ripple_db=30.0
):
    lam = c / f
    d = 4.0 / (N - 1)
    k = 2.0 * np.pi / lam
    # Convert the steering angle (in deg) to the needed phase shift in degrees.
    delta_phi_deg = - np.degrees(k * d * np.sin(np.radians(steering_angle_deg)))

    return compute_array_factor_intensity(
        N, f, c,
        angle_min, angle_max, angle_step,
        delta_phi_deg,
        window_type=window_type,
        ripple_db=ripple_db
    )

def find_mainlobe_for_center(
    guess_center_deg,
    N, frequency, c,
    angle_min, angle_max, angle_step,
    window_type, ripple_db,
    max_iter=5
):
    center = guess_center_deg
    for _ in range(max_iter):
        _, _, info = compute_array_factor_by_angle(
            N, frequency, c,
            angle_min, angle_max, angle_step,
            center,
            window_type=window_type,
            ripple_db=ripple_db
        )
        dom = info["direction_of_max"]
        if abs(dom - center) < 0.1:
            # Close enough
            break
        center = dom

    # Return final coverage from that center
    phi_deg, patt_norm, info = compute_array_factor_by_angle(
        N, frequency, c,
        angle_min, angle_max, angle_step,
        center,
        window_type=window_type,
        ripple_db=ripple_db
    )
    return center, info

def find_optimal_scan_bins(
    angle_min, angle_max,
    N, frequency, c=3e8,
    window_type="uniform",
    ripple_db=30.0,
    angle_step=0.1
):
    if N < 2 or angle_min >= angle_max:
        return [0.0]

    bins = []
    coverage_left = angle_min
    EPS = 1e-3

    while coverage_left < angle_max - EPS:
        # For each beam, guess some center a couple deg to the right
        guess = coverage_left + 2.0

        # Snap the main lobe to that guess
        center, info = find_mainlobe_for_center(
            guess, N, frequency, c,
            angle_min, angle_max, angle_step,
            window_type, ripple_db
        )
        left_3db = info["left_3db"]
        right_3db = info["right_3db"]

        # Local search around that center so left_3db ~ coverage_left
        def coverage_gap(beam_center):
            _, info2 = find_mainlobe_for_center(
                beam_center, N, frequency, c,
                angle_min, angle_max, angle_step,
                window_type, ripple_db
            )
            return info2["left_3db"] - coverage_left

        best_center = center
        best_gap = abs(left_3db - coverage_left)

        test_angles = np.arange(center - 2.0, center + 2.0, 0.05)
        for a in test_angles:
            gap_val = coverage_gap(a)
            if abs(gap_val) < best_gap:
                best_gap = abs(gap_val)
                best_center = a

        # Recompute coverage from that best_center
        final_center, infoC = find_mainlobe_for_center(
            best_center, N, frequency, c,
            angle_min, angle_max, angle_step,
            window_type, ripple_db
        )
        left_3db = infoC["left_3db"]
        right_3db = infoC["right_3db"]

        # If left_3db > coverage_left, shift coverage_left
        if left_3db > coverage_left + EPS:
            coverage_left = left_3db

        bins.append(final_center)
        coverage_left = right_3db

    return bins

# ------------------------- MAIN DEMO -------------------------
if __name__ == "__main__":
    # Parameters
    angle_min = -45
    angle_max = 45
    N = 40  # Number of elements
    frequency = 1.2e9  # 1.2 GHz
    c = 3e8
    window_type = "uniform"  # Could be 'uniform','chebyshev','hamming'
    ripple_db = 30.0  # Used if Chebyshev
    angle_step = 0.1

    # 1) Get the list of beam centers from -3 dB tiling
    beam_centers = find_optimal_scan_bins(
        angle_min, angle_max,
        N, frequency, c=c,
        window_type=window_type,
        ripple_db=ripple_db,
        angle_step=angle_step
    )
    print("Beam centers:", beam_centers)
    print(f"Number of beams = {len(beam_centers)}")

    # 2) Plot each beam pattern in dB
    plt.figure(figsize=(8, 5))
    for center in beam_centers:
        phi_deg, patt_norm, info = compute_array_factor_by_angle(
            N, frequency, c,
            angle_min, angle_max, angle_step,
            center,
            window_type=window_type,
            ripple_db=ripple_db
        )
        patt_dB = 10.0 * np.log10(np.maximum(patt_norm, 1e-12))
        plt.plot(phi_deg, patt_dB, label=f"{center:.1f}°")
        plt.plot([-45, 45], [-3, -3], "r")
        plt.ylim(-15, 3)
    
    plt.xlabel("Azimuth angle (deg)")
    plt.ylabel("Normalized |AF|^2 (dB)")
    plt.title("Array Patterns with -3 dB Tiling")
    plt.grid(True)
    plt.legend(loc="upper right", fontsize="small")
    plt.show()

    # 3) Show the -3 dB coverage intervals for each beam
    plt.figure(figsize=(7, 5))
    for i, center in enumerate(beam_centers):
        _, _, info = compute_array_factor_by_angle(
            N, frequency, c,
            angle_min, angle_max, angle_step,
            center,
            window_type=window_type,
            ripple_db=ripple_db
        )
        left_3db = info['left_3db']
        right_3db = info['right_3db']
        y_val = i
        plt.plot([left_3db, right_3db], [y_val, y_val], 'b-', lw=2)
        plt.plot(center, y_val, 'ro')

    plt.xlabel("Azimuth angle (deg)")
    plt.ylabel("Beam index")
    plt.title("3 dB Coverage of Each Beam")
    plt.grid(True)
    plt.show()

