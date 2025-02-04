"""
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
    N = 30  # Number of elements
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

def _interp_3db_edge(phi_deg, patt_norm, half_val, i1, i2):
    x1 = phi_deg[i1]
    x2 = phi_deg[i2]
    y1 = patt_norm[i1]
    y2 = patt_norm[i2]
    frac = (half_val - y1) / (y2 - y1 + 1e-20)
    return x1 + (x2 - x1) * frac

def compute_local_3db_beamwidth(phi_deg, patt_norm, steer_deg, search_span=10.0):
    if len(phi_deg) < 2:
        return 0.0, steer_deg, steer_deg, steer_deg

    # Restrict to +/- search_span around the steering angle
    mask = np.abs(phi_deg - steer_deg) <= search_span
    if not np.any(mask):
        return 0.0, steer_deg, steer_deg, steer_deg

    # Slice out the local region
    phi_local = phi_deg[mask]
    patt_local = patt_norm[mask]
    idx_local_peak = np.argmax(patt_local)
    max_val = patt_local[idx_local_peak]
    if max_val < 1e-12:
        return 0.0, steer_deg, steer_deg, steer_deg

    # Map back to global index
    global_indices = np.where(mask)[0]
    idx_peak = global_indices[0] + idx_local_peak
    half_val = 0.5 * max_val

    # Search left from local peak
    iL = idx_peak
    while iL > 0 and patt_norm[iL] >= half_val:
        iL -= 1
    if iL < idx_peak:
        left_3db = _interp_3db_edge(phi_deg, patt_norm, half_val, iL, iL+1)
    else:
        left_3db = phi_deg[idx_peak]

    # Search right
    iR = idx_peak
    while iR < len(patt_norm)-1 and patt_norm[iR] >= half_val:
        iR += 1
    if iR > idx_peak:
        right_3db = _interp_3db_edge(phi_deg, patt_norm, half_val, iR-1, iR)
    else:
        right_3db = phi_deg[idx_peak]

    HPBW = right_3db - left_3db
    angle_of_max = phi_deg[idx_peak]

    return HPBW, angle_of_max, left_3db, right_3db

def compute_3db_beamwidth(phi_deg, patt_norm):
    if len(phi_deg) < 2:
        return 0.0, 0.0, 0.0, 0.0

    idx_peak = np.argmax(patt_norm)
    max_val = patt_norm[idx_peak]
    if max_val < 1e-12:
        return 0.0, 0.0, 0.0, 0.0

    half_val = 0.5 * max_val

    # Search left
    iL = idx_peak
    while iL > 0 and patt_norm[iL] >= half_val:
        iL -= 1
    if iL < idx_peak:
        left_3db = _interp_3db_edge(phi_deg, patt_norm, half_val, iL, iL+1)
    else:
        left_3db = phi_deg[idx_peak]

    # Search right
    iR = idx_peak
    while iR < len(patt_norm)-1 and patt_norm[iR] >= half_val:
        iR += 1
    if iR > idx_peak:
        right_3db = _interp_3db_edge(phi_deg, patt_norm, half_val, iR-1, iR)
    else:
        right_3db = phi_deg[idx_peak]

    HPBW = right_3db - left_3db
    angle_of_max = phi_deg[idx_peak]

    return HPBW, angle_of_max, left_3db, right_3db

def compute_array_factor_intensity(
    N, f, c,
    angle_min, angle_max,
    delta_phi_deg,
    angle_step=0.01,
    window_type="uniform",
    ripple_db=30.0,
    steering_angle_deg=None,
    local_search_span=10.0
):
    if N < 2:
        # Degenerate: just return a flat pattern
        phi_deg = np.arange(angle_min, angle_max + angle_step, angle_step)
        patt_norm = np.ones_like(phi_deg)
        info = {
            'max_gain': 1.0,
            'direction_of_max': 0.0,
            'HPBW': 0.0,
            'left_3db': 0.0,
            'right_3db': 0.0,
            'peak_side_lobe': 0.0
        }

        return phi_deg, patt_norm, info

    # Double element spacing
    d = 2.0 * (4.0 / (N - 1))
    lam = c / f
    k = 2.0 * np.pi / lam

    # Amplitude window
    w = amplitude_window(N, window_type=window_type, ripple=ripple_db)

    # Angle array
    phi_deg = np.arange(angle_min, angle_max + angle_step, angle_step)
    phi_rad = np.radians(phi_deg)

    # Steering phase
    delta_phi_rads = np.radians(delta_phi_deg)

    # Compute array factor
    AF = np.zeros_like(phi_rad, dtype=complex)
    for i, phi in enumerate(phi_rad):
        beta = k * d * np.sin(phi) + delta_phi_rads
        accum = 0.+0.j
        for n in range(N):
            accum += w[n] * np.exp(1j * n * beta)
        AF[i] = accum

    pattern = np.abs(AF)**2
    peak_val = max(np.max(pattern), 1e-12)
    pattern_norm = pattern / peak_val

    # Global direction of max
    idx_max = np.argmax(pattern)
    direction_of_max = phi_deg[idx_max]

    # -3 dB beamwidth: local if steering specified
    if steering_angle_deg is not None:
        HPBW, main_ang, left_3db, right_3db = compute_local_3db_beamwidth(
            phi_deg, pattern_norm,
            steer_deg=steering_angle_deg,
            search_span=local_search_span
        )
    else:
        HPBW, main_ang, left_3db, right_3db = compute_3db_beamwidth(phi_deg, pattern_norm)

    # Quick side‐lobe estimate
    side_mask = np.abs(phi_deg - direction_of_max) > 2.0
    pattern_side = pattern_norm[side_mask] if side_mask.any() else [0.0]
    peak_side_lobe = np.max(pattern_side)

    info = {
        'max_gain': peak_val,
        'direction_of_max': direction_of_max,
        'HPBW': HPBW,
        'left_3db': left_3db,
        'right_3db': right_3db,
        'peak_side_lobe': peak_side_lobe
    }
    return phi_deg, pattern_norm, info

def compute_array_factor_by_angle(
    N, f, c,
    angle_min, angle_max,
    steering_angle_deg,
    angle_step=0.01,
    window_type="uniform",
    ripple_db=30.0
):
    lam = c / f
    d   = 2.0 * (4.0 / (N - 1))
    k   = 2.0 * np.pi / lam

    # Phase shift in degrees
    delta_phi_deg = -np.degrees(k * d * np.sin(np.radians(steering_angle_deg)))

    return compute_array_factor_intensity(
        N, f, c,
        angle_min, angle_max,
        delta_phi_deg,
        angle_step=angle_step,
        window_type=window_type,
        ripple_db=ripple_db,
        steering_angle_deg=steering_angle_deg,
        local_search_span=10.0
    )

def find_center_for_left_edge(
    left_edge_target,
    N, f, c,
    angle_min, angle_max,
    end_angle,
    window_type="uniform",
    ripple_db=30.0,
    angle_step=0.01,
    max_iter=100000
):
    low = angle_min
    high = angle_max

    for _ in range(max_iter):
        mid = 0.5*(low + high)
        print(f"mid = {mid}")

        # Compute the array factor for this steering
        _, _, info = compute_array_factor_by_angle(
            N, f, c, angle_min, angle_max,
            steering_angle_deg=mid,
            angle_step=angle_step,
            window_type=window_type,
            ripple_db=ripple_db
        )
        this_left_3db = info['left_3db']
        print("  left_edge_target =", left_edge_target)
        print("  this_left_3db    =", this_left_3db)

        # If the left -3 dB is still to the right of our target, we steer less negatively
        if this_left_3db > left_edge_target:
            high = mid
        else:
            low = mid

        # Check for convergence
        if (abs(this_left_3db - left_edge_target) < 1e-5) or ((high - low) < 1e-5):
            break
        print(f"  -> low = {low}, high = {high}")

    return 0.5*(low + high)

def find_optimal_scan_bins(
    angle_min, angle_max,
    end_angle,
    N, frequency, c=3e8,
    window_type="uniform",
    ripple_db=30.0,
    angle_step=0.01
):
    if N < 2 or angle_min >= end_angle:
        return [0.0]

    bins = []
    coverage_left = angle_min

    while coverage_left < end_angle:

        # 1) Solve for a steering angle that sets the left -3 dB at coverage_left
        center = find_center_for_left_edge(
            coverage_left, N, frequency, c,
            angle_min, angle_max,
            end_angle,
            window_type=window_type,
            ripple_db=ripple_db,
            angle_step=angle_step
        )

        # 1a) If new center is essentially the same as the last center, break
        if bins and abs(center - bins[-1]) < 1e-4:
            print("No new steering found (duplicate). Stopping.")
            break

        # 2) Find the actual coverage of that beam
        _, _, info = compute_array_factor_by_angle(
            N, frequency, c,
            angle_min, angle_max,
            steering_angle_deg=center,
            angle_step=angle_step,
            window_type=window_type,
            ripple_db=ripple_db
        )
        left_3db = info['left_3db']
        right_3db = info['right_3db']

        # 2a) If we make no progress, stop
        if right_3db <= coverage_left + 1e-5:
            print("No coverage progress. Stopping.")
            break

        # Accept this beam
        bins.append(center)

        # 3) Advance coverage to right_3db
        coverage_left = right_3db

    return bins

# ---------------------- MAIN DEMO (Single pass) ----------------------

if __name__ == "__main__":
    angle_min   = -45
    angle_max   =  45
    end_angle   =  45   # We'll tile from -45 .. +45 in one pass
    N           = 40    # number of elements
    frequency   = 1.2e9 # 1.2 GHz
    c           = 3e8
    window_type = "uniform"
    ripple_db   = 30.0

    print("\n=== Finding beam centers from -45..+45 in one pass ===")

    beam_centers = find_optimal_scan_bins(
        angle_min, angle_max,
        end_angle,
        N, frequency, c=c,
        window_type=window_type,
        ripple_db=ripple_db,
        angle_step=0.01
    )
    print("\nFinal beam centers:", beam_centers)
    print("Number of beams =", len(beam_centers))

    # -- (1) Plot each beam pattern in dB --
    plt.figure(figsize=(8,5))
    plot_step = 0.1
    for center in beam_centers:
        phi_deg, patt_norm, info = compute_array_factor_by_angle(
            N, frequency, c,
            angle_min, angle_max,
            steering_angle_deg=center,
            angle_step=plot_step,
            window_type=window_type,
            ripple_db=ripple_db
        )
        patt_dB = 10.0 * np.log10(np.maximum(patt_norm, 1e-12))
        plt.plot(phi_deg, patt_dB, label=f"{center:.1f}°")

    # Overplot the -3 dB line
    plt.plot([angle_min, angle_max], [-3, -3], "r")
    plt.plot([angle_min, angle_max], [0,  0],  "r")
    plt.ylim(-15, 3)
    plt.xlabel("Azimuth angle (deg)")
    plt.ylabel("Normalized |AF|^2 (dB)")
    plt.title("Array Patterns with -3 dB Tiling")
    plt.grid(True)
    plt.legend(loc="upper right", fontsize="small")
    plt.show()

    # -- (2) Show each beams -3 dB interval --
    plt.figure(figsize=(7,5))
    for i, center in enumerate(beam_centers):
        _, _, info = compute_array_factor_by_angle(
            N, frequency, c,
            angle_min, angle_max,
            steering_angle_deg=center,
            angle_step=0.01,
            window_type=window_type,
            ripple_db=ripple_db
        )
        left_3db, right_3db = info['left_3db'], info['right_3db']
        plt.plot([left_3db, right_3db], [i, i], 'b-', lw=2)
        plt.plot(center, i, 'ro')
    plt.xlabel("Azimuth angle (deg)")
    plt.ylabel("Beam index")
    plt.title("3 dB Coverage of Each Beam")
    plt.grid(True)
    plt.show()

    # -- (3) Cartesian Fan Plot --
    plt.figure(figsize=(6,6))
    plt.title("Cartesian Fan (edges only, single coverage)")

    max_radius = 10.0
    for center in beam_centers:
        _, _, info = compute_array_factor_by_angle(
            N, frequency, c,
            angle_min, angle_max,
            steering_angle_deg=center,
            angle_step=0.01,
            window_type=window_type,
            ripple_db=ripple_db

        )
        left_3db, right_3db = info["left_3db"], info["right_3db"]
        for edge_deg in [left_3db, right_3db]:
            theta = np.radians(edge_deg)
            x = max_radius * np.cos(theta)
            y = max_radius * np.sin(theta)
            plt.plot([0, x], [0, y], 'b-', lw=1)

        # Mark center in red (dashed)
        theta_c = np.radians(center)
        x_c = max_radius * np.cos(theta_c)
        y_c = max_radius * np.sin(theta_c)
        plt.plot([0, x_c], [0, y_c], 'r--', lw=1)

    plt.xlim(-max_radius, max_radius)
    plt.ylim(-max_radius, max_radius)
    plt.axhline(0, color='k', lw=0.5)
    plt.axvline(0, color='k', lw=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    # -- (4) Polar Plot --
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_title("Polar Plot of the Full Lobe (Single Pass)")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(-45)
    ax.set_thetamax(45)
    rmax = 0.0
    rmin = -40.0
    ax.set_rlim([rmin, rmax])
    rticks = [0, -10, -20, -30, -40]
    ax.set_yticks(rticks)
    ax.set_yticklabels([f"{v} dB" for v in rticks])

    for center in beam_centers:
        phi_deg, patt_norm, info = compute_array_factor_by_angle(
            N, frequency, c,
            angle_min, angle_max,
            steering_angle_deg=center,
            angle_step=0.5,
            window_type=window_type,
            ripple_db=ripple_db
        )
        theta = np.radians(phi_deg)
        patt_dB = 10.0*np.log10(np.maximum(patt_norm, 1e-12))
        patt_dB = np.clip(patt_dB, rmin, rmax)
        ax.plot(theta, patt_dB, label=f"{center:.1f}°")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3,1.1))
    plt.show()

    # -- (5) Print a Table of Beams --
    lam = c / frequency
    d   = 2.0 * (4.0 / (N - 1))
    k   = 2.0 * np.pi / lam
    print("\nBeam table:")
    print("Idx | Center(deg) | PhaseShift(deg) | 3dB Coverage [Left, Right]")

    for i, center in enumerate(beam_centers):
        phase_shift_deg = -np.degrees(k * d * np.sin(np.radians(center)))
        _, _, info = compute_array_factor_by_angle(
            N, frequency, c,
            angle_min, angle_max,
            steering_angle_deg=center,
            angle_step=0.01,
            window_type=window_type,
            ripple_db=ripple_db
        )

        left_3db = info['left_3db']
        right_3db = info['right_3db']
        print(f"{i:3d} | {center:10.3f} | {phase_shift_deg:15.3f}"
              f" | [{left_3db:6.2f}, {right_3db:6.2f}]")
"""

#############################################
# Use TkAgg for normal interactive Matplotlib windows on Windows
#############################################
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
    full_phi_deg = np.linspace(-90, 90, 10000)
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