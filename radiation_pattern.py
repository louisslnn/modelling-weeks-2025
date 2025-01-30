import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

def compute_beamwidth(phi_deg, pattern_norm):
    half_max = np.max(pattern_norm) / 2
    above_half_max = pattern_norm >= half_max
    indices = np.where(above_half_max)[0]
    beamwidth = phi_deg[indices[-1]] - phi_deg[indices[0]] if len(indices) > 1 else 0
    return beamwidth, phi_deg[np.argmax(pattern_norm)], phi_deg[indices[0]], phi_deg[indices[-1]]

def compute_sidelobe_gain(phi_deg, pattern_norm):
    main_lobe_idx = np.argmax(pattern_norm)
    sorted_indices = np.argsort(pattern_norm)[::-1]  # Sort indices by decreasing gain
    secondary_lobe_idx = next(idx for idx in sorted_indices if idx != main_lobe_idx)
    return phi_deg[main_lobe_idx], np.max(pattern_norm), phi_deg[secondary_lobe_idx], pattern_norm[secondary_lobe_idx]

def plot_array_factor_patterns(N, f, delta_phi, do_square, do_db, plot_beamwidth, angle_min=-45, angle_max=45, angle_step=0.1, c=3e8):
    # Wavelength and wavenumber
    L = 4
    lam = c / f
    k = 2.0 * np.pi / lam
    d = L / (N-1)

    # Angle array in degrees and radians
    phi_deg = np.arange(angle_min, angle_max + angle_step, angle_step)
    phi_rad = np.radians(phi_deg)

    # Compute Radiation intensity
    I = np.zeros_like(phi_rad, dtype=complex)
    for i, phi in enumerate(phi_rad):
        beta = k * d * np.sin(phi) + delta_phi
        I[i] = np.sum(np.exp(1j * np.arange(N) * beta))
    
    # Convert to magnitude or magnitude-squared
    pattern = np.abs(I) ** 2 if do_square else np.abs(I)
    pattern_norm = pattern / np.max(pattern)
    
    # Compute beamwidth and sidelobe levels
    beamwidth, max_gain_dir, half_power_left, half_power_right = compute_beamwidth(phi_deg, pattern_norm)
    main_lobe_dir, main_lobe_gain, secondary_lobe_dir, secondary_lobe_gain = compute_sidelobe_gain(phi_deg, pattern_norm)
    
    # Plot in linear scale
    plt.figure(figsize=(10, 7))
    plt.plot(phi_deg, pattern_norm, 'b-', label=f"Max Gain at {main_lobe_dir:.1f}°, HPBW: {abs(half_power_right - half_power_left):.1f}°")
    if plot_beamwidth:
        plt.axvline(main_lobe_dir, color='g', linestyle='--', label=f"Max Gain Direction: {main_lobe_dir:.1f}°")
        plt.axvline(half_power_left, color='r', linestyle='--', label=f"Half Intensity at {half_power_left:.1f}°")
        plt.axvline(half_power_right, color='r', linestyle='--', label=f"Half Intensity at {half_power_right:.1f}°")
        plt.axvline(secondary_lobe_dir, color='m', linestyle='--', label=f"Secondary Lobe at {secondary_lobe_dir:.1f}° ({secondary_lobe_gain:.2f})")
    plt.xlabel("Azimuth Angle (degrees)")
    plt.ylabel("Normalized Value")
    plt.title(f"Radiation Pattern (N={N}, Δϕ={delta_phi:.3f} rad)")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Plot in dB scale if selected
    if do_db:
        plt.figure(figsize=(10, 7))
        pattern_dB = 10.0 * np.log10(pattern_norm + 1e-12)
        plt.plot(phi_deg, pattern_dB, 'r-', label=f"Max Gain at {main_lobe_dir:.1f}° ({10*np.log10(main_lobe_gain+1e-12):.2f} dB), HPBW: {abs(half_power_right - half_power_left):.1f}°")
        if plot_beamwidth:
            plt.axvline(main_lobe_dir, color='g', linestyle='--', label=f"Max Gain Direction: {main_lobe_dir:.1f}°")
            plt.axvline(half_power_left, color='r', linestyle='--', label=f"Half Intensity at {half_power_left:.1f}°")
            plt.axvline(half_power_right, color='r', linestyle='--', label=f"Half Intensity at {half_power_right:.1f}°")
            plt.axvline(secondary_lobe_dir, color='m', linestyle='--', label=f"Secondary Lobe at {secondary_lobe_dir:.1f}° ({10*np.log10(secondary_lobe_gain+1e-12):.2f} dB)")
        plt.xlabel("Azimuth Angle (degrees)")
        plt.ylabel("Normalized (dB)")
        plt.title(f"Radiation Pattern in dB (N={N}, Δϕ={delta_phi:.3f} rad)")
        plt.grid(True)
        plt.legend()
        plt.show()

def update_plot():
    N = int(N_var.get())
    f = float(f_var.get())
    delta_phi = float(delta_phi_var.get())
    do_square = do_square_var.get()
    do_db = do_db_var.get()
    plot_beamwidth = plot_beamwidth_var.get()
    plot_array_factor_patterns(N, f, delta_phi, do_square, do_db, plot_beamwidth)

def create_interface():
    global N_var, f_var, delta_phi_var, do_square_var, do_db_var, plot_beamwidth_var
    root = tk.Tk()
    root.title("Antenna Array Parameters")
    
    ttk.Label(root, text="Number of Elements (N):").grid(column=0, row=0)
    N_var = tk.StringVar(value="40")
    N_entry = ttk.Entry(root, textvariable=N_var)
    N_entry.grid(column=1, row=0)
    
    ttk.Label(root, text="Frequency (Hz):").grid(column=0, row=1)
    f_var = tk.StringVar(value="1.2e9")
    f_entry = ttk.Entry(root, textvariable=f_var)
    f_entry.grid(column=1, row=1)
    
    ttk.Label(root, text="Phase Shift (rad):").grid(column=0, row=2)
    delta_phi_var = tk.StringVar(value="0.0")
    delta_phi_entry = ttk.Entry(root, textvariable=delta_phi_var)
    delta_phi_entry.grid(column=1, row=2)
    
    do_square_var = tk.BooleanVar(value=False)
    do_square_check = ttk.Checkbutton(root, text="Plot |I|²", variable=do_square_var)
    do_square_check.grid(column=0, row=3, columnspan=2)
    
    do_db_var = tk.BooleanVar(value=False)
    do_db_check = ttk.Checkbutton(root, text="Plot in dB", variable=do_db_var)
    do_db_check.grid(column=0, row=4, columnspan=2)
    
    plot_beamwidth_var = tk.BooleanVar(value=False)
    plot_beamwidth_check = ttk.Checkbutton(root, text="Show Beamwidth & Lobes", variable=plot_beamwidth_var)
    plot_beamwidth_check.grid(column=0, row=5, columnspan=2)
    
    plot_button = ttk.Button(root, text="Plot", command=update_plot)
    plot_button.grid(column=0, row=6, columnspan=2)
    
    root.mainloop()

if __name__ == "__main__":
    create_interface()

