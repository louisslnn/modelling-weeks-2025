"""
import numpy as np
import matplotlib.pyplot as plt

def plot_array_factor_patterns(
    N=40,            # Number of elements
    f=1.2e9,         # Frequency (Hz)
    c=3e8,           # Speed of light (m/s)
    angle_min=-45,   # Min angle in degrees
    angle_max=45,    # Max angle in degrees
    angle_step=0.1,  # Angular resolution for plotting
    delta_phi=5,   # Additional per-element phase shift (radians)
    do_square=False  # If True, plot |I|^2; if False, plot |I|
):
    # Wavelength and wavenumber
    lam = c / f
    k = 2.0 * np.pi / lam

    # Element spacing (for typical half-wavelength design)
    d = lam / 2.0

    # Angle array in degrees and radians
    phi_deg = np.arange(angle_min, angle_max + angle_step, angle_step)
    phi_rad = np.radians(phi_deg)

    # Prepare Radiation intensity storage
    I = np.zeros_like(phi_rad, dtype=complex)

    # Compute Radiation intensity
    for i, phi in enumerate(phi_rad):

        # Phase shift due to geometry + extra per-element shift
        beta = k * d * np.sin(phi) + delta_phi
        # Summation of contributions from each of N elements
        # n goes from 0 to N-1
        I_sum = 0.0 + 0j
        for n in range(N):
            I_sum += np.exp(1j * n * beta)
        I[i] = I_sum

    # Convert to magnitude or magnitude-squared
    if do_square:
        pattern = np.abs(I)**2
        label_str = r"$|I|^2$ (Intensity)"
    else:
        pattern = np.abs(I)
        label_str = r"$|I|$ (Field Magnitude)"

    # Normalize so the maximum is 1 (for easier comparison)
    pattern_norm = pattern / np.max(pattern)

    # Plot (in linear scale)
    plt.figure(figsize=(7,5))
    plt.plot(phi_deg, pattern_norm, 'b-', label=label_str + " (Linear Scale)")
    plt.xlabel("Azimuth Angle (degrees)")
    plt.ylabel("Normalized Value")
    plt.title(f"Radiation intensity, N={N}, ∆ϕ={delta_phi:.3f} rad")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Optionally, also plot in dB
    pattern_dB = 10.0 * np.log10(pattern_norm + 1e-12)  # add small offset to avoid log(0)
    plt.figure(figsize=(7,5))
    plt.plot(phi_deg, pattern_dB, 'r-', label=label_str + " (dB Scale)")
    plt.xlabel("Azimuth Angle (degrees)")
    plt.ylabel("Normalized (dB)")
    plt.title(f"Radiation intensity in dB, N={N}, ∆ϕ={delta_phi:.3f} rad")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":

    # --- EXAMPLES OF USAGE ---
    # 1) Step 1: No extra phase shift, plot field amplitude (-45 to 45 deg)
    plot_array_factor_patterns(N=40, delta_phi=0.0, do_square=False)

    # 2) Step 1 (alternative): No extra phase shift, but plot intensity (|I|^2)
    plot_array_factor_patterns(N=40, delta_phi=0.0, do_square=True)

    # 3) Step 2: Introduce a phase shift to steer, e.g. delta_phi = -kd*sin(20°)
    #    We'll compute it for a 20° steering angle as an example:
    #    delta_phi_steer = - (2*pi/lambda)*(lambda/2)*sin(20°) = - pi*sin(20°).
    #    For f=1.2e9, the lam is ~0.25, d=0.125 => k*d = pi
    #    so delta_phi_steer ~ - pi * sin(20°)
    import math
    delta_phi_steer = -math.pi * math.sin(math.radians(20.0))
    plot_array_factor_patterns(N=40, delta_phi=delta_phi_steer, do_square=False)

    # PREPARING FOR STEP 4:
    # Example: we want to systematically plot beams for scan angles from -45 to +45 in increments of 15 deg
    # (just demonstrating code that might be used in scanning).
    angles_to_scan_deg = range(-45, 46, 15)
    plt.figure(figsize=(8,6))
    for angle_scan in angles_to_scan_deg:
        # Compute delta_phi for steering
        beta_scan = -math.pi * math.sin(math.radians(angle_scan))  # (k*d ~ pi for half-wave spacing)
        # Evaluate Radiation intensity just for the plotting range
        # We'll re-use the above function but return the data or do something similar
        # For simplicity, call the function that draws it, but keep it on the same figure:
        lam = 3e8 / 1.2e9
        k = 2.0*np.pi/lam
        d = lam/2.0
        phi_deg = np.arange(-45, 45.1, 0.2)
        phi_rad = np.radians(phi_deg)
        I = []
        for phi in phi_rad:
            I_sum = 0.0+0j
            val = k*d*np.sin(phi) + beta_scan
            for n in range(40):
                I_sum += np.exp(1j * n * val)
            I.append(I_sum)

        I = np.array(I)
        pattern = np.abs(I)**2
        pattern /= np.max(pattern)
        plt.plot(phi_deg, 10*np.log10(pattern), label=f"Steer={angle_scan} deg")

    plt.xlabel("Azimuth (deg)")
    plt.ylabel("Normalized Intensity (dB)")
    plt.title("Step 4 Prep: Multiple Steering Angles from -45 to +45")
    plt.grid(True)
    plt.legend()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

def plot_array_factor_patterns(N, f, delta_phi, do_square, do_db, angle_min=-45, angle_max=45, angle_step=0.1, c=3e8):
    # Wavelength and wavenumber
    lam = c / f
    k = 2.0 * np.pi / lam
    d = lam / 2.0

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
    
    # Plot in linear scale
    plt.figure(figsize=(7, 5))
    plt.plot(phi_deg, pattern_norm, 'b-', label=r"$|I|^2$" if do_square else r"$|I|$")
    plt.xlabel("Azimuth Angle (degrees)")
    plt.ylabel("Normalized Value")
    plt.title(f"Radiation Pattern (N={N}, Δϕ={delta_phi:.3f} rad)")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Plot in dB scale if selected
    if do_db:
        plt.figure(figsize=(7, 5))
        pattern_dB = 10.0 * np.log10(pattern_norm + 1e-12)
        plt.plot(phi_deg, pattern_dB, 'r-', label=r"$|I|^2$ dB" if do_square else r"$|I|$ dB")
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
    plot_array_factor_patterns(N, f, delta_phi, do_square, do_db)

def create_interface():
    global N_var, f_var, delta_phi_var, do_square_var, do_db_var
    root = tk.Tk()
    root.title("Antenna Array Parameters")
    
    ttk.Label(root, text="Number of Elements:").grid(column=0, row=0)
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
    do_square_check.grid(column=0, row=4, columnspan=2)
    
    do_db_var = tk.BooleanVar(value=False)
    do_db_check = ttk.Checkbutton(root, text="Plot in dB", variable=do_db_var)
    do_db_check.grid(column=0, row=5, columnspan=2)
    
    plot_button = ttk.Button(root, text="Plot", command=update_plot)
    plot_button.grid(column=0, row=6, columnspan=2)
    
    root.mainloop()

if __name__ == "__main__":
    create_interface()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

# Constants
k = 25.13  # Wavenumber
initial_delta_phi = 30  # Initial phase shift in degrees

d = 0.1  # Antenna element spacing (normalized to wavelength)

# Function to calculate the radiation pattern
def calculate_pattern(delta_phi):
    x_deg = np.linspace(-45, 45, 400)  # Angle range
    x_rad = np.radians(x_deg)  # Convert to radians

    numerator = np.sin((k * d * np.sin(x_rad)) * 40 / 2)
    denominator = np.sin((k * d * np.sin(x_rad)) / 2)
    denominator[denominator == 0] = 1e-9  # Avoid division by zero

    y1 = np.abs(numerator / denominator)  # Without phase shift

    delta_phi_rad = np.radians(delta_phi)
    numerator2 = np.sin(((k * d * np.sin(x_rad)) + delta_phi_rad) * 40 / 2)
    denominator2 = np.sin(((k * d * np.sin(x_rad)) + delta_phi_rad) / 2)
    denominator2[denominator2 == 0] = 1e-9  # Avoid division by zero

    y2 = np.abs(numerator2 / denominator2)  # With phase shift

    return x_deg, y1, y2

# Function to update the plot
def update(val):
    delta_phi = slider.val
    x_deg, y1, y2 = calculate_pattern(delta_phi)

    line1.set_ydata(y1)
    line2.set_ydata(y2)

    # Calculate max gain and beamwidth
    max_gain = np.max(y2)
    direction_max_gain = x_deg[np.argmax(y2)]
    half_max = max_gain / 2
    beamwidth = np.sum(y2 >= half_max) * (x_deg[1] - x_deg[0])

    # Update text
    text_box.set_text(f"Max Gain: {max_gain:.2f}\nDirection: {direction_max_gain:.1f}°\nBeamwidth: {beamwidth:.1f}°")

    fig.canvas.draw_idle()

# Initial plot setup
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.25, right=0.75, bottom=0.25)
x_deg, y1, y2 = calculate_pattern(initial_delta_phi)
line1, = ax.plot(x_deg, y1, label='Step 1', color='m')
line2, = ax.plot(x_deg, y2, label='Step 2', color='c', linestyle="--")

# Configure plot
ax.set_title('Radiation Pattern')
ax.set_xlabel('Azimuthal Angle (degrees)')
ax.set_ylabel('Normalized Intensity')
ax.legend()
ax.grid(True)

# Slider for delta_phi
ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Delta Phi', 0, 360, valinit=initial_delta_phi)
slider.on_changed(update)

# Text box for displaying info
text_box = ax.text(1.05, 0.5, '', transform=ax.transAxes, fontsize=12, verticalalignment='center')

# Initial update to display values
update(initial_delta_phi)

plt.show()
"""

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

