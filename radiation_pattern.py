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
    d = 2*L / (N-1)

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
"""
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

###############################################################################
# 1) SNIPPET: HPBW vs Phase with Angle Range -90..+90
###############################################################################
def snippet_compute_beamwidth(N, d, wavelength, delta_phi_rad):
    """
    Scans angles from -90..90 (10000 points), computes the array factor,
    and finds the half-power beamwidth (HPBW) by locating the first and last
    angles where the normalized power exceeds 0.5.
   
    This vectorized version speeds up the calculation.
    """
    k = 2.0 * np.pi / wavelength
    phi_deg = np.linspace(-90, 90, 10000)  # Angle grid in degrees.
    phi_rad = np.radians(phi_deg)           # Convert to radians.
   
    # Compute the array factor for all angles at once via broadcasting.
    n = np.arange(N)[:, None]               # Shape (N,1)
    beta = k * d * np.sin(phi_rad) + delta_phi_rad  
    I = np.sum(np.exp(1j * n * beta), axis=0)
   
    # Normalize the power pattern.
    pattern = np.abs(I)**2
    pattern /= (np.max(pattern) + 1e-12)
   
    # Find indices where the pattern exceeds 0.5.
    half_val = 0.5
    idx = np.where(pattern >= half_val)[0]
   
    if len(idx) > 1:
        bw_deg = phi_deg[idx[-1]] - phi_deg[idx[0]]
    else:
        bw_deg = 0.0
       
    return bw_deg


###############################################################################
# 2) Full Summation for -90..+90 => Measure HPBW, Main Beam, and Secondary Lobe
###############################################################################
def compute_pattern_full(N=40, freq_hz=1.2e9, delta_phi=0.0, array_len=4.0, do_square=True):
    """
    Computes the radiation pattern over -90°..+90° (4000 steps) and returns:
      - full_phi_deg: angle array (deg)
      - pattern_norm: normalized power pattern (peak=1)
      - hpbw: half-power beamwidth (deg)
      - left_edge, right_edge: angles (deg) where the pattern crosses 0.5
      - main_beam_direction: angle (deg) of the maximum (main) beam
      - secondary_gain: normalized gain of the highest sidelobe (outside the main lobe)
      - secondary_angle: angle (deg) corresponding to the secondary lobe gain

    **Secondary Lobe Definition:**  
      We exclude a margin (2°) around the main lobe (defined by left_edge and right_edge)
      and search for the highest gain outside that expanded region.
    """
    c = 3e8
    lam = c / freq_hz
    k = 2.0 * np.pi / lam
    d = array_len / (N - 1) if N > 1 else 0.0

    full_phi_deg = np.linspace(-90, 90, 4000)
    full_phi_rad = np.radians(full_phi_deg)

    I = np.zeros_like(full_phi_rad, dtype=complex)
    for i, phi in enumerate(full_phi_rad):
        beta = k * d * np.sin(phi) + delta_phi
        I[i] = np.sum(np.exp(1j * np.arange(N) * beta))

    pattern = np.abs(I)**2
    pattern /= (np.max(pattern) + 1e-12)

    # Determine HPBW based on half-power threshold.
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

    # Main beam direction is the angle of maximum normalized power.
    peak_index = np.argmax(pattern)
    main_beam_direction = full_phi_deg[peak_index]

    # Define a margin (in degrees) around the main lobe to exclude.
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

    # Optionally square-root pattern if do_square is False (not used here).
    if not do_square:
        pattern = np.sqrt(pattern)

    return (full_phi_deg, pattern, hpbw, left_edge, right_edge,
            main_beam_direction, secondary_gain, secondary_angle)


###############################################################################
# 3) The Multi-Plot Tkinter App
###############################################################################
class LinearArrayApp:
    """
    A GUI for plotting the radiation pattern of a linear array.
      - The full domain [-90°..+90°] is always plotted.
      - HPBW is measured from this domain.
      - The initial zoom is set to [-45°, 45°] (can be changed).
      - "Dezoom" and "Rezoom" functions adjust the x-axis limits.
      - "Plot HPBW vs Phase" shows how HPBW changes with phase shift.
      - Detailed beam parameters (main beam direction and secondary lobe gain/direction)
        are shown in the Info section when requested.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Full-range Summation + Initial Zoom + Dezoom/Rezoom")

        self.plots_dict = {}

        # User input variables.
        self.N_var = tk.StringVar(value="40")
        self.freq_var = tk.StringVar(value="1.2e9")
        # Phase shift now input in degrees.
        self.delta_var = tk.StringVar(value="0.0")
        self.amin_var = tk.StringVar(value="-45.0")  # initial x-limits after plot
        self.amax_var = tk.StringVar(value="45.0")   # can 'dezoom' to [-90..90]
        self.len_var = tk.StringVar(value="4.0")
        self.scale_var = tk.StringVar(value="linear")

        self.plot_counter = 0

        input_frame = ttk.Frame(root, padding="5 5 5 5")
        input_frame.grid(row=0, column=0, sticky="nw")

        plot_frame = ttk.Frame(root, padding="5 5 5 5")
        plot_frame.grid(row=0, column=1, sticky="nsew")

        toolbar_frame = ttk.Frame(root)
        toolbar_frame.grid(row=1, column=1, sticky="ew")

        info_frame = ttk.Frame(root, padding="5 5 5 5")
        info_frame.grid(row=1, column=0, sticky="nw")

        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        r = 0
        ttk.Label(input_frame, text="Number of Elements:").grid(row=r, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.N_var, width=8).grid(row=r, column=1, pady=2)

        r += 1
        ttk.Label(input_frame, text="Frequency (Hz):").grid(row=r, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.freq_var, width=8).grid(row=r, column=1, pady=2)

        r += 1
        # Phase shift (deg)
        ttk.Label(input_frame, text="Phase Shift δφ (deg):").grid(row=r, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.delta_var, width=8).grid(row=r, column=1, pady=2)

        r += 1
        ttk.Label(input_frame, text="Init Zoom Min (deg):").grid(row=r, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.amin_var, width=8).grid(row=r, column=1, pady=2)

        r += 1
        ttk.Label(input_frame, text="Init Zoom Max (deg):").grid(row=r, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.amax_var, width=8).grid(row=r, column=1, pady=2)

        r += 1
        ttk.Label(input_frame, text="Array Length (m):").grid(row=r, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.len_var, width=8).grid(row=r, column=1, pady=2)

        r += 1
        ttk.Label(input_frame, text="Plot Scale:").grid(row=r, column=0, sticky="w")
        ttk.Radiobutton(input_frame, text="Linear", variable=self.scale_var, value="linear").grid(row=r, column=1, sticky="w")
        r += 1
        ttk.Radiobutton(input_frame, text="dB", variable=self.scale_var, value="db").grid(row=r, column=1, sticky="w")

        r += 1
        add_btn = ttk.Button(input_frame, text="Add Plot", command=self.on_add_plot)
        add_btn.grid(row=r, column=0, columnspan=2, pady=5)

        r += 1
        self.plot_combo = ttk.Combobox(input_frame, state="readonly")
        self.plot_combo.grid(row=r, column=0, columnspan=2, sticky="we", pady=5)

        r += 1
        remove_btn = ttk.Button(input_frame, text="Remove Plot", command=self.on_remove_plot)
        remove_btn.grid(row=r, column=0, sticky="ew", pady=3)
        info_btn = ttk.Button(input_frame, text="View Info", command=self.on_view_info)
        info_btn.grid(row=r, column=1, sticky="ew", pady=3)

        r += 1
        clear_btn = ttk.Button(input_frame, text="Clear All", command=self.on_clear_plots)
        clear_btn.grid(row=r, column=0, columnspan=2, pady=5)

        r += 1
        phase_btn = ttk.Button(input_frame, text="Plot HPBW vs Phase", command=self.on_plot_hpbw_vs_phase)
        phase_btn.grid(row=r, column=0, columnspan=2, pady=5)

        r += 1
        dezoom_btn = ttk.Button(input_frame, text="Dezoom [-90..90]", command=self.on_dezoom)
        dezoom_btn.grid(row=r, column=0, columnspan=2, pady=5)

        r += 1
        rezoom_btn = ttk.Button(input_frame, text="Rezoom to Above", command=self.on_rezoom)
        rezoom_btn.grid(row=r, column=0, columnspan=2, pady=5)

        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.axis = self.figure.add_subplot(111)
        self.axis.set_title("Radiation Pattern")
        self.axis.set_xlabel("Angle (deg)")
        self.axis.set_ylabel("Normalized Intensity of Radiation")
        self.axis.grid(True)

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        self.results_label = ttk.Label(info_frame, text="", width=60, justify="left")
        self.results_label.grid(row=0, column=0, sticky="nw")

    def on_add_plot(self):
        """
        1) Compute the radiation pattern over -90°..+90°.
        2) Measure HPBW, main beam direction, and secondary lobe gain.
        3) Plot the pattern with red dashed lines at the half-power edges.
        4) The plot legend displays only the phase (deg & rad) and HPBW.
           Detailed beam info (main beam and secondary lobe) is stored and shown via "View Info".
        5) The x-axis limits are set to the user-specified initial zoom.
        """
        try:
            N = int(self.N_var.get())
            freq = float(self.freq_var.get())
            # Read phase shift in degrees, then convert to radians.
            phase_deg = float(self.delta_var.get())
            phase_rad = np.radians(phase_deg)
            amin = float(self.amin_var.get())
            amax = float(self.amax_var.get())
            leng = float(self.len_var.get())
        except ValueError:
            self._set_msg("Check numeric inputs.")
            return

        (full_phi_deg, pattern_norm, hpbw, left_edge, right_edge,
         main_beam_direction, secondary_gain, secondary_angle) = compute_pattern_full(
            N=N,
            freq_hz=freq,
            delta_phi=phase_rad,
            array_len=leng,
            do_square=True
        )

        # Choose scale: linear or dB.
        scale_mode = self.scale_var.get().lower()
        if scale_mode == "db":
            y_data = 10.0 * np.log10(pattern_norm + 1e-12)
            y_label = "Power (dB)"
        else:
            y_data = pattern_norm
            y_label = "Power (normalized)"

        self.plot_counter += 1
        # Plot legend shows only phase and HPBW.
        plot_label = (f"Graph {self.plot_counter}, δφ={phase_deg:.2f}deg ({phase_rad:.3f}rad), "
                      f"HPBW={hpbw:.2f}°")
        line_main = self.axis.plot(full_phi_deg, y_data, label=plot_label)

        left_line, right_line = None, None
        if hpbw > 0:
            left_line = self.axis.axvline(left_edge, color='red', linestyle='--')
            right_line = self.axis.axvline(right_edge, color='red', linestyle='--')

        self.axis.set_title("Radiation Pattern")
        self.axis.set_xlabel("Angle (deg)")
        self.axis.set_ylabel(y_label)
        self.axis.legend()

        # Set x-axis limits as per initial zoom.
        self.axis.set_xlim(amin, amax)
        self.canvas.draw()

        lines_list = [line_main[0]]
        if left_line:
            lines_list.append(left_line)
        if right_line:
            lines_list.append(right_line)

        # Save all beam parameters in the plots dictionary.
        self.plots_dict[plot_label] = {
            "lines": lines_list,
            "N": N,
            "freq": freq,
            "phase_deg": phase_deg,
            "phase_rad": phase_rad,
            "hpbw": hpbw,
            "left_edge": left_edge,
            "right_edge": right_edge,
            "main_beam_direction": main_beam_direction,
            "secondary_gain": secondary_gain,
            "secondary_angle": secondary_angle
        }

        val_list = list(self.plot_combo['values'])
        val_list.append(plot_label)
        self.plot_combo['values'] = val_list
        self.plot_combo.current(len(val_list) - 1)

    def on_remove_plot(self):
        sel = self.plot_combo.get().strip()
        if not sel or sel not in self.plots_dict:
            self._set_msg("No valid plot selected.")
            return
        for ln in self.plots_dict[sel]["lines"]:
            ln.remove()
        del self.plots_dict[sel]
        new_vals = list(self.plot_combo['values'])
        if sel in new_vals:
            new_vals.remove(sel)
        self.plot_combo['values'] = new_vals
        if new_vals:
            self.plot_combo.current(0)
        else:
            self.plot_combo.set("")
        self.axis.legend()
        self.canvas.draw()
        self.results_label.config(text="")

    def on_clear_plots(self):
        for lbl, info in self.plots_dict.items():
            for ln in info["lines"]:
                ln.remove()
        self.plots_dict.clear()
        self.plot_combo['values'] = []
        self.plot_combo.set("")
        self.axis.legend()
        self.canvas.draw()
        self.results_label.config(text="")

    def on_view_info(self):
        """
        Displays detailed beam parameters for the selected plot:
          - Number of elements, frequency, phase (deg & rad), HPBW, half-power edges,
            main beam direction, and secondary lobe gain and its angle.
        """
        sel = self.plot_combo.get().strip()
        if not sel or sel not in self.plots_dict:
            self.results_label.config(text="No valid plot selected.")
            return
        data = self.plots_dict[sel]
        msg = (
            f"{sel}\n"
            f"N = {data['N']}, freq = {data['freq']:.2e}\n"
            f"Phase = {data['phase_deg']:.2f} deg ({data['phase_rad']:.3f} rad)\n"
            f"HPBW = {data['hpbw']:.2f} deg\n"
            f"Left Edge = {data['left_edge']:.2f} deg\n"
            f"Right Edge = {data['right_edge']:.2f} deg\n"
            f"Direction of Max Gain = {data['main_beam_direction']:.2f} deg\n"
            f"Secondary Lobe: Gain = {data['secondary_gain']:.2f} at {data['secondary_angle']:.2f} deg\n"
        )
        self.results_label.config(text=msg)

    def on_plot_hpbw_vs_phase(self):
        """
        Uses the snippet approach (scanning -90°..+90° for the radiation pattern)
        to plot HPBW vs. phase shift. Here, the phase shift range is set from -100° to 100°.
        """
        try:
            N = int(self.N_var.get())
            freq = float(self.freq_var.get())
            leng = float(self.len_var.get())
        except ValueError:
            self._set_msg("Check numeric inputs for N, freq, length.")
            return

        c = 3e8
        lam = c / freq
        d = (leng / (N - 1)) if N > 1 else 0.0

        # Phase shift range from -100° to 100°.
        phase_shifts_deg = np.linspace(-100, 100, 100)
        hpbw_list = []
        for deg_val in phase_shifts_deg:
            rad_val = np.radians(deg_val)
            bw = snippet_compute_beamwidth(N, d, lam, rad_val)
            hpbw_list.append(bw)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(phase_shifts_deg, hpbw_list, 'r-')
        ax.set_xlabel("Phase Shift (deg)")
        ax.set_ylabel("Half-Power Beamwidth (deg)")
        ax.set_title("HPBW vs Phase (Snippet, angles -100°..+100°)")
        ax.grid(True)
        fig.tight_layout()
        plt.show()

    def on_dezoom(self):
        """Sets x-axis limits to [-90, 90]."""
        self.axis.set_xlim(-90, 90)
        self.canvas.draw()

    def on_rezoom(self):
        """Resets x-axis limits to the user-specified initial zoom."""
        try:
            amin = float(self.amin_var.get())
            amax = float(self.amax_var.get())
        except ValueError:
            self._set_msg("Check numeric inputs for rezoom domain.")
            return
        self.axis.set_xlim(amin, amax)
        self.canvas.draw()

    def _set_msg(self, msg):
        self.results_label.config(text=msg)


def main():
    root = tk.Tk()
    app = LinearArrayApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
