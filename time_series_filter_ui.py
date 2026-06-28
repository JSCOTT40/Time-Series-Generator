#!/usr/bin/env python3
"""
Time Series Generator and FFT Filter UI

A standalone Tkinter app based on the notebook design:
- Builds a time series from multiple sinusoids
- Supports random, set, or manual frequency/amplitude/phase values
- Adds optional uniform white noise
- Applies Gaussian-smoothed high-pass, low-pass, or band-pass filters in the FFT domain
- Displays all graphs in one window

Run:
    python time_series_filter_ui.py

Dependencies:
    pip install numpy matplotlib
"""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import List, Tuple
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


def parse_float(value: str, field_name: str) -> float:
    """Parse a required float field with a useful UI error message."""
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a number.") from exc


def parse_int(value: str, field_name: str) -> int:
    """Parse a required integer field with a useful UI error message."""
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc


def parse_csv_floats(value: str, expected_count: int, field_name: str) -> List[float]:
    """
    Parse comma-separated values such as:
        1, 11, 21
    """
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != expected_count:
        raise ValueError(f"{field_name} must contain exactly {expected_count} comma-separated values.")
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"{field_name} must only contain numeric values.") from exc


def psd_db(series: np.ndarray) -> np.ndarray:
    """
    Power spectral density in dB.

    Small floor avoids log10(0) warnings when a bin has zero power.
    """
    n = len(series)
    fft_series = np.fft.fft(series)
    psd = np.abs(fft_series) ** 2 / max(n, 1)
    psd = np.maximum(psd, 1e-20)
    return 10 * np.log10(psd)


def choose_values(
    mode: str,
    count: int,
    manual_text: str,
    random_low: float,
    random_high: float,
    set_start: float,
    set_step: float,
    set_value: float,
    field_name: str,
    rng: np.random.Generator,
    set_pattern: str = "constant",
) -> List[float]:
    """
    Generate a parameter list using one of the notebook's three modes:
    Random, Set, or Manual.

    set_pattern:
        "linear"   -> set_start + set_step * i
        "constant" -> set_value
    """
    mode = mode.strip().lower()

    if mode == "manual":
        return parse_csv_floats(manual_text, count, field_name)

    if mode == "random":
        if random_high < random_low:
            raise ValueError(f"{field_name} random high must be greater than or equal to random low.")
        return list(rng.uniform(random_low, random_high, count))

    if mode == "set":
        if set_pattern == "linear":
            return [set_start + set_step * i for i in range(count)]
        return [set_value for _ in range(count)]

    raise ValueError(f"{field_name} mode must be Random, Set, or Manual.")


def build_series(
    n_sinusoids: int,
    t0: float,
    tf: float,
    dt: float,
    frequencies: List[float],
    amplitudes: List[float],
    phases: List[float],
    add_noise: bool,
    noise_low: float,
    noise_high: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the generated time series:
        sum_i A_i sin(2π f_i t + φ_i) + uniform white noise
    """
    if n_sinusoids <= 0:
        raise ValueError("Number of sinusoids must be greater than zero.")
    if dt <= 0:
        raise ValueError("Sample interval dt must be greater than zero.")
    if tf <= t0:
        raise ValueError("Final time must be greater than starting time.")

    # Match the notebook idea of snapping tf to the closest sample interval.
    steps = int(round((tf - t0) / dt))
    if steps < 2:
        raise ValueError("Time range is too short for the chosen sample interval.")
    tf_adjusted = t0 + steps * dt

    t = np.arange(t0, tf_adjusted, dt)
    series = np.zeros_like(t, dtype=float)

    for amplitude, frequency, phase in zip(amplitudes, frequencies, phases):
        series += amplitude * np.sin(2 * np.pi * frequency * t + phase)

    if add_noise:
        if noise_high < noise_low:
            raise ValueError("Noise high bound must be greater than or equal to noise low bound.")
        noise = rng.uniform(noise_low, noise_high, len(t))
    else:
        noise = np.zeros_like(t)

    return t, series + noise, noise


def filter_response(
    frequencies: np.ndarray,
    filter_type: str,
    cutoff_low: float,
    cutoff_high: float,
    transition_width: float,
) -> np.ndarray:
    """
    Create a Gaussian-smoothed FFT filter.

    The original notebook uses hand-built Gaussian-ish filter arrays.
    This version keeps the same idea but makes the band-pass behavior clean:
        high-pass  -> suppresses frequencies below cutoff_low
        low-pass   -> suppresses frequencies above cutoff_high
        band-pass  -> high-pass at cutoff_low multiplied by low-pass at cutoff_high
    """
    abs_freq = np.abs(frequencies)

    if transition_width <= 0:
        raise ValueError("Transition width must be greater than zero.")

    def lowpass(cutoff: float) -> np.ndarray:
        if cutoff < 0:
            raise ValueError("Low-pass cutoff must be non-negative.")
        return np.where(
            abs_freq <= cutoff,
            1.0,
            np.exp(-((abs_freq - cutoff) / transition_width) ** 2),
        )

    def highpass(cutoff: float) -> np.ndarray:
        if cutoff < 0:
            raise ValueError("High-pass cutoff must be non-negative.")
        return np.where(
            abs_freq >= cutoff,
            1.0,
            np.exp(-((cutoff - abs_freq) / transition_width) ** 2),
        )

    filter_type = filter_type.strip().lower()

    if filter_type == "none":
        return np.ones_like(frequencies, dtype=float)

    if filter_type == "high pass":
        return highpass(cutoff_low)

    if filter_type == "low pass":
        return lowpass(cutoff_high)

    if filter_type == "band pass":
        if cutoff_high <= cutoff_low:
            raise ValueError("For a band-pass filter, upper cutoff must be greater than lower cutoff.")
        return highpass(cutoff_low) * lowpass(cutoff_high)

    raise ValueError("Filter type must be None, High pass, Low pass, or Band pass.")


def apply_fft_filter(series: np.ndarray, response: np.ndarray) -> np.ndarray:
    """Apply an FFT-domain filter and return the real-valued time-domain result."""
    filtered_freq = np.fft.fft(series) * response
    return np.real(np.fft.ifft(filtered_freq))


class TimeSeriesFilterApp(tk.Tk):
    """Main Tkinter application."""

    def __init__(self) -> None:
        super().__init__()

        self.title("Time Series Generator and Filter")
        self.geometry("1500x950")
        self.minsize(1100, 720)

        self.t: np.ndarray | None = None
        self.original_series: np.ndarray | None = None
        self.filtered_series: np.ndarray | None = None
        self.frequency_axis: np.ndarray | None = None
        self.current_filter_response: np.ndarray | None = None
        self.current_parameters: dict[str, List[float]] = {}

        self._build_variables()
        self._build_layout()
        self.generate_and_plot()

    def _build_variables(self) -> None:
        """Create Tk variables used by input controls."""
        self.n_sinusoids_var = tk.StringVar(value="3")
        self.t0_var = tk.StringVar(value="0")
        self.tf_var = tk.StringVar(value="10")
        self.dt_var = tk.StringVar(value="0.01")
        self.seed_var = tk.StringVar(value="")
        self.display_samples_var = tk.StringVar(value="200")

        self.freq_mode_var = tk.StringVar(value="Set")
        self.freq_manual_var = tk.StringVar(value="1, 11, 21")
        self.freq_random_low_var = tk.StringVar(value="0")
        self.freq_random_high_var = tk.StringVar(value="50")
        self.freq_set_start_var = tk.StringVar(value="1")
        self.freq_set_step_var = tk.StringVar(value="10")

        self.amp_mode_var = tk.StringVar(value="Set")
        self.amp_manual_var = tk.StringVar(value="10, 10, 10")
        self.amp_random_low_var = tk.StringVar(value="0")
        self.amp_random_high_var = tk.StringVar(value="50")
        self.amp_set_value_var = tk.StringVar(value="10")

        self.phase_mode_var = tk.StringVar(value="Set")
        self.phase_manual_var = tk.StringVar(value="3.14, 3.14, 3.14")
        self.phase_random_low_var = tk.StringVar(value="0")
        self.phase_random_high_var = tk.StringVar(value="50")
        self.phase_set_value_var = tk.StringVar(value="3.14")

        self.add_noise_var = tk.BooleanVar(value=False)
        self.noise_low_var = tk.StringVar(value="0")
        self.noise_high_var = tk.StringVar(value="2")

        self.filter_type_var = tk.StringVar(value="Low pass")
        self.cutoff_low_var = tk.StringVar(value="5")
        self.cutoff_high_var = tk.StringVar(value="15")
        self.transition_width_var = tk.StringVar(value="2.2")

        self.status_var = tk.StringVar(value="Ready.")

    def _build_layout(self) -> None:
        """Create the control panel and plot area."""
        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        controls_outer = ttk.Frame(main, width=360)
        plot_frame = ttk.Frame(main)

        main.add(controls_outer, weight=0)
        main.add(plot_frame, weight=1)

        self._build_controls(controls_outer)
        self._build_plots(plot_frame)

    def _build_controls(self, parent: ttk.Frame) -> None:
        """Build a scrollable left-side control panel."""
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        self.controls_frame = ttk.Frame(canvas)

        self.controls_frame.bind(
            "<Configure>",
            lambda event: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=self.controls_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        row = 0
        title = ttk.Label(self.controls_frame, text="Signal Inputs", font=("TkDefaultFont", 14, "bold"))
        title.grid(row=row, column=0, columnspan=2, padx=12, pady=(14, 8), sticky="w")
        row += 1

        row = self._entry(row, "Number of sinusoids", self.n_sinusoids_var)
        row = self._entry(row, "Start time", self.t0_var)
        row = self._entry(row, "Final time", self.tf_var)
        row = self._entry(row, "Sample interval dt", self.dt_var)
        row = self._entry(row, "Random seed (optional)", self.seed_var)
        row = self._entry(row, "Time plot samples", self.display_samples_var)

        row = self._section(row, "Frequency")
        row = self._combo(row, "Mode", self.freq_mode_var, ("Random", "Set", "Manual"))
        row = self._entry(row, "Manual CSV", self.freq_manual_var)
        row = self._entry(row, "Random low", self.freq_random_low_var)
        row = self._entry(row, "Random high", self.freq_random_high_var)
        row = self._entry(row, "Set start", self.freq_set_start_var)
        row = self._entry(row, "Set step", self.freq_set_step_var)

        row = self._section(row, "Amplitude")
        row = self._combo(row, "Mode", self.amp_mode_var, ("Random", "Set", "Manual"))
        row = self._entry(row, "Manual CSV", self.amp_manual_var)
        row = self._entry(row, "Random low", self.amp_random_low_var)
        row = self._entry(row, "Random high", self.amp_random_high_var)
        row = self._entry(row, "Set value", self.amp_set_value_var)

        row = self._section(row, "Phase")
        row = self._combo(row, "Mode", self.phase_mode_var, ("Random", "Set", "Manual"))
        row = self._entry(row, "Manual CSV", self.phase_manual_var)
        row = self._entry(row, "Random low", self.phase_random_low_var)
        row = self._entry(row, "Random high", self.phase_random_high_var)
        row = self._entry(row, "Set value", self.phase_set_value_var)

        row = self._section(row, "White Noise")
        ttk.Checkbutton(self.controls_frame, text="Add uniform white noise", variable=self.add_noise_var).grid(
            row=row, column=0, columnspan=2, padx=12, pady=4, sticky="w"
        )
        row += 1
        row = self._entry(row, "Noise low", self.noise_low_var)
        row = self._entry(row, "Noise high", self.noise_high_var)

        row = self._section(row, "FFT Filter")
        row = self._combo(row, "Filter type", self.filter_type_var, ("None", "High pass", "Low pass", "Band pass"))
        row = self._entry(row, "Lower cutoff", self.cutoff_low_var)
        row = self._entry(row, "Upper cutoff", self.cutoff_high_var)
        row = self._entry(row, "Transition width", self.transition_width_var)

        ttk.Button(self.controls_frame, text="Generate / Update Graphs", command=self.generate_and_plot).grid(
            row=row, column=0, columnspan=2, padx=12, pady=(14, 4), sticky="ew"
        )
        row += 1

        ttk.Button(self.controls_frame, text="Reset Filter to None", command=self.reset_filter).grid(
            row=row, column=0, columnspan=2, padx=12, pady=4, sticky="ew"
        )
        row += 1

        ttk.Button(self.controls_frame, text="Save Graph Window as PNG", command=self.save_figure).grid(
            row=row, column=0, columnspan=2, padx=12, pady=4, sticky="ew"
        )
        row += 1

        ttk.Label(self.controls_frame, textvariable=self.status_var, wraplength=320).grid(
            row=row, column=0, columnspan=2, padx=12, pady=(10, 18), sticky="ew"
        )

        self.controls_frame.columnconfigure(1, weight=1)

    def _section(self, row: int, text: str) -> int:
        label = ttk.Label(self.controls_frame, text=text, font=("TkDefaultFont", 11, "bold"))
        label.grid(row=row, column=0, columnspan=2, padx=12, pady=(14, 4), sticky="w")
        return row + 1

    def _entry(self, row: int, label_text: str, variable: tk.StringVar) -> int:
        ttk.Label(self.controls_frame, text=label_text).grid(row=row, column=0, padx=12, pady=3, sticky="w")
        ttk.Entry(self.controls_frame, textvariable=variable, width=22).grid(
            row=row, column=1, padx=12, pady=3, sticky="ew"
        )
        return row + 1

    def _combo(self, row: int, label_text: str, variable: tk.StringVar, values: Tuple[str, ...]) -> int:
        ttk.Label(self.controls_frame, text=label_text).grid(row=row, column=0, padx=12, pady=3, sticky="w")
        combo = ttk.Combobox(self.controls_frame, textvariable=variable, values=values, state="readonly", width=20)
        combo.grid(row=row, column=1, padx=12, pady=3, sticky="ew")
        return row + 1

    def _build_plots(self, parent: ttk.Frame) -> None:
        """Create the Matplotlib figure embedded in Tkinter."""
        self.figure = Figure(figsize=(12, 8.5), dpi=100, constrained_layout=True)
        self.axes = self.figure.subplots(3, 2)

        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, parent)
        toolbar.update()

    def _collect_inputs(self) -> dict:
        """Read and validate all UI inputs."""
        n_sinusoids = parse_int(self.n_sinusoids_var.get(), "Number of sinusoids")
        t0 = parse_float(self.t0_var.get(), "Start time")
        tf = parse_float(self.tf_var.get(), "Final time")
        dt = parse_float(self.dt_var.get(), "Sample interval dt")
        display_samples = parse_int(self.display_samples_var.get(), "Time plot samples")

        seed_text = self.seed_var.get().strip()
        seed = None if seed_text == "" else parse_int(seed_text, "Random seed")
        rng = np.random.default_rng(seed)

        frequencies = choose_values(
            mode=self.freq_mode_var.get(),
            count=n_sinusoids,
            manual_text=self.freq_manual_var.get(),
            random_low=parse_float(self.freq_random_low_var.get(), "Frequency random low"),
            random_high=parse_float(self.freq_random_high_var.get(), "Frequency random high"),
            set_start=parse_float(self.freq_set_start_var.get(), "Frequency set start"),
            set_step=parse_float(self.freq_set_step_var.get(), "Frequency set step"),
            set_value=0.0,
            field_name="Frequencies",
            rng=rng,
            set_pattern="linear",
        )

        amplitudes = choose_values(
            mode=self.amp_mode_var.get(),
            count=n_sinusoids,
            manual_text=self.amp_manual_var.get(),
            random_low=parse_float(self.amp_random_low_var.get(), "Amplitude random low"),
            random_high=parse_float(self.amp_random_high_var.get(), "Amplitude random high"),
            set_start=0.0,
            set_step=0.0,
            set_value=parse_float(self.amp_set_value_var.get(), "Amplitude set value"),
            field_name="Amplitudes",
            rng=rng,
            set_pattern="constant",
        )

        phases = choose_values(
            mode=self.phase_mode_var.get(),
            count=n_sinusoids,
            manual_text=self.phase_manual_var.get(),
            random_low=parse_float(self.phase_random_low_var.get(), "Phase random low"),
            random_high=parse_float(self.phase_random_high_var.get(), "Phase random high"),
            set_start=0.0,
            set_step=0.0,
            set_value=parse_float(self.phase_set_value_var.get(), "Phase set value"),
            field_name="Phases",
            rng=rng,
            set_pattern="constant",
        )

        return {
            "n_sinusoids": n_sinusoids,
            "t0": t0,
            "tf": tf,
            "dt": dt,
            "display_samples": display_samples,
            "rng": rng,
            "frequencies": frequencies,
            "amplitudes": amplitudes,
            "phases": phases,
            "add_noise": self.add_noise_var.get(),
            "noise_low": parse_float(self.noise_low_var.get(), "Noise low"),
            "noise_high": parse_float(self.noise_high_var.get(), "Noise high"),
            "filter_type": self.filter_type_var.get(),
            "cutoff_low": parse_float(self.cutoff_low_var.get(), "Lower cutoff"),
            "cutoff_high": parse_float(self.cutoff_high_var.get(), "Upper cutoff"),
            "transition_width": parse_float(self.transition_width_var.get(), "Transition width"),
        }

    def generate_and_plot(self) -> None:
        """Generate signal, apply selected filter, and update all plots."""
        try:
            params = self._collect_inputs()

            t, original, noise = build_series(
                n_sinusoids=params["n_sinusoids"],
                t0=params["t0"],
                tf=params["tf"],
                dt=params["dt"],
                frequencies=params["frequencies"],
                amplitudes=params["amplitudes"],
                phases=params["phases"],
                add_noise=params["add_noise"],
                noise_low=params["noise_low"],
                noise_high=params["noise_high"],
                rng=params["rng"],
            )

            frequency_axis = np.fft.fftfreq(len(t), d=params["dt"])
            response = filter_response(
                frequencies=frequency_axis,
                filter_type=params["filter_type"],
                cutoff_low=params["cutoff_low"],
                cutoff_high=params["cutoff_high"],
                transition_width=params["transition_width"],
            )
            filtered = apply_fft_filter(original, response)

            self.t = t
            self.original_series = original
            self.filtered_series = filtered
            self.frequency_axis = frequency_axis
            self.current_filter_response = response
            self.current_parameters = {
                "frequencies": params["frequencies"],
                "amplitudes": params["amplitudes"],
                "phases": params["phases"],
            }

            self._plot_all(params)
            self.status_var.set(
                f"Generated {len(t)} samples. Nyquist frequency = {1 / (2 * params['dt']):.3g} Hz."
            )

        except Exception as exc:
            messagebox.showerror("Input Error", str(exc))
            self.status_var.set(f"Error: {exc}")

    def _plot_all(self, params: dict) -> None:
        """Refresh the six-panel graph window."""
        if (
            self.t is None
            or self.original_series is None
            or self.filtered_series is None
            or self.frequency_axis is None
            or self.current_filter_response is None
        ):
            return

        t = self.t
        original = self.original_series
        filtered = self.filtered_series
        freq = self.frequency_axis
        response = self.current_filter_response

        display_samples = params["display_samples"]
        if display_samples <= 0 or display_samples > len(t):
            display_samples = len(t)

        half = len(freq) // 2
        freq_positive = freq[:half]
        original_psd = psd_db(original)[:half]
        filtered_psd = psd_db(filtered)[:half]

        for ax_row in self.axes:
            for ax in ax_row:
                ax.clear()

        ax_original_time = self.axes[0][0]
        ax_original_freq = self.axes[0][1]
        ax_filter = self.axes[1][0]
        ax_filtered_time = self.axes[1][1]
        ax_filtered_freq = self.axes[2][0]
        ax_table = self.axes[2][1]

        ax_original_time.plot(t[:display_samples], original[:display_samples])
        ax_original_time.set_title("Original time series")
        ax_original_time.set_xlabel("Time")
        ax_original_time.set_ylabel("Amplitude")
        ax_original_time.grid(True)

        ax_original_freq.plot(freq_positive, original_psd)
        ax_original_freq.set_title("Original frequency domain")
        ax_original_freq.set_xlabel("Frequency (Hz)")
        ax_original_freq.set_ylabel("PSD (dB)")
        ax_original_freq.grid(True)

        ax_filter.plot(np.abs(freq), np.abs(response), linestyle="", marker=".", markersize=2)
        ax_filter.set_title(f"Filter response: {params['filter_type']}")
        ax_filter.set_xlabel("Frequency (Hz)")
        ax_filter.set_ylabel("Amplitude")
        ax_filter.set_ylim(-0.05, 1.05)
        ax_filter.set_xlim(0, max(0.01, np.max(np.abs(freq))))
        ax_filter.grid(True)

        if params["filter_type"].strip().lower() in {"high pass", "band pass"}:
            ax_filter.axvline(params["cutoff_low"], linestyle="--", alpha=0.65, label="Lower cutoff")
        if params["filter_type"].strip().lower() in {"low pass", "band pass"}:
            ax_filter.axvline(params["cutoff_high"], linestyle="--", alpha=0.65, label="Upper cutoff")
        if params["filter_type"].strip().lower() != "none":
            ax_filter.legend()

        ax_filtered_time.plot(t[:display_samples], original[:display_samples], label="Original")
        ax_filtered_time.plot(t[:display_samples], filtered[:display_samples], label="Filtered")
        ax_filtered_time.set_title("Time domain comparison")
        ax_filtered_time.set_xlabel("Time")
        ax_filtered_time.set_ylabel("Amplitude")
        ax_filtered_time.legend()
        ax_filtered_time.grid(True)

        ax_filtered_freq.plot(freq_positive, original_psd, label="Original")
        ax_filtered_freq.plot(freq_positive, filtered_psd, label="Filtered")
        ax_filtered_freq.set_title("Frequency domain comparison")
        ax_filtered_freq.set_xlabel("Frequency (Hz)")
        ax_filtered_freq.set_ylabel("PSD (dB)")
        ax_filtered_freq.set_ylim(bottom=-200)
        ax_filtered_freq.legend()
        ax_filtered_freq.grid(True)

        ax_table.axis("off")
        rows = ["i        Frequency        Amplitude        Phase"]
        for index, (f_val, a_val, p_val) in enumerate(
            zip(
                self.current_parameters["frequencies"],
                self.current_parameters["amplitudes"],
                self.current_parameters["phases"],
            ),
            start=1,
        ):
            rows.append(f"{index:<8}{f_val:<17.6g}{a_val:<17.6g}{p_val:<.6g}")

        parameter_text = "\n".join(rows)
        ax_table.set_title("Generated sinusoid parameters")
        ax_table.text(
            0.02,
            0.95,
            parameter_text,
            va="top",
            ha="left",
            family="monospace",
            fontsize=10,
            transform=ax_table.transAxes,
        )

        self.canvas.draw_idle()

    def reset_filter(self) -> None:
        """Set filter to None and redraw."""
        self.filter_type_var.set("None")
        self.generate_and_plot()

    def save_figure(self) -> None:
        """Save the current graph panel as a PNG."""
        file_path = filedialog.asksaveasfilename(
            title="Save graphs as PNG",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if not file_path:
            return

        try:
            self.figure.savefig(file_path, dpi=150)
            self.status_var.set(f"Saved graph window to {file_path}")
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc))
            self.status_var.set(f"Error saving file: {exc}")


if __name__ == "__main__":
    app = TimeSeriesFilterApp()
    app.mainloop()
