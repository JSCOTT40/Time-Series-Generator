# Time Series Generator and FFT Filter

A Python/Jupyter Notebook project for generating synthetic time-series signals made from multiple sinusoidal components, adding optional white noise, and applying frequency-domain filters using the Fast Fourier Transform.

This project demonstrates basic signal generation, spectral analysis, and digital filtering concepts using `NumPy` and `Matplotlib`.

---

## Overview

This notebook allows the user to build a custom time-series signal by choosing:

* Number of sinusoidal components
* Start and end time
* Frequencies
* Amplitudes
* Phases
* Optional white noise

After generating the signal, the notebook displays the data in both the time domain and frequency domain. The user can then apply different FFT-based filters to modify the signal.

---

## Features

* Generate a time-series from multiple sine waves
* Choose random, preset, or manual values for:

  * Frequency
  * Amplitude
  * Phase
* Add optional white noise
* Plot the generated signal in the time domain
* Compute and plot the power spectral density
* Apply custom frequency-domain filters:

  * High-pass filter
  * Low-pass filter
  * Band-pass filter
* Reset the filtered signal back to the original signal
* Compare original and filtered signals visually

---

## Sampling Configuration

The notebook uses a fixed sampling interval:

```python
SampF = 0.01
```

This means the time step between samples is:

```text
Δt = 0.01 seconds
```

This corresponds to a sampling frequency of:

```text
fs = 1 / Δt = 100 Hz
```

The Nyquist frequency is therefore:

```text
f_Nyquist = 50 Hz
```

This is why the generated random frequencies are selected between 0 Hz and 50 Hz.

---

## How It Works

### 1. Input Parameters

The user is prompted to enter the number of sinusoids and the time range for the signal.

The notebook then asks whether frequency, amplitude, and phase values should be:

* Randomly generated
* Automatically set
* Manually entered

---

### 2. Signal Generation

Each sinusoidal component is generated using:

```python
A * sin(2πft + φ)
```

where:

| Symbol | Meaning   |
| ------ | --------- |
| `A`    | Amplitude |
| `f`    | Frequency |
| `t`    | Time      |
| `φ`    | Phase     |

The final signal is built by summing all sinusoidal components together.

Optional white noise can also be added to the generated signal.

---

### 3. Frequency Analysis

The notebook uses the Fast Fourier Transform to analyze the signal in the frequency domain.

The power spectral density is calculated from the FFT result and displayed in decibels:

```python
psd = abs(fft_series) ** 2 / n
psd_db = 10 * log10(psd)
```

This allows the frequency components of the generated signal to be visualized clearly.

---

### 4. Filtering

The notebook supports three main filtering operations.

| Filter Type | Description                               |
| ----------- | ----------------------------------------- |
| High-pass   | Attenuates frequencies below a cutoff     |
| Low-pass    | Attenuates frequencies above a cutoff     |
| Band-pass   | Keeps frequencies within a selected range |

The filters are applied in the frequency domain by multiplying the FFT of the signal by a custom filter response. The filtered signal is then converted back to the time domain using the inverse FFT.

---

## Example Workflow

1. Run the notebook.
2. Enter the number of sine waves to generate.
3. Choose the time range.
4. Select random, set, or manual parameters.
5. Add optional white noise.
6. View the time-domain signal.
7. View the frequency-domain power spectrum.
8. Apply a high-pass, low-pass, or band-pass filter.
9. Compare the original and filtered signals.

---

## Project Structure

```text
.
├── Time Series Generator and Filter for Sample rate equal to 0.01.ipynb
└── README.md
```

---

## Requirements

This project requires Python 3 and the following libraries:

```text
numpy
matplotlib
jupyter
```

You can install the dependencies with:

```bash
pip install numpy matplotlib jupyter
```

---

## Running the Notebook

Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
```

Navigate into the project folder:

```bash
cd your-repo-name
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
Time Series Generator and Filter for Sample rate equal to 0.01.ipynb
```

---

## Educational Purpose

This project was built as a signal-processing exercise to explore:

* Time-series generation
* Sinusoidal signal construction
* White noise
* Fourier transforms
* Power spectral density
* Frequency-domain filtering
* High-pass, low-pass, and band-pass filter behavior

It is intended as a learning project for understanding how signals can be created, analyzed, and filtered using Python.

---

## Possible Improvements

Future improvements could include:

* Refactoring the notebook into reusable Python functions
* Adding a cleaner user interface
* Replacing repeated input validation code with helper functions
* Adding saved output files for generated plots
* Supporting configurable sampling intervals
* Adding Butterworth, Chebyshev, or FIR filters
* Exporting generated time-series data to CSV
* Adding unit tests for the signal and filter functions

---

## Author

Created by Jayce Scott.

---

## License

This project is open-source and available for educational use.
