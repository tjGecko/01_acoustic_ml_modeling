import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.io import save
from bokeh.layouts import column
from pathlib import Path

def load_mic_array_data(npy_path: Path) -> np.ndarray:
    arr = np.load(npy_path)
    if arr.ndim != 2 or arr.shape[1] != 16:
        raise ValueError(f"Expected shape (samples, 16), got {arr.shape}")
    return arr

def compute_tdoa(signal_array: np.ndarray, fs: int = 16000) -> np.ndarray:
    """Computes TDOA in milliseconds between mic[0] and others."""
    ref_signal = signal_array[:, 0]
    tdoa_ms = []

    for i in range(signal_array.shape[1]):
        mic_signal = signal_array[:, i]

        # Cross-correlation
        corr = np.correlate(mic_signal - np.mean(mic_signal),
                            ref_signal - np.mean(ref_signal), mode='full')
        lags = np.arange(-len(ref_signal)+1, len(ref_signal))
        lag_at_max = lags[np.argmax(corr)]
        time_shift = lag_at_max / fs  # seconds
        tdoa_ms.append(time_shift * 1000)  # convert to ms

    return np.array(tdoa_ms)

def plot_tdoa(tdoa_array: np.ndarray):
    mic_ids = [f'Mic {i}' for i in range(len(tdoa_array))]
    p = figure(x_range=mic_ids, title="TDOA (ms) relative to Mic 0",
               y_axis_label="Time Difference (ms)", height=400)

    p.vbar(x=mic_ids, top=tdoa_array, width=0.8)

    p.xgrid.grid_line_color = None
    p.y_range.start = min(tdoa_array.min(), 0)
    p.y_range.end = max(tdoa_array.max(), 0)
    p.title.text_font_size = '16pt'

    output_file("tdoa_plot.html")
    save(p)
    show(p)

if __name__ == "__main__":
    # Example usage
    npy_path = Path("/home/tj/99_tmp/11 - synthetic mic array data/01_time_domain/capture_2025_05_29/data/Membo_1_047-membo_003__az000_el+00.npy")  # UPDATE THIS
    mic_signals = load_mic_array_data(npy_path)
    tdoa = compute_tdoa(mic_signals, fs=16000)
    plot_tdoa(tdoa)
