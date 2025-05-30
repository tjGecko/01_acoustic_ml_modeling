import numpy as np
from pathlib import Path
from datetime import datetime
import re
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import column, row
from bokeh.models import (
    Slider, ColumnDataSource, ColorBar, LinearColorMapper,
    Div, Panel, Tabs
)
from bokeh.palettes import Viridis256

# ---------- Config ----------
SAMPLE_RATE = 16000
NUM_MICS = 16
WINDOW_SIZE = 2048
STEP = 512
SCRIPT_NAME = __file__
PLOT_DATE = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def extract_angles_from_filename(filename: str):
    match = re.search(r'az(-?\d{3})_el([+-]\d{2})', filename)
    if match:
        az = int(match.group(1))
        el = int(match.group(2))
        return az, el
    else:
        raise ValueError(f"Filename '{filename}' does not contain valid az/el information.")


# ---------- Load Data ----------
def load_mic_array_data(npy_path: Path) -> np.ndarray:
    arr = np.load(npy_path)
    if arr.ndim != 2 or arr.shape[1] != NUM_MICS:
        raise ValueError(f"Expected shape (samples, {NUM_MICS}), got {arr.shape}")
    return arr

# ---------- TDOA Matrix ----------
def compute_pairwise_tdoa(window: np.ndarray, fs: int) -> np.ndarray:
    n_mics = window.shape[1]
    tdoa_matrix = np.zeros((n_mics, n_mics))

    for i in range(n_mics):
        for j in range(n_mics):
            if i == j:
                continue
            corr = np.correlate(window[:, i] - np.mean(window[:, i]),
                                window[:, j] - np.mean(window[:, j]),
                                mode='full')
            lags = np.arange(-len(window) + 1, len(window))
            lag_at_max = lags[np.argmax(corr)]
            tdoa_ms = lag_at_max / fs * 1000
            tdoa_matrix[i, j] = tdoa_ms
    return tdoa_matrix

# ---------- Visualization ----------
def create_heatmap(tdoa_matrix: np.ndarray):
    mic_labels = [f"{i}" for i in range(NUM_MICS)]
    mapper = LinearColorMapper(palette=Viridis256, low=tdoa_matrix.min(), high=tdoa_matrix.max())

    p = figure(
        x_range=mic_labels, y_range=list(reversed(mic_labels)),
        title="Pairwise TDOA Heatmap (ms)",
        x_axis_label="Receiver Microphone Index",
        y_axis_label="Reference Microphone Index",
        toolbar_location=None, tools="hover"
    )

    source = ColumnDataSource(data=dict(
        x=[str(i) for i in range(NUM_MICS) for j in range(NUM_MICS)],
        y=[str(j) for i in range(NUM_MICS) for j in range(NUM_MICS)],
        val=tdoa_matrix.flatten(),
    ))

    p.rect(x="x", y="y", width=1, height=1, source=source,
           fill_color={'field': 'val', 'transform': mapper}, line_color=None)

    color_bar = ColorBar(color_mapper=mapper, location=(0, 0))
    p.add_layout(color_bar, 'right')

    return p, source, mapper

# ---------- UI Update ----------
def update(attr, old, new):
    start = slider.value * STEP
    end = start + WINDOW_SIZE
    window = mic_data[start:end]
    tdoa_matrix = compute_pairwise_tdoa(window, SAMPLE_RATE)

    source.data.update({
        'x': [str(i) for i in range(NUM_MICS) for j in range(NUM_MICS)],
        'y': [str(j) for i in range(NUM_MICS) for j in range(NUM_MICS)],
        'val': tdoa_matrix.flatten(),
    })
    mapper.low = tdoa_matrix.min()
    mapper.high = tdoa_matrix.max()

# ---------- Main ----------
# npy_path = Path("/home/tj/99_tmp/11 - synthetic mic array data/01_time_domain/capture_2025_05_29/data/Membo_1_047-membo_003__az000_el+00.npy")  # <--- UPDATE ME
npy_path = Path("/media/tj/Samsung_T5/Ziz/01_time_domain/capture_2025_05_29/data/mixed_46-bebop_001__az090_el+30.npy")  # <--- UPDATE ME
mic_data = load_mic_array_data(npy_path)
initial_window = mic_data[:WINDOW_SIZE]
tdoa_matrix = compute_pairwise_tdoa(initial_window, SAMPLE_RATE)

# Plot elements
heatmap, source, mapper = create_heatmap(tdoa_matrix)

slider = Slider(start=0, end=(mic_data.shape[0] - WINDOW_SIZE) // STEP,
                value=0, step=1, title="Select Time Slice")

slider.on_change('value', update)

# ---------- Side Panel ----------
# Extract az/el from filename
azimuth, elevation = extract_angles_from_filename(npy_path.name)

interpretation_text = f"""
<h2>TDOA Heatmap Interpretation</h2>
<ul>
    <li>Each cell shows the time delay (ms) between a pair of microphones.</li>
    <li><b>Positive</b>: Receiver mic heard the sound <i>after</i> the reference mic.</li>
    <li><b>Negative</b>: Receiver mic heard the sound <i>before</i> the reference mic.</li>
    <li>Diagonal = zero (mic compared to itself).</li>
    <li>Color scale indicates magnitude and direction of delay.</li>
</ul>
<hr>
<h3>Metadata</h3>
<ul>
    <li><b>Script:</b> {SCRIPT_NAME}</li>
    <li><b>Date:</b> {PLOT_DATE}</li>
    <li><b>File:</b> {npy_path.name}</li>
    <li><b>Azimuth:</b> {azimuth}°</li>
    <li><b>Elevation:</b> {elevation}°</li>
</ul>
"""

description_panel = Div(text=interpretation_text, width=400)

# ---------- Layout & Display ----------
layout = row(column(slider, heatmap), description_panel)

output_file("tdoa_heatmap_mixed_46.html")
show(layout)
