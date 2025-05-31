# /media/tj/Samsung_T5/Ziz/01_time_domain/capture_2025_05_29/data/mixed_46-bebop_001__az090_el+30.npy
from pathlib import Path
import numpy as np
from scipy.io import wavfile

# Parameters
input_path = Path("/media/tj/Samsung_T5/Ziz/01_time_domain/capture_2025_05_29/data/mixed_46-bebop_001__az090_el+30.npy")
output_wav = Path("/home/tj/99_tmp/11 - synthetic mic array data/01_time_domain/output.wav")
sample_rate = 16000  # Adjust this to match your recording rate

# Load the serialized NumPy array
data = np.load(input_path)

# If it's a .npz with named arrays
if isinstance(data, np.lib.npyio.NpzFile):
    data = data['arr_0']  # or change to the appropriate key

# Check data shape
if data.ndim != 2 or data.shape[1] != 16:
    raise ValueError(f"Expected shape (n_samples, 16), but got {data.shape}")

# Normalize or scale if needed (e.g., convert float to int16)
if data.dtype != np.int16:
    # Normalize to int16 range
    data = (data / np.max(np.abs(data)) * 32767).astype(np.int16)

# Write to WAV file (scipy writes multi-channel if shape is (n_samples, channels))
wavfile.write(output_wav, sample_rate, data)

print(f"✅ WAV file saved to {output_wav.resolve()}")
