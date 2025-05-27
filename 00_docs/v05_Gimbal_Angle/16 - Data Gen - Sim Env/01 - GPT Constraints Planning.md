Below is a **complete design blueprint** you can drop straight into a repository and begin coding.

---

## 1  Coordinate-system & geometry assumptions

| Axis  | Positive direction   | Notes                           |
| ----- | -------------------- | ------------------------------- |
| **x** | left → right (East)  | origin at room corner (0 ,0 ,0) |
| **y** | front → back (North) | floor plane = *x-y*             |
| **z** | floor → ceiling (Up) | 0 m at floor                    |

*Room*: cube **50 m × 50 m × 50 m**.
*Mic-array centre*: fixed on the centre of the **East wall** at mid-height

$$
\mathbf p_\text{array} = (50,\;25,\;25)\;\text{m}
$$

*Speaker*: placed on an imaginary **unit sphere** centred on $\mathbf p_\text{array}$ and scaled to a user-chosen radius $r_\text{src}$ (defaults to 5 m).
Angles follow the acoustics convention:

* **Azimuth φ**: 0° = +x (towards the array), positive counter-clockwise in *x-y* plane.
* **Elevation θ**: 0° in horizontal plane, +90° straight up, –90° straight down.

---

## 2  Pydantic data model

```python
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
from pydantic import BaseModel, Field, validator

class RoomCfg(BaseModel):
    dimensions: Tuple[float, float, float] = (50.0, 50.0, 50.0)
    fs: int = 16_000
    max_order: int = 0                            # = free-field
    abs_wall: float = 1.0                        # 1 → perfectly absorbent
    abs_floor: float = 0.35                      # grass-like ground
    abs_ceiling: float = 1.0
    air_absorption: bool = True

class MicArrayCfg(BaseModel):
    xml_path: Path                                # SO-VITESS‐style XML
    position: Tuple[float, float, float] = (50, 25, 25)

class SpeakerGridCfg(BaseModel):
    radius: float = 5.0
    az_start: int = 0
    az_end: int = 355
    el_start: int = -40
    el_end: int = 40
    step: int = 5

class DatasetCfg(BaseModel):
    wav_files: List[Path]                         # will be cycled through
    output_dir: Path
    meta_header: str = "synthetic-pa-dataset"
    run_tag: str = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

class ExperimentCfg(BaseModel):
    room: RoomCfg
    mics: MicArrayCfg
    grid: SpeakerGridCfg
    data: DatasetCfg

    @validator('data')
    def _check_outdir(cls, v):
        v.output_dir.mkdir(parents=True, exist_ok=True)
        return v
```

Save a **YAML/JSON** file like:

```yaml
room:
  fs: 16000
  abs_floor: 0.35
mics:
  xml_path: "./arrays/kemar_32ch.xml"
grid:
  radius: 5           # metres
  el_start: -40
  el_end: 40
data:
  wav_files:
    - "./speech/f01_sentence01.wav"
    - "./speech/m01_sentence02.wav"
  output_dir: "./recordings"
```

---

## 3  Main generation script outline

```python
import json, itertools, soundfile as sf, numpy as np
import pyroomacoustics as pra
from pathlib import Path
from experiment_cfg import ExperimentCfg   # pydantic model above
from inverse_square import scale_signal    # see below

cfg = ExperimentCfg.parse_file("config.yml")

# ---------- build room ----------
absorption = [cfg.room.abs_wall]*4 + [cfg.room.abs_floor, cfg.room.abs_ceiling]
room = pra.ShoeBox(
    cfg.room.dimensions,
    fs=cfg.room.fs,
    max_order=cfg.room.max_order,
    materials=pra.Material(absorption),
    air_absorption=cfg.room.air_absorption
)

# ---------- load mic array ----------
mic_coords = load_mic_xml(cfg.mics.xml_path)          # → (3,N) ndarray
mic_array = pra.MicrophoneArray(mic_coords + np.c_[cfg.mics.position].T,
                                cfg.room.fs)
room.add_microphone_array(mic_array)

# ---------- iterate over angles ----------
az_vals = range(cfg.grid.az_start, cfg.grid.az_end+1, cfg.grid.step)
el_vals = range(cfg.grid.el_start, cfg.grid.el_end+1, cfg.grid.step)
wav_cycle = itertools.cycle(cfg.data.wav_files)

for az, el in itertools.product(az_vals, el_vals):
    wav_path = Path(next(wav_cycle))
    signal, sr = sf.read(wav_path, dtype='float32')
    assert sr == cfg.room.fs, "WAV must match room.fs"

    # --- place speaker ---
    src_vec = pra.direction_vector(np.deg2rad(az), np.deg2rad(el))
    src_pos = cfg.mics.position + cfg.grid.radius * src_vec
    room.remove_all_sources()
    room.add_source(src_pos, signal=signal)

    # --- attenuation ---
    dist = np.linalg.norm(src_pos - cfg.mics.position)
    room.sources[0].signal = scale_signal(signal, dist)

    # --- simulate & save ---
    room.simulate()
    rec = room.mic_array.signals.T         # shape: (nsamples, n_mics)

    npy_name = f"az{az:03d}_el{el:+03d}.npy"
    np.save(cfg.data.output_dir / npy_name, rec)

    meta = {
        "header": cfg.data.meta_header,
        "generated": cfg.data.run_tag,
        "wav": str(wav_path),
        "distance_m": dist,
        "azimuth_deg": az,
        "elevation_deg": el,
        "n_mics": rec.shape[1],
        "script": "generate_dataset.py"
    }
    with open(cfg.data.output_dir / (npy_name.replace('.npy','.json')), 'w') as f:
        json.dump(meta, f, indent=2)
```

### Helper: inverse-square scaling

```python
def scale_signal(sig: np.ndarray, dist_m: float, ref_dist=1.0):
    """Energy obeys 1/r²; amplitude obeys 1/r."""
    scale = ref_dist / max(dist_m, 1e-6)
    return sig * scale
```

### Helper: XML mic loader (very barebones)

```python
import numpy as np, xml.etree.ElementTree as ET

def load_mic_xml(path):
    tree = ET.parse(path)
    xyz = []
    for mic in tree.findall(".//Microphone"):
        x = float(mic.find("x").text)
        y = float(mic.find("y").text)
        z = float(mic.find("z").text)
        xyz.append((x,y,z))
    return np.array(xyz).T          # shape (3, N)
```

---

## 4  File-naming & duplication safety

* **Numpy**: `az000_el+00.npy` – three-digit azimuth, signed two-digit elevation.
* **Metadata JSON** shares basename – uniqueness is guaranteed by the exhaustive angle grid; script simply overwrites its own previous output when rerun with identical `run_tag`.
* To prevent accidental reuse, give each run a new `run_tag` or place in a separate sub-folder.

---

## 5  Assumptions & tunables

| Parameter              | Default                            | Change by…            |
| ---------------------- | ---------------------------------- | --------------------- |
| Number of reflections  | 0 (free field)                     | `room.max_order`      |
| Air absorption coeff.  | *Pra* default                      | `room.air_absorption` |
| Ground (grass) α       | 0.35 (typical, 100 Hz–8 kHz)       | `room.abs_floor`      |
| Radius of speaker ring | 5 m                                | `grid.radius`         |
| Angle limits & step    | –40°↔+40° elev, 0–355° az, 5° step | `grid.*`              |

---

## 6  Next steps / extensibility

* **Parallelisation** with `concurrent.futures` over angle tuples.
* **Different radii**: loop over multiple `grid.radius` values.
* **Multiple rooms**: extend `RoomCfg` list inside the root config.
* **Saving as multichannel WAV**: use `soundfile.write`.
* **Metadata schema evolution**: version field in `meta_header`.

You now have a reproducible pipeline to churn out hundreds of **labelled, distance-aware, angle-resolved synthetic recordings** for training beamformers, DOA estimators, or dereverberation networks. Happy data making!
