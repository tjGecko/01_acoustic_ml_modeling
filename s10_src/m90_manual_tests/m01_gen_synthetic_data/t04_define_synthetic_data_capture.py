import itertools
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np, xml.etree.ElementTree as ET
import pyroomacoustics as pra
import soundfile as sf

from s10_src.m05_data_models.d04_acoustic_sim_env import AcousticSimulationConfig


def load_configs(project_root: Path) -> Dict[str, Any]:
    # Path to the YAML configuration file
    config_path = project_root / '05_config' / 'c11_syn_acoustic_data_gen.yaml'
    print(f"Loading configuration from: {config_path}")

    # Load and validate the configuration
    config = AcousticSimulationConfig.from_yaml(config_path)
    print("\n✅ Configuration loaded and validated successfully!")

    mic_coords = load_mic_xml(config.mics.xml_path)  # → (3,N) ndarray

    # WAVE files to replay
    wav_json_path = project_root / '05_config' / 'clean_wav_registry.json'

    # Load the JSON data
    with wav_json_path.open('r', encoding='utf-8') as f:
        wav_json = json.load(f)

    cfg = {
        'AcousticSimulationConfig': config,
        'mic_coords': mic_coords,
        'wav_cfgs': wav_json,
    }

    return cfg

def load_mic_xml(path):
    tree = ET.parse(path)
    xyz = []
    for mic in tree.findall(".//Microphone"):
        x = float(mic.find("x").text)
        y = float(mic.find("y").text)
        z = float(mic.find("z").text)
        xyz.append((x,y,z))
    return np.array(xyz).T

def define_virtual_env(cfg: Dict[str, Any]) -> Dict[str, Any]:
    acfg = cfg['AcousticSimulationConfig']

    # ---------- build room ----------
    absorption = [acfg.room.abs_wall] * 4 + [acfg.room.abs_floor, acfg.room.abs_ceiling]
    room = pra.ShoeBox(
        acfg.room.dimensions,
        fs=acfg.room.fs,
        max_order=acfg.room.max_order,
        materials=pra.Material(absorption),
        air_absorption=acfg.room.air_absorption
    )

    mic_array = pra.MicrophoneArray(cfg['mic_coords'] + np.c_[acfg.mics.position].T, acfg.room.fs)
    room.add_microphone_array(mic_array)

    cfg['sim_env'] = {
        'room': room,
        'mic_array': mic_array
    }

    return cfg


def generate_data(cfg: Dict[str, Any]) -> Dict[str, Any]:
    room = cfg['sim_env']['room']
    acfg = cfg['AcousticSimulationConfig']
    wav_cfgs = cfg['wav_cfgs']

    # Build angles for grid coverage
    az_vals = range(acfg.grid.az_start, acfg.grid.az_end + 1, acfg.grid.step)
    el_vals = range(acfg.grid.el_start, acfg.grid.el_end + 1, acfg.grid.step)

    for az, el in itertools.product(az_vals, el_vals):
        # --- place speaker (e.g. virtual drone) ---
        src_vec = pra.direction_vector(np.deg2rad(az), np.deg2rad(el))
        src_pos = acfg.mics.position + acfg.grid.radius * src_vec
        room.remove_all_sources()

        wav_path = Path(wav_cfgs[0]['file_path'])     # fixme: Hard coded
        signal, sr = sf.read(wav_path.resolve(), dtype='float32')
        room.add_source(src_pos, signal=signal)

        # # --- attenuation ---
        # dist = np.linalg.norm(src_pos - acfg.mics.position)
        # room.sources[0].signal = scale_signal(signal, dist)

        # --- simulate & save ---
        room.simulate()
        rec = room.mic_array.signals.T  # shape: (nsamples, n_mics)

        npy_name = f"{wav_path.stem}_az{az:03d}_el{el:+03d}.npy"
        np.save(acfg.data.output_dir / npy_name, rec)

        meta = {
            "header": cfg.data.meta_header,
            "generated": cfg.data.run_tag,
            "wav": str(wav_path),
            "distance_m": acfg.grid.radius,
            "azimuth_deg": az,
            "elevation_deg": el,
            "n_mics": rec.shape[1],
            "script": __file__
        }
        with open(acfg.data.output_dir / (npy_name.replace('.npy', '.json')), 'w') as f:
            json.dump(meta, f, indent=2)


if __name__ == '__main__':
    # Add the project root to the Python path
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parents[2]  # Go up to s10_src directory
    wav_json = Path('clean_wav_registry.json')

    acfg = load_configs(project_root)