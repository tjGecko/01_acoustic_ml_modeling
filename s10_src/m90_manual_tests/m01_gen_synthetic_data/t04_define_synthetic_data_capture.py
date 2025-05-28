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
    room_cfg = acfg.pyroomacoustics        # ← rename for clarity

    # ---------- build room ----------
    # absorption = [room_cfg.abs_wall] * 4 + [room_cfg.abs_floor, room_cfg.abs_ceiling]
    # https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.materials.database.html

    room = pra.ShoeBox(
        room_cfg.dimensions,
        fs=room_cfg.fs_hz,
        max_order=room_cfg.max_order,
        materials=pra.Material('fibre_absorber_2'),
        air_absorption=room_cfg.air_absorption,
    )

    mic_center = np.c_[acfg.mics.position].T         # shape (3,1)
    mic_array = pra.MicrophoneArray(cfg['mic_coords'] + mic_center, room_cfg.fs_hz)
    room.add_microphone_array(mic_array)

    cfg['sim_env'] = {'room': room, 'mic_array': mic_array}
    return cfg


def get_virtual_speaker_position(azimuth_deg, elevation_deg, acfg: AcousticSimulationConfig):
    # Convert azimuth and elevation to radians
    az_rad = np.deg2rad(azimuth_deg)
    el_rad = np.deg2rad(elevation_deg)

    # Calculate direction vector components
    x = np.cos(el_rad) * np.cos(az_rad)
    y = np.cos(el_rad) * np.sin(az_rad)
    z = np.sin(el_rad)
    src_vec = np.array([x, y, z])

    # Compute source position
    src_pos = acfg.mics.position + acfg.grid.radius * src_vec

    return src_pos


def generate_data(cfg: Dict[str, Any], wav_path, az, el) -> Dict[str, Any]:
    room = cfg['sim_env']['room']
    acfg = cfg['AcousticSimulationConfig']

    # --- place speaker (e.g. virtual drone) ---
    src_pos = get_virtual_speaker_position(az, el, acfg)

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
        # "header": cfg.data.meta_header,
        # "generated": cfg.data.run_tag,
        # "wav": str(wav_path),
        # "distance_m": acfg.grid.radius,
        # "azimuth_deg": az,
        # "elevation_deg": el,
        # "n_mics": rec.shape[1],
        "script": __file__
    }
    with open(acfg.data.output_dir / (npy_name.replace('.npy', '.json')), 'w') as f:
        json.dump(meta, f, indent=2)


def generate_data_loop(cfg: Dict[str, Any]) -> None:
    acfg = cfg['AcousticSimulationConfig']
    wav_prov = cfg['wav_cfgs']['header']
    wav_cfgs = cfg['wav_cfgs']['entries']

    # Build angles for grid coverage
    az_vals = range(acfg.grid.az_start, acfg.grid.az_end + 1, acfg.grid.step)
    el_vals = range(acfg.grid.el_start, acfg.grid.el_end + 1, acfg.grid.step)
    debug_ct = 0
    debug_trigger = 3

    for az, el in itertools.product(az_vals, el_vals):
        if debug_ct < debug_trigger:
            debug_ct += 1
        else:
            break

        wav_path = Path(wav_cfgs[0]['file_path'])  # fixme: Hard coded
        cfg = define_virtual_env(cfg)
        cfg = generate_data(cfg, wav_path, az, el)


if __name__ == '__main__':
    # Add the project root to the Python path
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parents[2]  # Go up to s10_src directory
    wav_json = Path('clean_wav_registry.json')

    cfg = load_configs(project_root)
    generate_data_loop(cfg)

