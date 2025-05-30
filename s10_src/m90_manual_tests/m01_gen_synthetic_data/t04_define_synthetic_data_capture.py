import itertools
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import shutil
import numpy as np, xml.etree.ElementTree as ET
import pyroomacoustics as pra
import soundfile as sf

from s10_src.m05_data_models.d04_acoustic_sim_env import AcousticSimulationConfig


def load_configs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Path to the YAML configuration file
    project_root = Path(cfg['header']['project_root'])
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

    temp = {
        'AcousticSimulationConfig': config,
        'mic_coords': mic_coords,
        'wav_cfgs': wav_json,
        'provenance': {
            'info': 'Files to copy in case of config changes',
            'config_path': str(config_path),
            'mic_xml_path': str(config.mics.xml_path),
            'wav_json_path': str(wav_json_path),
        }
    }

    merged = {**cfg, **temp}

    return merged


def load_mic_xml(path):
    tree = ET.parse(path)
    xyz = []
    # Corrected: Find 'pos' elements. The './/' means anywhere in the tree.
    for pos_element in tree.findall(".//pos"):
        try:
            # Corrected: Get 'x', 'y', 'z' as attributes
            x = float(pos_element.get("x"))
            y = float(pos_element.get("y"))
            z = float(pos_element.get("z"))
            xyz.append((x, y, z))
        except (TypeError, ValueError) as e:
            # Handle cases where an attribute might be missing or not a float
            print(f"Warning: Could not parse attributes for a <pos> element in {path}: {e}")
            continue # Skip this problematic element

    if not xyz:
        # This check is crucial. If it's still empty, something is wrong.
        raise ValueError(f"No microphone positions found in XML file: {path}. "
                         "Check XPath './/pos' and ensure 'x', 'y', 'z' attributes exist and are valid numbers.")
    return np.array(xyz).T

def define_virtual_env(cfg: Dict[str, Any]) -> Dict[str, Any]:
    acfg = cfg['AcousticSimulationConfig']
    room_cfg = acfg.pyroomacoustics
    room_dims = np.array(room_cfg.dimensions) # For validation

    # ---------- build room ----------
    # https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.materials.database.html
    materials = pra.make_materials(
        ceiling=room_cfg.abs_ceiling,
        floor=room_cfg.abs_floor,
        east=room_cfg.abs_wall,
        west=room_cfg.abs_wall,
        north=room_cfg.abs_wall,
        south=room_cfg.abs_wall
    )

    room = pra.ShoeBox(
        room_cfg.dimensions,
        fs=room_cfg.fs_hz,
        max_order=room_cfg.max_order,
        materials=materials,
        air_absorption=room_cfg.air_absorption,
    )

    mic_center = np.array(acfg.mics.position).reshape(3, 1)
    mic_locations = cfg['mic_coords'] + mic_center

    # Validate microphone positions
    eps = 1e-6
    if not np.all(
        (mic_locations > eps) & (mic_locations < room_dims[:, np.newaxis] - eps)
    ):
        # Find problematic mics for better error message
        problem_mics = mic_locations[:, np.any(~((mic_locations > eps) & (mic_locations < room_dims[:, np.newaxis] - eps)), axis=0)]
        raise ValueError(f"One or more microphone positions are outside or on boundary of room {room_dims}. Problematic mic coordinates (subset):\n{problem_mics.T.round(3)}")

    mic_array = pra.MicrophoneArray(mic_locations, room.fs)
    room.add_microphone_array(mic_array)

    cfg['sim_env'] = {'room': room, 'mic_array': mic_array, 'room_cfg': room_cfg} # Pass room_cfg for generate_data
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
    # Ensure acfg.mics.position is treated as a numpy array for element-wise addition
    mic_pos_array = np.array(acfg.mics.position)
    src_pos = mic_pos_array + acfg.grid.radius * src_vec

    return src_pos


def generate_data(cfg: Dict[str, Any], wav_path: Path, az: int, el: int) -> None:  # Return type should be None
    sim_env = cfg['sim_env']
    room = sim_env['room']
    acfg = cfg['AcousticSimulationConfig']
    room_dims = np.array(acfg.pyroomacoustics.dimensions)
    output_dir = Path(acfg.capture_metadata.output_dir)  # Corrected path

    # --- place speaker (e.g. virtual drone) ---
    src_pos = get_virtual_speaker_position(az, el, acfg)

    # Validate source position: must be strictly inside room
    eps = 1e-6  # A small margin from the walls
    if not (
            (src_pos[0] > eps) and (src_pos[0] < room_dims[0] - eps) and
            (src_pos[1] > eps) and (src_pos[1] < room_dims[1] - eps) and
            (src_pos[2] > eps) and (src_pos[2] < room_dims[2] - eps)
    ):
        print(
            f"Warning: Source position {src_pos.round(3)} for az={az}, el={el} is outside or on boundary of room {room_dims}. Skipping this configuration.")
        # Clean up room.sources if it was modified by a previous failed attempt in a reused room (not the case here as room is rebuilt)
        return  # Skip this configuration

    signal, sr = sf.read(wav_path.resolve(), dtype='float32')
    if sr != room.fs:
        print(f"Warning: WAV file sr ({sr}Hz) and room.fs ({room.fs}Hz) differ. Pyroomacoustics will resample.")

    # Check if room.sources already exists (it shouldn't if room is fresh)
    # If sources were being appended to a persistent room, clear them first: room.sources = []
    room.add_source(src_pos, signal=signal)

    try:
        room.simulate()
    except ValueError as e:
        if "zero-size array to reduction operation maximum" in str(e):
            print(f"ValueError during simulation for az={az}, el={el}, src_pos={src_pos.round(3)}: {e}")
            print(
                "This likely means no valid RIR was generated. Review source/mic positions, room parameters, and absorption settings.")
            if room.sources:  # Clean up for safety, though room is rebuilt each time
                room.sources.pop()
            return
        else:
            raise  # Re-raise other ValueErrors

    if room.mic_array.signals is None or room.mic_array.signals.size == 0:
        print(
            f"Warning: Simulation for az={az}, el={el}, src_pos={src_pos.round(3)} resulted in empty signals. Skipping saving.")
        if room.sources:
            room.sources.pop()
        return

    rec = room.mic_array.signals.T  # shape: (nsamples, n_mics)

    output_dir = Path(cfg['header']['data_dir_path'])
    npy_name = f"{wav_path.stem}_az{az:03d}_el{el:+03d}.npy"
    abs_path = output_dir / npy_name
    np.save(abs_path, rec)
    print(f'Numpy: {abs_path}')

    # Clean up the source from room.sources list. This is crucial if the same room object
    # were to be used for another simulation call without re-initialization.
    # Since define_virtual_env creates a new room object in each loop iteration,
    # this pop is for strict correctness if that pattern changed.
    if room.sources:
        room.sources.pop()


def define_save_location(cfg: Dict[str, Any]) -> None:
    acfg = cfg['AcousticSimulationConfig']
    output_dir = Path(acfg.capture_metadata.output_dir)
    today = datetime.today().strftime('%Y_%m_%d')
    capture_dir = output_dir / f'capture_{today}'
    provenance_dir = capture_dir / 'provenance'
    data_dir_path = capture_dir / 'data'

    # Ensure directories exist
    provenance_dir.mkdir(parents=True, exist_ok=True)
    data_dir_path.mkdir(parents=True, exist_ok=True)
    cfg['header']['provenance_dir'] = provenance_dir
    cfg['header']['provenance_dir_info'] = 'Config snapshots for this run'
    cfg['header']['data_dir_path'] = data_dir_path
    cfg['header']['data_dir_path_info'] = 'Data generated for this run'

    # Copy provenance files with original filenames
    for key in ['config_path', 'mic_xml_path', 'wav_json_path']:
        src_path = Path(cfg['provenance'][key])
        dst_path = provenance_dir / src_path.name
        shutil.copy(src_path, dst_path)

    run_config_path = provenance_dir / 'run_config.json'
    with run_config_path.open('w', encoding='utf-8') as fout:
        json.dump(cfg, fout, indent=4, default=str)

    print(f'Wrote: {run_config_path}')

    return cfg


def generate_data_loop(cfg: Dict[str, Any]) -> None:
    acfg = cfg['AcousticSimulationConfig']
    wav_cfgs = cfg['wav_cfgs']['entries']

    # Build angles for grid coverage
    az_vals = range(acfg.grid.az_start, acfg.grid.az_end + 1, acfg.grid.step)
    el_vals = range(acfg.grid.el_start, acfg.grid.el_end + 1, acfg.grid.step)

    # debug_ct = 0
    # debug_trigger = 3

    for az, el in itertools.product(az_vals, el_vals):
        # if debug_trigger > 0 and debug_ct >= debug_trigger :
        #     print(f"Debug limit ({debug_trigger} iterations) reached.")
        #     break
        # debug_ct += 1
        # print(f"\nProcessing: Azimuth={az}, Elevation={el} (Iteration {debug_ct})")

        for wav_cfg in wav_cfgs:
            wav_path = Path(wav_cfg['file_path'])
            # Re-define the environment for each run to ensure a clean state
            current_run_cfg_modified = define_virtual_env(cfg.copy()) # Pass a copy if define_virtual_env modifies input dict cfg directly
            generate_data(current_run_cfg_modified, wav_path, az, el) # Does not return/reassign


if __name__ == '__main__':
    # Add the project root to the Python path
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parents[2]  # Go up to s10_src directory
    wav_json = Path('clean_wav_registry.json')

    cfg = {
        'header': {
            "author": 'TJ Hoeft',
            "author_info": 'Who made the capture',
            "date": datetime.now().isoformat(),
            "date_info": 'When the capture was made',
            "script_file": __file__,
            "script_file_info": 'What script generated the synthetic capture',
            'project_root': project_root,
            'project_root_info': 'Where the source is stored',
            'git_hash': 'TODO',
            'git_hash_info': 'What version of the script was used',
            'wav_json': wav_json,
            'wav_json_info': 'What files were played through virtual speaker (e.g. drone)',
        }
    }

    cfg = load_configs(cfg)
    cfg = define_save_location(cfg)

    start = time.time()
    generate_data_loop(cfg)
    duration_seconds = time.time() - start
    readable = time.strftime('%H:%M:%S', time.gmtime(duration_seconds))
    print(f"Function runtime: {readable}")