"""
Test script to load and validate the acoustic simulation configuration from YAML.
"""

from pathlib import Path
import sys
import os
import importlib.util
from pprint import pprint

from s10_src.m05_data_models.d04_acoustic_sim_env import AcousticSimulationConfig

# Add the project root to the Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[2]  # Go up to s10_src directory
# sys.path.insert(0, str(project_root.parent))
#
# # Import the module directly from the file path
# module_path = project_root / 'm05_data_models' / 'd04_acoustic_sim_env.py'
# spec = importlib.util.spec_from_file_location("acoustic_sim_env", module_path)
# acoustic_sim_env = importlib.util.module_from_spec(spec)
# sys.modules["acoustic_sim_env"] = acoustic_sim_env
# spec.loader.exec_module(acoustic_sim_env)
# AcousticSimulationConfig = acoustic_sim_env.AcousticSimulationConfig

def main():
    # Path to the YAML configuration file
    config_path = project_root / '05_config' / 'c11_syn_acoustic_data_gen.yaml'
    
    print(f"Loading configuration from: {config_path}")
    
    try:
        # Load and validate the configuration
        config = AcousticSimulationConfig.from_yaml(config_path)
        print("\n✅ Configuration loaded and validated successfully!")
        
        # Print the configuration in a readable format
        print("\nConfiguration details:")
        print("=" * 80)
        
        # Pyroomacoustics settings
        print("\nPyroomacoustics Settings:")
        print("-" * 40)
        print(f"Dimensions (x,y,z): {config.pyroomacoustics.dimensions} m")
        print(f"Sample rate: {config.pyroomacoustics.fs_hz} Hz")
        print(f"Max order: {config.pyroomacoustics.max_order}")
        print(f"Wall absorption: {config.pyroomacoustics.abs_wall}")
        print(f"Floor absorption: {config.pyroomacoustics.abs_floor}")
        print(f"Ceiling absorption: {config.pyroomacoustics.abs_ceiling}")
        print(f"Air absorption: {config.pyroomacoustics.air_absorption}")
        
        # Microphone settings
        print("\nMicrophone Configuration:")
        print("-" * 40)
        print(f"XML Path: {config.mics.xml_path}")
        print(f"Position (x,y,z): {config.mics.position} m")
        print(f"Array Type: {config.mics.mic_array_type}")
        print(f"Info: {config.mics.mic_array_info}")
        
        # Grid settings
        print("\nSpeaker Grid Configuration:")
        print("-" * 40)
        print(f"Radius: {config.grid.radius} m")
        print(f"Azimuth Range: {config.grid.az_start}° to {config.grid.az_end}°")
        print(f"Elevation Range: {config.grid.el_start}° to {config.grid.el_end}°")
        print(f"Step Size: {config.grid.step}°")
        
        # Capture metadata
        print("\nCapture Metadata:")
        print("-" * 40)
        print(f"WAV Files JSON: {config.capture_metadata.wav_files_json}")
        print(f"Output Directory: {config.capture_metadata.output_dir}")
        print(f"Author: {config.capture_metadata.author}")
        print(f"Author Info: {config.capture_metadata.author_info}")
        print(f"Capture Type: {config.capture_metadata.capture_type}")
        print(f"Capture Approach: {config.capture_metadata.capture_approach}")
        print(f"Capture Info: {config.capture_metadata.capture_info}")
        
        # Test saving the configuration back to a YAML file
        test_output_path = project_root / '99_tmp' / 'test_acoustic_config.yaml'
        test_output_path.parent.mkdir(parents=True, exist_ok=True)
        config.to_yaml(test_output_path)
        print(f"\n✅ Test configuration saved to: {test_output_path}")
        
    except Exception as e:
        print(f"\n❌ Error loading configuration:", file=sys.stderr)
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()