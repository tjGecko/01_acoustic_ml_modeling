"""
Script to scan and filter WAV files based on configuration.

This script:
1. Loads a YAML configuration file containing a root directory path
2. Scans for WAV files within that directory and its subdirectories
3. Filters files to only include those containing 'bebop' or 'membo' in their names
4. Creates a structured registry with metadata and file information
5. Saves the registry to a JSON file
"""

import yaml
import sys
from pathlib import Path
from typing import List, Dict, Any, Set
from datetime import datetime

from s10_src.m05_data_models.d02_clean_wav_registry import (
    CleanWavRegistry,
    DroneType
)

def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load and parse the YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        dict: Parsed configuration
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If there's an error parsing the YAML
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    return config

def find_wav_files(root_dir: Path, filter_terms: List[str] = None) -> List[Path]:
    """
    Recursively find WAV files in the specified directory, optionally filtered by terms.
    
    Args:
        root_dir: Root directory to search in
        filter_terms: List of substrings to filter filenames by (case-insensitive).
                     If None or empty, no filtering is applied.
        
    Returns:
        List of Path objects for each matching WAV file found
    """
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")
    
    # Get all WAV files
    wav_files = list(root_dir.rglob('*.wav'))
    
    # Apply filters if any terms are provided
    if filter_terms:
        filter_terms = [term.lower() for term in filter_terms]
        wav_files = [
            f for f in wav_files
            if any(term in str(f).lower() for term in filter_terms)
        ]
        
    return wav_files


def determine_drone_type(file_path: Path) -> DroneType:
    """Determine the drone type based on the file path."""
    file_str = str(file_path).lower()
    if 'bebop' in file_str:
        return DroneType.BEBOP
    elif 'membo' in file_str:
        return DroneType.MEMBO
    else:
        # Default to BEBOP if no clear match (shouldn't happen with proper filtering)
        return DroneType.BEBOP

def main():
    """Main function to execute the script."""
    try:
        # Define absolute paths
        config_path = Path("/home/tj/02_Windsurf_Projects/r03_Gimbal_Angle_Root/05_config/training_data.yaml")
        output_path = Path("/home/tj/02_Windsurf_Projects/r03_Gimbal_Angle_Root/05_config/clean_wav_registry.json")
        
        # Assert the config file exists at the expected location
        assert config_path.exists(), (
            f"Configuration file not found at expected location: {config_path}\n"
            "Please ensure the file exists at this exact path."
        )
        
        # Define filter terms and script info
        filter_terms = ['bebop', 'membo']
        script_name = Path(__file__).name
        
        # Load configuration
        print(f"Loading configuration from: {config_path}")
        config = load_config(config_path)
        
        # Get root directory from config
        root_dir = Path(config['data_lake']['root_dir']).expanduser()
        print(f"Scanning for WAV files in: {root_dir}")
        print(f"Filtering for files containing: {', '.join(filter_terms)}")
        
        # Find and filter WAV files
        wav_files = find_wav_files(root_dir, filter_terms)
        
        # Create registry
        registry = CleanWavRegistry.create(
            created_by=script_name,
            filter_terms=filter_terms,
            root_dir=root_dir,
            description="Registry of clean WAV files for training data generation"
        )
        
        # Add files to registry
        for file_path in wav_files:
            drone_type = determine_drone_type(file_path)
            try:
                registry.add_entry(file_path, drone_type)
            except Exception as e:
                print(f"Warning: Could not add {file_path}: {e}")
        
        # Print summary
        print(f"\nFound {len(registry.entries)} matching WAV files:")
        for drone_type, count in registry.count_by_drone_type.items():
            print(f"- {drone_type.value.upper()}: {count} files")
        
        # Save registry
        registry.save_to_file(output_path)
        print(f"\nSaved registry to: {output_path}")
        
        # Show sample of files (first 5 of each type)
        print("\nSample files:")
        samples = {}
        for entry in registry.entries:
            if entry.drone_type not in samples:
                samples[entry.drone_type] = []
            if len(samples[entry.drone_type]) < 5:
                samples[entry.drone_type].append(entry.file_path.relative_to(root_dir))
        
        for drone_type, files in samples.items():
            print(f"\n{drone_type.value.upper()} samples:")
            for file_path in files:
                print(f"- {file_path}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()