"""
Script to process numpy files for training/testing split using sliding window segmentation.

This script:
1. Reads configuration from c12_t10_training_split.yaml
2. Processes each numpy file in the input directory
3. Applies sliding window segmentation
4. Saves training and test slices
5. Creates a manifest file with metadata
"""

import os
import json
import hashlib
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import sys

# Add the project root to the Python path
project_root = Path(__file__).parents[3]
sys.path.append(str(project_root))

# Import the timer decorator
from s10_src.p55_util.f01_timer_decorator import exe_duration

def parse_angle_from_filename(filename: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse azimuth and elevation from filename.
    
    Expected format: 'basename_az123_el+045.npy' or 'basename_az045_el-015.npy'
    where az is 3-digit zero-padded and el is 3-digit with sign
    """
    try:
        stem = Path(filename).stem
        
        # Find the az and el parts using string operations
        az_start = stem.find('az')
        if az_start == -1:
            raise ValueError("Could not find 'az' in filename")
            
        # Extract az part (3 digits after 'az')
        az_str = stem[az_start+2:az_start+5]
        az = float(az_str)
        
        # Find el part (starts with 'el' and has 3 digits with optional sign)
        el_start = stem.find('el', az_start)
        if el_start == -1:
            raise ValueError("Could not find 'el' in filename")
            
        # Extract el part (sign + 3 digits after 'el')
        el_str = stem[el_start+2:el_start+6]  # includes sign and 3 digits
        el = float(el_str)
        
        return az, el
    except (ValueError, IndexError) as e:
        print(f"Warning: Could not parse angles from {filename}: {e}")
        return None, None

def generate_file_hash(data: np.ndarray) -> str:
    """Generate a hash from the first channel of the numpy array."""
    # Use first channel for hashing
    if len(data.shape) > 1:
        channel_data = data[0]
    else:
        channel_data = data
    
    # Convert to bytes and generate hash
    return hashlib.md5(channel_data.tobytes()).hexdigest()

def process_numpy_file(file_path: Path, output: Dict, config: Dict) -> None:
    """Process a single numpy file and update the output manifest."""
    try:
        # Check file size (approximately 2.3MB)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if not (2.2 <= file_size_mb <= 2.4):
            print(f"Warning: File {file_path} size {file_size_mb:.2f}MB is not in expected range ~2.3MB")
            
        # Load numpy file
        data = np.load(file_path)
        if data.size == 0:
            print(f"Warning: Empty file {file_path}")
            return
            
        # Validate and transpose array if needed
        if len(data.shape) != 2:
            print(f"Warning: Expected 2D array, got shape {data.shape} in {file_path}")
            return
            
        # Check if we need to transpose (time, channels) to (channels, time)
        if data.shape[1] == 16:  # If second dimension is 16, it's (time, channels)
            data = data.T  # Transpose to (channels, time)
        elif data.shape[0] != 16:  # If first dimension is not 16, it's invalid
            print(f"Warning: Expected 16 channels, got shape {data.shape} in {file_path}")
            return
            
        sample_count = data.shape[1]
        if not (17000 <= sample_count <= 18000):  # Adjusted range for actual data
            print(f"Warning: Expected ~17.6k samples, got {sample_count} in {file_path}")
            return
            
        # Parse angles from filename
        az, el = parse_angle_from_filename(file_path.name)
        
        # Generate file hash
        # file_hash = generate_file_hash(data)
        
        # Get segmentation parameters
        samples_per_segment = config['segmentation']['samples_per_segment']
        step = config['segmentation']['sliding_win_offset']
        
        # Calculate number of segments
        num_segments = (data.shape[-1] - samples_per_segment) // step + 1
        
        if num_segments < 1:
            print(f"Warning: File {file_path} is too small for segmentation. Has {sample_count} samples, needs at least {samples_per_segment}")
            print(f"File info: {file_path}, shape: {data.shape}, size: {file_size_mb:.2f}MB")
            return
        
        # Process each segment
        for i in range(num_segments):
            start = i * step
            end = start + samples_per_segment
            segment = data[..., start:end]
            
            # Apply min-max scaling to the segment
            min_val = np.min(segment, axis=-1, keepdims=True)
            max_val = np.max(segment, axis=-1, keepdims=True)
            # Avoid division by zero in case of constant segment
            range_val = max_val - min_val
            range_val[range_val == 0] = 1.0  # Set range to 1.0 to avoid division by zero
            # Scale to [0, 1] range
            segment = (segment - min_val) / range_val
            
            # Generate unique filename using hash and segment index
            segment_hash = generate_file_hash(segment)
            segment_filename = f"{segment_hash}.npy"
            
            # Determine if this is the last segment (test) or not (train)
            is_test = (i == num_segments - 1)
            output_dir = Path(config['training_splits']['test_dir' if is_test else 'train_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save segment
            np.save(output_dir / segment_filename, segment)
            
            # Add to manifest
            output['samples'].append({
                'original_file': file_path.name,
                'segment_file': segment_filename,
                'is_test': is_test,
                'azimuth': az,
                'elevation': el,
                'segment_index': i,
                'total_segments': num_segments
            })
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

@exe_duration
def main():
    project_dir = Path(__file__).parents[3]
    # Load configuration
    config_path = project_dir / "05_config/c12_t10_training_split.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize output manifest
    manifest = {
        'config': config,
        'samples': []
    }
    
    # Get input directory and find all numpy files
    input_dir = Path(config['training_splits']['input_dir'])
    npy_files = list(input_dir.glob('*.npy'))
    
    if not npy_files:
        print(f"No .npy files found in {input_dir}")
        return
    
    print(f"Found {len(npy_files)} .npy files to process")
    
    # Process each file
    for file_path in tqdm(npy_files, desc="Processing files"):
        process_numpy_file(file_path, manifest, config)
    
    # Save manifest
    manifest_path = Path(config['training_splits']['manifest'])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"Manifest saved to: {manifest_path}")
    print(f"Training samples: {sum(1 for s in manifest['samples'] if not s['is_test'])}")
    print(f"Test samples: {sum(1 for s in manifest['samples'] if s['is_test'])}")

if __name__ == "__main__":
    main()
