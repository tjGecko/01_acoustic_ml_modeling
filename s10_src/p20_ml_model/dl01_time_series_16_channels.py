"""
PyTorch dataset for 16-channel time series data with lazy loading support.

This module provides a PyTorch Dataset and DataLoader implementation for
efficiently loading and batching 16-channel audio data for machine learning.
"""

import os
import json
import torch
import random
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class TimeSeries16ChannelDataset(Dataset):
    """
    PyTorch Dataset for 16-channel time series data with lazy loading.
    
    Args:
        manifest_path: Path to the manifest JSON file
        data_dir: Base directory containing the data files
        transform: Optional transform to apply to the data
        target_transform: Optional transform to apply to the targets
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        manifest_path: Union[str, Path],
        data_dir: Optional[Union[str, Path]] = None,
        transform=None,
        target_transform=None,
        shuffle: bool = True,
        seed: int = 42
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.data_dir = Path(data_dir) if data_dir else self.manifest_path.parent
        self.transform = transform
        self.target_transform = target_transform
        
        # Load manifest
        with open(self.manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        # Store samples and their paths
        self.samples = self.manifest.get('samples', [])
        
        # Set random seed if shuffle is True
        if shuffle:
            random.seed(seed)
            random.shuffle(self.samples)
        
        # Verify all files exist
        self._validate_files()
    
    def _validate_files(self) -> None:
        """Verify that all files in the manifest exist."""
        missing_files = []
        for sample in self.samples:
            # Handle both relative and absolute paths in the manifest
            segment_file = sample.get('segment_file')
            if not segment_file:
                segment_file = sample.get('original_file', '')
            
            # Try both the segment file name and the original file name
            possible_paths = [
                self.data_dir / segment_file,
                self.data_dir / Path(segment_file).name,
                self.data_dir / sample.get('original_file', '')
            ]
            
            # Check if any of the possible paths exist
            if not any(p.exists() for p in possible_paths if p):
                missing_files.append(segment_file)
        
        if missing_files:
            print(f"Warning: {len(missing_files)} files listed in manifest not found. "
                  f"First missing file: {missing_files[0]}")
            print("This might be expected if you're using a subset of the data.")
            # Don't raise an error, just warn and continue with the files we have
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _find_data_file(self, sample_info: Dict) -> Path:
        """Find the correct data file path from sample info."""
        # Try different possible file paths
        possible_paths = [
            self.data_dir / sample_info['segment_file'],
            self.data_dir / Path(sample_info['segment_file']).name,
            self.data_dir / sample_info.get('original_file', '')
        ]
        
        for path in possible_paths:
            if path and path.exists():
                return path
        
        raise FileNotFoundError(
            f"Could not find data file for sample: {sample_info}"
        )
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get a single item from the dataset.
        
        Returns:
            tuple: (sample, target) where sample is the audio data and target is a dict
                   containing metadata including azimuth and elevation.
        """
        sample_info = self.samples[idx]
        
        try:
            # Find and load the numpy file
            file_path = self._find_data_file(sample_info)
            data = np.load(file_path)
            
            # Ensure data is in the correct shape (channels, time)
            if len(data.shape) == 2 and data.shape[0] > data.shape[1]:
                data = data.T  # Transpose to (channels, time)
            
            # Convert to PyTorch tensor and ensure float32
            data = torch.from_numpy(data).float()
            
            # Apply transforms if specified
            if self.transform:
                data = self.transform(data)
            
            # Prepare target
            target = {
                'azimuth': float(sample_info.get('azimuth', 0)),
                'elevation': float(sample_info.get('elevation', 0)),
                'original_file': sample_info.get('original_file', ''),
                'is_test': sample_info.get('is_test', False)
            }
            
            if self.target_transform:
                target = self.target_transform(target)
            
            return data, target
            
        except Exception as e:
            print(f"Error loading {sample_info.get('original_file', 'unknown')}: {e}")
            # Return a zero tensor of the expected shape if there's an error
            expected_shape = (16, self.manifest.get('segmentation', {}).get('samples_per_segment', 4096))
            zero_data = torch.zeros(expected_shape, dtype=torch.float32)
            return zero_data, {'azimuth': 0, 'elevation': 0, 'original_file': 'error', 'is_test': False}


def create_data_loaders(
    config_path: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle_train: bool = True,
    shuffle_test: bool = False,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Create training and validation data loaders from a config file.
    
    Args:
        config_path: Path to the training split config YAML file
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        shuffle_train: Whether to shuffle training data
        shuffle_test: Whether to shuffle test data
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (train_loader, test_loader, config) where config is the loaded config dict
    """
    import yaml
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create datasets
    train_dir = Path(config['training_splits']['train_dir'])
    test_dir = Path(config['training_splits']['test_dir'])
    manifest_path = Path(config['training_splits']['manifest'])
    
    train_dataset = TimeSeries16ChannelDataset(
        manifest_path=manifest_path,
        data_dir=train_dir,
        shuffle=shuffle_train,
        seed=seed
    )
    
    test_dataset = TimeSeries16ChannelDataset(
        manifest_path=manifest_path,
        data_dir=test_dir,
        shuffle=shuffle_test,
        seed=seed
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle_test,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, test_loader, config


def get_sample_shape(config_path: Union[str, Path]) -> Tuple[int, int]:
    """
    Get the shape of a single sample from the dataset.
    
    Args:
        config_path: Path to the training split config YAML file
        
    Returns:
        tuple: (n_channels, sequence_length)
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return (16, config['segmentation']['samples_per_segment'])


# Example usage
if __name__ == "__main__":
    # Example configuration
    config_path = "05_config/c12_t10_training_split.yaml"
    
    # Create data loaders
    train_loader, test_loader, config = create_data_loaders(
        config_path=config_path,
        batch_size=32,
        num_workers=4
    )
    
    # Get sample shape
    n_channels, seq_len = get_sample_shape(config_path)
    print(f"Sample shape: ({n_channels}, {seq_len})")
    
    # Example training loop
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Data shape: {data.shape}")
        print(f"  Azimuth: {target['azimuth']}")
        print(f"  Elevation: {target['elevation']}")
        
        if batch_idx >= 2:  # Just show first 3 batches
            break