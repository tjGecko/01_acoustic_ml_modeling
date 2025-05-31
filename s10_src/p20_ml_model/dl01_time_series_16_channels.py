"""
PyTorch dataset for 16-channel time series data with lazy loading support.

This module provides a PyTorch Dataset and DataLoader implementation for
efficiently loading and batching 16-channel audio data for machine learning.
"""

import os
import json
import yaml
import torch
import random
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union, Callable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class TimeSeries16ChannelDataset(Dataset):
    """
    PyTorch Dataset for 16-channel time series data with lazy loading.
    
    Args:
        manifest_path: Path to the manifest JSON file
        data_dir: Base directory where the data files are stored. If None, uses the directory
                 containing the manifest file.
        transform: Optional transform to be applied to the data
        target_transform: Optional transform to be applied to the target
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        is_test: Whether this is a test dataset (uses test_dir instead of train_dir)
    """
    
    def __init__(
        self,
        manifest_path: Union[str, Path],
        data_dir: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        shuffle: bool = True,
        seed: Optional[int] = 42,
        is_test: bool = False
    ) -> None:
        """
        Initialize the dataset.
        
        Args:
            manifest_path: Path to the manifest JSON file
            data_dir: Base directory where the data files are stored. If None, uses the directory
                     containing the manifest file.
            transform: Optional transform to be applied to the data
            target_transform: Optional transform to be applied to the target
            shuffle: Whether to shuffle the dataset
            seed: Random seed for shuffling
            is_test: Whether this is a test dataset (uses test_dir instead of train_dir)
        """
        self.manifest_path = Path(manifest_path).resolve()
        self.data_dir = Path(data_dir).resolve() if data_dir else self.manifest_path.parent
        self.transform = transform
        self.target_transform = target_transform
        self.shuffle = shuffle
        self.seed = seed
        self.is_test = is_test
        
        # Load the manifest
        with open(self.manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Extract config and samples from the manifest
        self.config = manifest.get('config', {})
        if 'samples' not in manifest:
            raise ValueError("Manifest file must contain a 'samples' key")
            
        # Get the base directory for this dataset (train or test)
        training_splits = self.config.get('training_splits', {})
        self.base_dir = Path(training_splits.get('test_dir' if is_test else 'train_dir', ''))
        
        if not self.base_dir:
            raise ValueError("Manifest config is missing required 'training_splits' with 'train_dir' and 'test_dir'")
            
        # Make sure the base directory exists
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Base directory not found: {self.base_dir}")
        
        # Filter samples based on is_test flag
        self.samples = [
            {**sample, 'is_test': sample.get('is_test', False)}
            for sample in manifest['samples']
            if sample.get('is_test', False) == is_test
        ]
        
        if not self.samples:
            raise ValueError(f"No {'test' if is_test else 'training'} samples found in the manifest")
        
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(self.samples)
        
        # Add base_dir to each sample for path resolution
        for sample in self.samples:
            sample['base_dir'] = self.base_dir
        
        # Validate that all required files exist
        self._validate_files()
    
    def _validate_files(self) -> None:
        """
        Validate that all required files exist.
        Raises FileNotFoundError if any files are missing.
        """
        missing_files = []
        
        for i, sample in enumerate(self.samples):
            if 'segment_file' not in sample:
                raise ValueError(f"Sample at index {i} is missing required 'segment_file' field")
            
            try:
                # This will raise FileNotFoundError if the file doesn't exist
                file_path = self._find_data_file(sample)
            except FileNotFoundError as e:
                missing_files.append(str(e))
        
        if missing_files:
            error_msg = f"Error: {len(missing_files)} segment files not found. "
            error_msg += f"First error: {missing_files[0]}"
            raise FileNotFoundError(error_msg)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _find_data_file(self, sample_info: Dict) -> Path:
        """
        Find the data file path from the sample info.
        
        Args:
            sample_info: Dictionary containing file information from the manifest
            
        Returns:
            Path to the data file
        """
        if 'segment_file' not in sample_info:
            raise KeyError(f"Sample is missing required 'segment_file' field: {sample_info}")
        
        # Get the base directory that was set in __init__
        base_dir = sample_info.get('base_dir')
        if not base_dir:
            raise ValueError("Sample is missing required 'base_dir' field. This should be set during dataset initialization.")
            
        # Build the full path using the base directory and segment file
        file_path = Path(base_dir) / sample_info['segment_file']
        
        # Resolve any relative paths and normalize
        file_path = file_path.resolve()
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Segment file not found: {file_path}. "
                f"Original file: {sample_info.get('original_file', 'unknown')}. "
                f"Base dir: {base_dir}"
            )
            
        return file_path
    
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
    seed: Optional[int] = 42,
    **kwargs
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create data loaders for training and testing.
    
    Args:
        config_path: Path to the YAML configuration file
        batch_size: Batch size for both training and testing
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        shuffle_train: Whether to shuffle the training data
        shuffle_test: Whether to shuffle the test data
        seed: Random seed for reproducibility
        **kwargs: Additional keyword arguments to pass to the DataLoader
        
    Returns:
        Tuple of (train_loader, test_loader, data_info) where data_info is a dictionary
        containing information about the dataset
    """
    # Set random seeds for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    # Load the config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get the manifest path from config
    manifest_path = Path(config['training_splits']['manifest'])
    
    try:
        # Create training dataset
        train_dataset = TimeSeries16ChannelDataset(
            manifest_path=manifest_path,
            data_dir=manifest_path.parent,  # Use manifest directory as data_dir
            shuffle=shuffle_train,
            seed=seed,
            is_test=False  # This will filter for training samples
        )
        
        # Verify we can load at least one training sample
        try:
            train_sample = train_dataset[0]
            print(f"Successfully loaded training sample. Data shape: {train_sample[0].shape}")
        except Exception as e:
            raise RuntimeError(f"Failed to load training sample: {str(e)}") from e
        
        # Create test dataset
        test_dataset = TimeSeries16ChannelDataset(
            manifest_path=manifest_path,
            data_dir=manifest_path.parent,  # Use manifest directory as data_dir
            shuffle=shuffle_test,
            seed=seed,
            is_test=True  # This will filter for test samples
        )
        
        # Verify we can load at least one test sample
        try:
            test_sample = test_dataset[0]
            print(f"Successfully loaded test sample. Data shape: {test_sample[0].shape}")
        except Exception as e:
            raise RuntimeError(f"Failed to load test sample: {str(e)}") from e
            
    except Exception as e:
        error_msg = (
            "Failed to initialize datasets. Please check the configuration and manifest file.\n"
            f"Manifest path: {manifest_path}\n"
            f"Error: {str(e)}"
        )
        raise RuntimeError(error_msg) from e
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle_test,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs
    )
    
    # Prepare data info
    data_info = {
        'train_samples': len(train_dataset),
        'test_samples': len(test_dataset),
        'input_shape': train_sample[0].shape if 'train_sample' in locals() else None,
        'target_keys': list(train_sample[1].keys()) if 'train_sample' in locals() else []
    }
    
    return train_loader, test_loader, data_info


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