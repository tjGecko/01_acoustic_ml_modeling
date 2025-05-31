"""
Test script to verify the dataset loading functionality.
"""
import sys
import os
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
# sys.path.append(str(project_root))

from s10_src.p20_ml_model.dl01_time_series_16_channels import create_data_loaders

def main():

    # Path to the config file - using relative path from project root
    config_path = project_root / ".." / "05_config" / "c12_t10_training_split.yaml"
    config_path = config_path.resolve()  # Convert to absolute path
    
    print(f"Using config file: {config_path}")
    
    # Load the config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\nConfiguration:")
    print(yaml.dump(config, default_flow_style=False))
    
    # Verify training and test directories exist
    train_dir = Path(config['training_splits']['train_dir'])
    test_dir = Path(config['training_splits']['test_dir'])
    manifest_path = Path(config['training_splits']['manifest'])
    
    print("\nChecking directories and files:")
    print(f"Training directory exists: {train_dir.exists()}")
    print(f"Test directory exists: {test_dir.exists()}")
    print(f"Manifest file exists: {manifest_path.exists()}")
    
    if not all([train_dir.exists(), test_dir.exists(), manifest_path.exists()]):
        print("\nError: One or more required directories/files do not exist.")
        return
    
    # Try to create data loaders with a very small batch size
    print("\nCreating data loaders...")
    try:
        train_loader, test_loader, _ = create_data_loaders(
            config_path=config_path,
            batch_size=1,  # Small batch size for testing
            num_workers=0,  # Don't use multiprocessing for testing
            pin_memory=False,
            shuffle_train=False,
            shuffle_test=False
        )
        
        print("\nSuccessfully created data loaders!")
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of test batches: {len(test_loader)}")
        
        # Try to load one batch from each
        print("\nLoading one training batch...")
        train_batch = next(iter(train_loader))
        print(f"Training batch - data shape: {train_batch[0].shape}, target keys: {train_batch[1].keys()}")
        
        print("\nLoading one test batch...")
        test_batch = next(iter(test_loader))
        print(f"Test batch - data shape: {test_batch[0].shape}, target keys: {test_batch[1].keys()}")
        
    except Exception as e:
        print(f"\nError creating data loaders: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
