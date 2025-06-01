import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import the model and dataset classes
from m03_m01_gccphat_cnn import SoundSourceLocalizationCNN
from dl01_time_series_16_channels import TimeSeries16ChannelDataset, create_data_loaders

# Configuration
CHECKPOINT_DIR = Path("/home/tj/99_tmp/11 - synthetic mic array data/02_training_data/checkpoints")
RESULTS_DIR = Path("/home/tj/99_tmp/11 - synthetic mic array data/02_training_data/prediction_results")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_best_model(checkpoint_dir, model_params, device='cpu'):
    """Load the best model from checkpoints."""
    # Look for the specific model file
    best_checkpoint = Path(checkpoint_dir) / "best_model.pth"
    if not best_checkpoint.exists():
        # Fallback to looking for any .pt file if best_model.pth doesn't exist
        checkpoints = list(Path(checkpoint_dir).glob("*.pth")) + list(Path(checkpoint_dir).glob("*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No model checkpoints found in {checkpoint_dir}")
        best_checkpoint = checkpoints[0]  # Take the first one found
        print(f"Warning: best_model.pth not found, using {best_checkpoint} instead")
    
    print(f"Loading model from {best_checkpoint}")
    
    # Initialize model
    model = SoundSourceLocalizationCNN(**model_params)
    try:
        # Try loading with map_location to handle different device configurations
        state_dict = torch.load(best_checkpoint, map_location=device)
        # Handle case where the model was saved as a dict with 'model_state_dict' key
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        print("Trying to load with strict=False...")
        model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    return model

def denormalize_angles(normalized_angles, angle_ranges):
    """Convert normalized angles back to original ranges."""
    # Assuming normalized_angles are in range [-1, 1]
    # and angle_ranges is a dict with 'azimuth' and 'elevation' ranges
    denorm_angles = normalized_angles.copy()
    for i, angle_type in enumerate(['azimuth', 'elevation']):
        min_val, max_val = angle_ranges[angle_type]
        denorm_angles[:, i] = (normalized_angles[:, i] + 1) * (max_val - min_val) / 2 + min_val
    return denorm_angles

def evaluate_model(model, test_loader, device='cpu'):
    """Run evaluation on the test set and return predictions and ground truth."""
    all_preds = []
    all_targets = []
    all_files = []
    
    # Print dataloader info
    print("\nDataLoader Info:")
    print(f"Number of batches: {len(test_loader)}")
    if len(test_loader) > 0:
        sample_batch = next(iter(test_loader))
        print(f"Sample batch type: {type(sample_batch)}")
        print(f"Sample batch length: {len(sample_batch) if hasattr(sample_batch, '__len__') else 'N/A'}")
        print(f"Sample batch[0] shape: {sample_batch[0].shape if hasattr(sample_batch[0], 'shape') else 'N/A'}")
        if len(sample_batch) > 1:
            print(f"Sample batch[1] type: {type(sample_batch[1])}")
            if isinstance(sample_batch[1], dict):
                print("Sample batch[1] keys:", sample_batch[1].keys())
            elif hasattr(sample_batch[1], 'shape'):
                print(f"Sample batch[1] shape: {sample_batch[1].shape}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            print(f"\nProcessing batch {batch_idx+1}/{len(test_loader)}")
            
            # Handle different batch formats
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, targets = batch
                print(f"Batch contains inputs: {type(inputs)}, targets: {type(targets)}")
            else:
                print(f"Unexpected batch format: {type(batch)}")
                continue
                
            # Move inputs to device
            inputs = inputs.to(device)
            print(f"Input shape: {inputs.shape}")
            
            # Forward pass
            outputs = model(inputs)
            print(f"Output shape: {outputs.shape}")
            
            # Store predictions
            all_preds.append(outputs.cpu().numpy())
            
            # Process targets
            print(f"Target type: {type(targets)}")
            
            if isinstance(targets, dict):
                # Handle dictionary format
                print("Target keys:", targets.keys())
                if 'azimuth' in targets and 'elevation' in targets:
                    batch_targets = torch.stack([
                        targets['azimuth'].float(),
                        targets['elevation'].float()
                    ], dim=1).numpy()
                    all_targets.append(batch_targets)
                    all_files.extend(targets.get('original_file', [''] * len(batch_targets)))
            elif isinstance(targets, torch.Tensor):
                # Direct tensor format
                print(f"Target shape: {targets.shape}")
                all_targets.append(targets.numpy())
                all_files.extend([''] * len(targets))
            else:
                print(f"Unhandled target type: {type(targets)}")
                print(f"Target content: {targets}")
    
    # Print summary of collected data
    print("\nCollected Data Summary:")
    print(f"Number of prediction batches: {len(all_preds)}")
    print(f"Number of target batches: {len(all_targets)}")
    
    if not all_preds:
        raise ValueError("No predictions were made. Check the data loader output format.")
    
    try:
        # Concatenate all batches
        all_preds = np.vstack(all_preds) if len(all_preds[0].shape) > 1 else np.concatenate(all_preds)
        if all_targets:
            all_targets = np.vstack(all_targets) if len(all_targets[0].shape) > 1 else np.concatenate(all_targets)
        else:
            all_targets = np.zeros((len(all_preds), 2))  # Dummy targets if none found
            
        # Ensure we have the same number of predictions and targets
        if len(all_preds) != len(all_targets):
            print(f"Warning: Mismatch in number of predictions ({len(all_preds)}) and targets ({len(all_targets)})")
            min_len = min(len(all_preds), len(all_targets))
            all_preds = all_preds[:min_len]
            all_targets = all_targets[:min_len]
            
        # Ensure filenames match predictions length
        if len(all_files) != len(all_preds):
            all_files = all_files[:len(all_preds)]
            if len(all_files) < len(all_preds):
                all_files.extend([''] * (len(all_preds) - len(all_files)))
                
    except Exception as e:
        print(f"Error processing predictions/targets: {e}")
        print(f"Predictions shape: {[p.shape for p in all_preds]}")
        print(f"Targets shape: {[t.shape for t in all_targets]}")
        raise
    
    return all_preds, all_targets, all_files

def save_results(preds, targets, filenames, angle_ranges, output_path):
    """Save predictions and ground truth to a CSV file."""
    # Denormalize angles
    denorm_preds = denormalize_angles(preds, angle_ranges)
    denorm_targets = denormalize_angles(targets, angle_ranges)
    
    # Calculate errors
    errors = np.abs(denorm_preds - denorm_targets)
    
    # Create DataFrame
    results = pd.DataFrame({
        'filename': filenames,
        'pred_azimuth': denorm_preds[:, 0],
        'pred_elevation': denorm_preds[:, 1],
        'true_azimuth': denorm_targets[:, 0],
        'true_elevation': denorm_targets[:, 1],
        'azimuth_error': errors[:, 0],
        'elevation_error': errors[:, 1]
    })
    
    # Save to CSV
    results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    # Print summary statistics
    print("\nError Summary:")
    print(f"Mean Azimuth Error: {errors[:, 0].mean():.2f} degrees")
    print(f"Mean Elevation Error: {errors[:, 1].mean():.2f} degrees")
    print(f"Median Azimuth Error: {np.median(errors[:, 0]):.2f} degrees")
    print(f"Median Elevation Error: {np.median(errors[:, 1]):.2f} degrees")

def main():
    # Model parameters - these should match the training configuration
    model_params = {
        'n_mics': 16,  # Assuming 16 microphones based on the dataset
        'fs': 16000,   # Sample rate
        'n_samples_in_frame': 4096,  # Frame size
        'max_tau': 0.001,  # Maximum time delay in seconds
        'interp_factor': 4  # Interpolation factor for GCC-PHAT
    }
    
    # Angle ranges for denormalization
    angle_ranges = {
        'azimuth': (0, 360),    # Full circle for azimuth
        'elevation': (-40, 90)  # Typical range for elevation
    }
    
    # Load the test dataset
    print("Loading test dataset...")
    script = Path(__file__)
    project_root = script.parents[2]
    config_path = project_root / "05_config/c12_t10_training_split.yaml"
    print(f'Loading config from {config_path}')

    _, test_loader, _ = create_data_loaders(
        config_path=config_path,
        batch_size=32,
        num_workers=4,
        shuffle_test=False,
        pin_memory=torch.cuda.is_available()
    )
    
    # Load the best model
    print("Loading model...")
    model = load_best_model(CHECKPOINT_DIR, model_params, device)
    
    # Run evaluation
    print("Running evaluation...")
    preds, targets, filenames = evaluate_model(model, test_loader, device)
    
    # Save results
    output_path = RESULTS_DIR / "angle_predictions.csv"
    save_results(preds, targets, filenames, angle_ranges, output_path)

if __name__ == "__main__":
    main()