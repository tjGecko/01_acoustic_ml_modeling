import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import datetime
from functools import partial # Import partial

# Import the model and dataset classes
from m03_m01_gccphat_cnn import SoundSourceLocalizationCNN
from dl01_time_series_16_channels import TimeSeries16ChannelDataset, create_data_loaders
from s10_src.p20_ml_model.u02_angle_normalizer import AngleNormalizer

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
        checkpoints = list(Path(checkpoint_dir).glob("*.pth")) + list(Path(checkpoint_dir).glob("*.pt")) # Fixed glob pattern
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


def collate_fn(batch, normalizer):
    """Custom collate function to handle the dictionary structure of our dataset."""
    inputs = torch.stack([item[0] for item in batch])  # Stack audio tensors

    # Extract and normalize angles
    az = torch.tensor([item[1]['azimuth'] for item in batch], dtype=torch.float32)
    el = torch.tensor([item[1]['elevation'] for item in batch], dtype=torch.float32)

    # Normalize angles to [-1, 1] range
    az_norm, el_norm = normalizer.normalize(az, el)

    # Stack normalized angles
    targets = torch.stack([az_norm, el_norm], dim=1)

    return inputs, targets


def denormalize_angles(normalized_angles, normalizer):
    """Convert normalized angles back to original ranges using AngleNormalizer."""
    if isinstance(normalized_angles, np.ndarray):
        normalized_angles = torch.from_numpy(normalized_angles).float() # ensure float
    
    # Split into azimuth and elevation
    az_norm = normalized_angles[:, 0]
    el_norm = normalized_angles[:, 1] if normalized_angles.shape[1] > 1 else torch.zeros_like(az_norm)
    
    # Denormalize using the provided normalizer
    az_denorm, el_denorm = normalizer.denormalize(az_norm, el_norm)
    
    # Stack back together
    return torch.stack([az_denorm, el_denorm], dim=1).numpy()

def evaluate_model(model, test_loader, device='cpu'): # Removed normalizer argument
    """
    Run evaluation on the test set and return predictions and ground truth.
    
    Args:
        model: Trained model to evaluate
        test_loader: DataLoader for the test set (already configured with collate_fn)
        device: Device to run evaluation on
        
    Returns:
        tuple: (all_preds, all_targets) where:
            - all_preds: Normalized model predictions in [-1, 1] range
            - all_targets: Normalized ground truth angles in [-1, 1] range
    """
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Evaluating"):
            # Your custom_collate_fn returns (inputs, targets)
            inputs, targets = batch_data  # Correctly unpack the 2-tuple

            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Store normalized predictions
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy()) # targets are already on CPU from collate_fn
            # If filenames were handled: all_filenames.extend(filenames_batch)

    # Concatenate all batches
    all_preds = np.vstack(all_preds) if all_preds else np.array([])
    all_targets = np.vstack(all_targets) if all_targets else np.array([])
    
    # Ensure we have the same number of predictions and targets
    min_len = min(len(all_preds), len(all_targets))
    if min_len == 0:
        raise ValueError("No predictions or targets were collected. Check the data loader output format.")

    # All normalized angle data
    all_preds = all_preds[:min_len]
    all_targets = all_targets[:min_len]

    return all_preds, all_targets

def save_results(preds_norm, targets_norm, normalizer, output_path):
    """
    Save predictions and ground truth to a CSV file.
    
    Args:
        preds_norm: Normalized model predictions in [-1, 1] range
        targets_norm: Normalized ground truth angles in [-1, 1] range
        normalizer: AngleNormalizer instance used for denormalization
        output_path: Path to save the results CSV
    """
    denorm_preds_deg = denormalize_angles(preds_norm, normalizer)

    # targets_norm are normalized. The column names like 'true_azimuth_dec'
    # in the original code imply these might be the normalized decimal values.
    # If actual degrees are desired for true angles, they also need denormalization:
    # denorm_targets_deg = denormalize_angles(targets_norm, normalizer)
    # For now, stick to original structure where 'true_..._dec' uses normalized values.

    # Error calculation based on normalized values, as in original code
    errors_normalized = np.abs(preds_norm - targets_norm)

    results = pd.DataFrame({
        'true_azimuth_norm_dec': targets_norm[:, 0], # Normalized true angles
        'true_elevation_norm_dec': targets_norm[:, 1], # Normalized true angles
        'azimuth_error_norm_dec': errors_normalized[:, 0], # Error in normalized space
        'elevation_error_norm_dec': errors_normalized[:, 1], # Error in normalized space
        'total_error_norm_dec': np.sqrt(errors_normalized[:, 0]**2 + errors_normalized[:, 1]**2),
        'pred_azimuth_deg': denorm_preds_deg[:, 0],
        'pred_elevation_deg': denorm_preds_deg[:, 1],
        'pred_az_norm_dec': preds_norm[:, 0],
        'pred_el_norm_dec': preds_norm[:, 1] if preds_norm.shape[1] > 1 else np.zeros_like(preds_norm[:, 0])
    })
    

    # Save results to CSV
    results.to_csv(output_path, index=False, float_format='%.6f')
    print(f"Results saved to {output_path}")

def main():
    # Model parameters (should match training configuration)
    model_params = {
        'n_mics': 16,           # Number of microphones in the array
        'fs': 16000,            # Sample rate (Hz)
        'n_samples_in_frame': 4096,  # Frame size
        'max_tau': 0.001,       # Maximum time delay in seconds
        'interp_factor': 4      # Interpolation factor for GCC-PHAT
    }
    # Initialize angle normalizer with the same ranges used during training
    normalizer = AngleNormalizer(az_range=(0, 360), el_range=(-90, 90))
    
    # Create results directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"eval_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Starting evaluation run: {timestamp}")
    print(f"[INFO] Saving results to: {run_dir}")

    # Load the best model
    print("[INFO] Loading model...")
    model = load_best_model(CHECKPOINT_DIR, model_params, device)
    print("[INFO] Model loaded successfully")

    # Print model architecture
    print(f"[INFO] Model architecture:\n{model}")

    # Load test dataset
    print("[INFO] Loading test dataset...")
    # Create a partial function for collate_fn with normalizer baked in
    custom_collate_fn = partial(collate_fn, normalizer=normalizer)

    # Correctly unpack: we only need the test_loader from the returned tuple
    # The other returned values (train_loader, data_info) are ignored with _
    _, test_loader_from_func, _ = create_data_loaders(
        config_path=Path("/home/tj/02_Windsurf_Projects/r03_Gimbal_Angle_Root/05_config/c12_t10_training_split.yaml"),
        batch_size=32,
        num_workers=4,
        collate_fn=custom_collate_fn
    )

    # Run evaluation
    print("[INFO] Starting evaluation...")
    # evaluate_model no longer takes normalizer, no longer returns filenames (unless adapted)
    preds_normalized, targets_normalized = evaluate_model(model, test_loader_from_func, device) # Pass the actual DataLoader

    # Save results
    results_file = run_dir / "evaluation_results.csv"
    # Call save_results with correct arguments
    save_results(preds_normalized, targets_normalized, normalizer, results_file)

    print("[INFO] Evaluation completed successfully!")
        

if __name__ == "__main__":
    main()