# RUN DATE: 2025-06-01 16:47:33
# File: s11_evaluate_cnn_model_vectorized.py

# RUN DATE: will be inserted by your utility
import os
import torch
import torch.nn as nn
import torch.nn.functional as F  # For cosine_similarity
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import datetime
import math  # For AngleVectorizer
from functools import partial

# Assuming these are in your project structure and Python path
from s10_src.p20_ml_model.m02_gcc_phat_features import GCCPHATFeatures
from s10_src.p20_ml_model.dl01_time_series_16_channels import \
    create_data_loaders  # TimeSeries16ChannelDataset is also used by it
from s10_src.p55_util.f02_script_comments import insert_run_date_comment  # For consistency if you run this utility


# ---- AngleVectorizer Class Definition (Copied from training script) ----
class AngleVectorizer:
    """
    Encodes/decodes angular directions (azimuth, elevation) as 3D unit vectors.
    """

    def __init__(self, angle_units='degrees'):
        if angle_units not in ['degrees', 'radians']:
            raise ValueError("angle_units must be 'degrees' or 'radians'")
        self.angle_units = angle_units
        self.pi = getattr(torch, 'pi', math.pi)

    def _process_input_angles(self, az, el):
        if not isinstance(az, torch.Tensor):
            az_tensor = torch.tensor(az, dtype=torch.float32)
        else:
            az_tensor = az.float()

        if not isinstance(el, torch.Tensor):
            device = az_tensor.device if isinstance(az, torch.Tensor) else None
            el_tensor = torch.tensor(el, dtype=torch.float32, device=device)
        else:
            el_tensor = el.float()

        if az_tensor.device != el_tensor.device:
            el_tensor = el_tensor.to(az_tensor.device)

        try:
            az_tensor, el_tensor = torch.broadcast_tensors(az_tensor, el_tensor)
        except RuntimeError as e:
            raise ValueError(
                f"Azimuth and elevation shapes are not broadcastable: az_shape={az_tensor.shape}, el_shape={el_tensor.shape}. Error: {e}")

        if self.angle_units == 'degrees':
            az_rad = torch.deg2rad(az_tensor)
            el_rad = torch.deg2rad(el_tensor)
        else:
            az_rad = az_tensor
            el_rad = el_tensor
        return az_rad, el_rad

    def angles_to_vector(self, az, el):
        az_rad, el_rad = self._process_input_angles(az, el)
        x = torch.cos(el_rad) * torch.cos(az_rad)
        y = torch.cos(el_rad) * torch.sin(az_rad)
        z = torch.sin(el_rad)
        target_vec = torch.stack([x, y, z], dim=-1)
        return target_vec

    def vector_to_angles(self, vec):
        if not isinstance(vec, torch.Tensor):
            try:
                vec_tensor = torch.tensor(vec, dtype=torch.float32)
            except Exception as e:
                raise TypeError(f"Input 'vec' could not be converted to a tensor. Original error: {e}")
        else:
            vec_tensor = vec.float()

        if vec_tensor.shape[-1] != 3:
            raise ValueError(f"Input vector must have its last dimension of size 3, got {vec_tensor.shape}")

        x = vec_tensor[..., 0]
        y = vec_tensor[..., 1]
        z = vec_tensor[..., 2]
        z = torch.clamp(z, -1.0, 1.0)

        el_rad = torch.asin(z)
        az_rad = torch.atan2(y, x)
        az_rad_normalized = torch.remainder(az_rad, 2 * self.pi)

        if self.angle_units == 'degrees':
            az_out = torch.rad2deg(az_rad_normalized)
            el_out = torch.rad2deg(el_rad)
        else:
            az_out = az_rad_normalized
            el_out = el_rad
        return az_out, el_out


# ---- End of AngleVectorizer Class Definition ----


# ---- SoundSourceLocalizationCNN Class Definition (Copied from training script) ----
class SoundSourceLocalizationCNN(nn.Module):
    def __init__(self, n_mics: int, fs: int = 16000, n_samples_in_frame: int = 4096,
                 max_tau: float = 0.001, interp_factor: int = 4):
        super().__init__()
        self.gcc_phat_extractor = GCCPHATFeatures(
            n_mics=n_mics,
            fs=fs,
            n_samples_in_frame=n_samples_in_frame,
            max_tau=max_tau,
            interp_factor=interp_factor
        )
        self.n_pairs = self.gcc_phat_extractor.n_pairs
        self.gcc_feature_length = self.gcc_phat_extractor.gcc_feature_length

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        with torch.no_grad():
            dummy_input_height = self.n_pairs
            dummy_input_width = self.gcc_feature_length
            dummy_conv_input = torch.zeros(1, 1, dummy_input_height, dummy_input_width)
            dummy_after_conv1 = self.pool1(self.relu1(self.conv1(dummy_conv_input)))
            dummy_after_conv2 = self.pool2(self.relu2(self.conv2(dummy_after_conv1)))
            self.flattened_size = dummy_after_conv2.numel()

        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 64)
        self.relu_fc2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc_out = nn.Linear(64, 3)
        self.tanh_out = nn.Tanh()

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        gcc_features = self.gcc_phat_extractor(signals)
        x = gcc_features.unsqueeze(1)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.dropout1(self.relu_fc1(self.fc1(x)))
        x = self.dropout2(self.relu_fc2(self.fc2(x)))
        x = self.tanh_out(self.fc_out(x))
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        x = x / (norm + 1e-8)
        return x


# ---- End of SoundSourceLocalizationCNN Class Definition ----


# Global AngleVectorizer instance
angle_vectorizer_degrees = AngleVectorizer(angle_units='degrees')

# Configuration
# IMPORTANT: Update this path to where your *vectorized* model checkpoint is saved
CHECKPOINT_DIR = Path("/home/tj/99_tmp/11 - synthetic mic array data/02_training_data/checkpoints_vectorized/")
RESULTS_DIR = Path("/home/tj/99_tmp/11 - synthetic mic array data/02_training_data/prediction_results_vectorized/")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_best_vectorized_model(checkpoint_dir, model_params, device='cpu'):
    """Load the best vectorized model from checkpoints."""
    best_checkpoint_path = Path(checkpoint_dir) / "best_model.pth"

    if not best_checkpoint_path.exists():
        # Fallback: try to find any .pth file if best_model.pth is not present
        checkpoints = list(Path(checkpoint_dir).glob("*.pth"))
        if not checkpoints:
            raise FileNotFoundError(f"No model checkpoints (.pth files) found in {checkpoint_dir}")
        best_checkpoint_path = checkpoints[0]
        print(f"Warning: 'best_model.pth' not found. Using the first available checkpoint: {best_checkpoint_path}")

    print(f"Loading vectorized model from {best_checkpoint_path}")

    model = SoundSourceLocalizationCNN(**model_params)

    try:
        checkpoint = torch.load(best_checkpoint_path, map_location=device)

        # Check if the loaded checkpoint is a dictionary and contains 'model_state_dict'
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            print(
                f"Loaded model from epoch {checkpoint.get('epoch', 'N/A')}, Test Loss (MSE Vector): {checkpoint.get('test_loss', 'N/A'):.4f}, Angular Error: {checkpoint.get('angular_error_deg', 'N/A'):.2f}°")
            if 'wandb_config' in checkpoint and checkpoint['wandb_config'] is not None:
                # Validate model_params against saved config if desired
                saved_params = {k: checkpoint['wandb_config'][k] for k in model_params if
                                k in checkpoint['wandb_config']}
                if saved_params != model_params:
                    print(
                        f"Warning: Provided model_params differ from saved wandb_config. Using provided model_params.")
                    print(f"Provided: {model_params}")
                    print(f"Saved: {saved_params}")

        elif isinstance(checkpoint, dict):  # Checkpoint is a state_dict itself
            model_state_dict = checkpoint
        else:  # Checkpoint is directly the state_dict (less common for complex saves)
            model_state_dict = checkpoint

        model.load_state_dict(model_state_dict)

    except Exception as e:  # Catch a broader range of exceptions during loading
        print(f"Error loading model state_dict: {e}")
        print("Attempting to load with strict=False as a fallback...")
        try:
            model.load_state_dict(model_state_dict, strict=False)  # Ensure model_state_dict is defined
        except Exception as e_strict_false:
            raise RuntimeError(
                f"Failed to load model even with strict=False. Original error: {e}, Fallback error: {e_strict_false}")

    model = model.to(device)
    model.eval()
    return model


def collate_fn_vectorized(batch, vectorizer):
    """Custom collate function to convert (az, el) targets to 3D vectors using the provided vectorizer."""
    inputs = torch.stack([item[0] for item in batch])

    az_list = [item[1]['azimuth'] for item in batch]
    el_list = [item[1]['elevation'] for item in batch]

    az_tensor = torch.tensor(az_list, dtype=torch.float32)
    el_tensor = torch.tensor(el_list, dtype=torch.float32)

    # Use the passed AngleVectorizer instance
    target_vectors = vectorizer.angles_to_vector(az_tensor, el_tensor)  # Output: [B, 3]

    return inputs, target_vectors


def evaluate_vectorized_model(model, test_loader, device):
    """
    Run evaluation on the test set and return predicted and true 3D vectors.
    """
    all_predicted_vectors = []
    all_true_target_vectors = []

    model.eval()  # Ensure model is in evaluation mode
    with torch.no_grad():
        for inputs, true_vectors_batch in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            # true_vectors_batch is already a tensor from collate_fn

            predicted_vectors_batch = model(inputs)  # Model outputs [B, 3] unit vectors

            all_predicted_vectors.append(predicted_vectors_batch.cpu().numpy())
            all_true_target_vectors.append(
                true_vectors_batch.cpu().numpy())  # Assuming collate_fn keeps them on CPU or moves to CPU

    if not all_predicted_vectors:
        raise ValueError("No predictions were collected. Check data loader or model.")

    all_predicted_vectors = np.vstack(all_predicted_vectors)
    all_true_target_vectors = np.vstack(all_true_target_vectors)

    return all_predicted_vectors, all_true_target_vectors


def process_and_save_results(
        predicted_vectors_np,
        true_vectors_np,
        vectorizer,
        output_csv_path
):
    """
    Processes predicted and true vectors to angles, calculates errors, and saves to CSV.
    """
    # Convert numpy arrays to tensors for vectorizer
    predicted_vectors_torch = torch.from_numpy(predicted_vectors_np).float()
    true_vectors_torch = torch.from_numpy(true_vectors_np).float()

    # Convert vectors back to angles (degrees)
    pred_az_deg, pred_el_deg = vectorizer.vector_to_angles(predicted_vectors_torch)
    true_az_deg, true_el_deg = vectorizer.vector_to_angles(true_vectors_torch)

    # Calculate overall angular error (more robust than individual az/el errors)
    # Cosine similarity: dot(v1, v2) / (||v1|| * ||v2||)
    # Since they are unit vectors, this is just dot(v1, v2)
    # Ensure vectors are on the same device if using GPU for F.cosine_similarity
    cosine_sim = F.cosine_similarity(predicted_vectors_torch, true_vectors_torch, dim=-1)
    # Clamp for numerical stability with acos
    cosine_sim_clamped = torch.clamp(cosine_sim, -1.0, 1.0)
    angular_error_rad = torch.acos(cosine_sim_clamped)
    angular_error_deg = torch.rad2deg(angular_error_rad)

    # Individual Az/El errors (degrees) - can be misleading due to azimuth wrapping if not careful
    # For direct comparison, these are fine if angles are consistently in [0,360) and [-90,90]
    az_error_deg = torch.abs(pred_az_deg - true_az_deg)
    # Handle azimuth wrapping for error calculation (e.g., 355 vs 5 degrees is 10 deg error, not 350)
    az_error_deg = torch.min(az_error_deg, 360.0 - az_error_deg)
    el_error_deg = torch.abs(pred_el_deg - true_el_deg)

    results_df = pd.DataFrame({
        'true_azimuth_deg': true_az_deg.numpy(),
        'true_elevation_deg': true_el_deg.numpy(),
        'pred_azimuth_deg': pred_az_deg.numpy(),
        'pred_elevation_deg': pred_el_deg.numpy(),
        'azimuth_error_deg': az_error_deg.numpy(),
        'elevation_error_deg': el_error_deg.numpy(),
        'overall_angular_error_deg': angular_error_deg.numpy(),
        'true_vector_x': true_vectors_np[:, 0],
        'true_vector_y': true_vectors_np[:, 1],
        'true_vector_z': true_vectors_np[:, 2],
        'pred_vector_x': predicted_vectors_np[:, 0],
        'pred_vector_y': predicted_vectors_np[:, 1],
        'pred_vector_z': predicted_vectors_np[:, 2],
    })

    results_df.to_csv(output_csv_path, index=False, float_format='%.6f')
    print(f"Results saved to {output_csv_path}")

    # Print some summary statistics
    mean_angular_error = results_df['overall_angular_error_deg'].mean()
    median_angular_error = results_df['overall_angular_error_deg'].median()
    std_angular_error = results_df['overall_angular_error_deg'].std()
    print(f"\nSummary Statistics for Overall Angular Error (degrees):")
    print(f"  Mean:   {mean_angular_error:.2f}°")
    print(f"  Median: {median_angular_error:.2f}°")
    print(f"  StdDev: {std_angular_error:.2f}°")


def main():
    insert_run_date_comment(script_path=__file__)  # Optional: for consistency

    print(f"[INFO] Using device: {device}")

    # Model parameters (MUST match the parameters used for training the loaded checkpoint)
    model_params = {
        'n_mics': 16,
        'fs': 16000,
        'n_samples_in_frame': 4096,
        'max_tau': 0.001,
        'interp_factor': 4
    }

    # Create results directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    current_run_results_dir = RESULTS_DIR / f"eval_vectorized_{timestamp}"
    current_run_results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Evaluation run ID: eval_vectorized_{timestamp}")
    print(f"[INFO] Saving results to: {current_run_results_dir}")

    print("[INFO] Loading vectorized model...")
    model = load_best_vectorized_model(CHECKPOINT_DIR, model_params, device)
    print("[INFO] Vectorized model loaded successfully.")
    # print(f"[INFO] Model architecture:\n{model}") # Can be verbose

    print("[INFO] Loading test dataset...")
    # The global 'angle_vectorizer_degrees' will be used by this partial function
    custom_collate_fn_for_loader = partial(collate_fn_vectorized, vectorizer=angle_vectorizer_degrees)

    # Path to your data configuration YAML (same as used in training)
    # Ensure this path is correct for your project structure
    script_file_path = Path(__file__)
    project_root_path = script_file_path.parents[2]  # Adjust if s11_evaluate... is in a different location
    data_config_yaml = project_root_path / "05_config/c12_t10_training_split.yaml"

    # We only need the test_loader from create_data_loaders
    _, test_loader, test_data_info = create_data_loaders(
        config_path=data_config_yaml,  # Make sure this path is correct
        batch_size=32,  # Can be larger for evaluation if memory allows
        num_workers=os.cpu_count() // 2 if os.cpu_count() else 4,
        pin_memory=True if device.type == 'cuda' else False,
        shuffle_train=False,  # Not relevant here, but good to be explicit
        shuffle_test=False,  # Test set should not be shuffled for consistent eval
        collate_fn=custom_collate_fn_for_loader  # Pass the vectorized collate_fn
    )
    print(f"[INFO] Test dataset loaded. Number of test samples: {test_data_info.get('test_samples', 'N/A')}")

    print("[INFO] Starting evaluation on test set...")
    predicted_vectors, true_target_vectors = evaluate_vectorized_model(model, test_loader, device)

    print("[INFO] Processing results and saving to CSV...")
    results_csv_file = current_run_results_dir / "evaluation_results_vectorized.csv"
    process_and_save_results(
        predicted_vectors,
        true_target_vectors,
        angle_vectorizer_degrees,  # Pass the global vectorizer
        results_csv_file
    )

    print(f"[INFO] Evaluation completed successfully. Results are in {results_csv_file}")


if __name__ == "__main__":
    main()