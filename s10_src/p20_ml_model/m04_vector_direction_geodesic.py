# RUN DATE: 2025-06-01 15:33:24
# File: s10_train_cnn_model_vectorized.py (suggested new name)

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from dotenv import load_dotenv
import wandb
import math  # For AngleVectorizer


# Assuming AngleVectorizer is in a separate file or defined above
# from s10_src.p20_ml_model.u03_angle_vectorizer import AngleVectorizer
# For self-containment, I'll include the AngleVectorizer class definition here.
# Replace this with an import if you have it in a separate module.

# ---- AngleVectorizer Class Definition ----
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
            # Ensure el_tensor is created on the same device as az_tensor if az_tensor already exists
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
        z = torch.clamp(z, -1.0, 1.0)  # Important for asin stability

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


from s10_src.p20_ml_model.m02_gcc_phat_features import GCCPHATFeatures
from s10_src.p20_ml_model.u01_Early_Stopping import PatienceEarlyStopping
# AngleNormalizer is no longer needed by this script directly
# from s10_src.p20_ml_model.u02_angle_normalizer import AngleNormalizer
from s10_src.p55_util.f02_script_comments import insert_run_date_comment
from s10_src.p55_util.f03_auto_git import auto_commit_and_get_hash

# Global instance for convenience, assuming angles from dataset are in degrees
# This should be initialized before use in collate_fn or train_model
angle_vectorizer_degrees = AngleVectorizer(angle_units='degrees')


def initialize_wandb(project_name="my_project"):
    wandb.init(project=project_name)
    git_commit_hash = auto_commit_and_get_hash(script_path=__file__)
    if git_commit_hash:
        wandb.config.update({"git_commit_hash": git_commit_hash})
    else:
        print("Git commit hash not available.")


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

        # Output Layer: 3 units (x, y, z of the unit vector)
        self.fc_out = nn.Linear(64, 3)  # MODIFIED: Output 3 values for the vector
        self.tanh_out = nn.Tanh()  # Output for each component between -1 and 1

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        - signals: [batch_size, n_mics, n_samples] time-domain signals

        Returns:
        - predictions: [batch_size, 3] tensor of (x,y,z) unit direction vectors.
        """
        gcc_features = self.gcc_phat_extractor(signals)
        x = gcc_features.unsqueeze(1)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.dropout1(self.relu_fc1(self.fc1(x)))
        x = self.dropout2(self.relu_fc2(self.fc2(x)))
        x = self.tanh_out(self.fc_out(x))  # Output components in (-1, 1)

        # MODIFIED: Normalize to ensure it's a unit vector
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        # Add a small epsilon to the norm before division for numerical stability
        x = x / (norm + 1e-8)
        return x


def train_model(model, train_loader, test_loader, early_stopping_strategy=None,
                epochs=200, lr=1e-3, device='cpu', checkpoint_dir='checkpoints',
                grad_accum_steps=1, wandb_config=None):  # Added wandb_config
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # MSELoss is suitable for comparing predicted 3D vectors to target 3D vectors.
    # Each component of the vector will be compared.
    criterion = nn.MSELoss()
    # Alternative: Cosine Embedding Loss or 1 - CosineSimilarity for unit vectors
    # criterion = nn.CosineEmbeddingLoss() # target would be torch.ones(batch_size)
    # or custom: lambda pred, target: (1 - nn.functional.cosine_similarity(pred, target, dim=-1)).mean()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    print(f"Initial learning rate: {optimizer.param_groups[0]['lr']}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    if wandb.run:  # Check if wandb is initialized
        wandb.watch(model, log="all")

    best_metric_val = float('inf')  # Can be loss or angular error

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()  # Initialize gradients once per epoch if not accumulating over epoch

        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)  # targets are now [B, 3]

            outputs = model(inputs)  # outputs are [B, 3]

            # For CosineEmbeddingLoss: loss = criterion(outputs, targets, torch.ones(outputs.size(0), device=device))
            loss = criterion(outputs, targets)

            if torch.isnan(loss).any():
                print(f"WARNING: NaN detected in loss at epoch {epoch + 1}, batch {i + 1}. Skipping batch.")
                print("Outputs sample:", outputs[0] if outputs.numel() > 0 else "Empty")
                print("Targets sample:", targets[0] if targets.numel() > 0 else "Empty")
                # Reset gradients for this problematic step if they were computed
                if (i + 1) % grad_accum_steps != 0:  # If optimizer.step() was not called
                    optimizer.zero_grad()  # Clear gradients from this bad batch
                continue

            loss_accum = loss / grad_accum_steps
            loss_accum.backward()

            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                # Optional: Gradient Clipping
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        test_loss = 0.0
        total_angular_error_deg = 0.0
        num_test_samples = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                # loss = criterion(outputs, targets, torch.ones(outputs.size(0), device=device)) # For CosineEmbeddingLoss
                loss = criterion(outputs, targets)
                test_loss += loss.item()

                # Calculate angular error
                # Ensure outputs and targets are unit vectors (model output is, target should be)
                # Clamp dot product to [-1, 1] for acos stability
                cosine_sim = nn.functional.cosine_similarity(outputs, targets, dim=-1)
                cosine_sim_clamped = torch.clamp(cosine_sim, -1.0, 1.0)
                angular_error_rad = torch.acos(cosine_sim_clamped)
                angular_error_deg_batch = torch.rad2deg(angular_error_rad)

                total_angular_error_deg += angular_error_deg_batch.sum().item()
                num_test_samples += inputs.size(0)

        avg_test_loss = test_loss / len(test_loader)
        avg_angular_error_deg = total_angular_error_deg / num_test_samples if num_test_samples > 0 else float('nan')

        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, "
              f"Avg Angular Error (deg): {avg_angular_error_deg:.2f}")

        if wandb.run:
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "test_loss": avg_test_loss,
                "avg_angular_error_deg": avg_angular_error_deg,
                "learning_rate": current_lr,
            })

        # Decide metric for scheduler and early stopping
        metric_for_scheduler = avg_angular_error_deg  # Or avg_test_loss
        scheduler.step(metric_for_scheduler)

        if metric_for_scheduler < best_metric_val:
            best_metric_val = metric_for_scheduler
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss,  # MSE vector loss
                'angular_error_deg': avg_angular_error_deg,  # Angular error
                'wandb_config': wandb_config  # Save config used for this training
            }, checkpoint_path)
            print(
                f"Saved new best model to {checkpoint_path} (Loss: {avg_test_loss:.4f}, Angular Err: {avg_angular_error_deg:.2f}°)")
            if wandb.run:
                wandb.save(checkpoint_path)  # Save best model to W&B

        if early_stopping_strategy and early_stopping_strategy.should_stop(metric_for_scheduler):
            print(f"Early stopping triggered at epoch {epoch + 1} with metric value: {metric_for_scheduler:.4f}")
            break
    print("Training complete.")


def collate_fn(batch):
    """Custom collate function to convert (az, el) targets to 3D vectors."""
    inputs = torch.stack([item[0] for item in batch])

    # Extract raw angles (assuming degrees as per dataset)
    az_list = [item[1]['azimuth'] for item in batch]
    el_list = [item[1]['elevation'] for item in batch]

    # Convert to tensors (AngleVectorizer can also handle lists, but this is explicit)
    az_tensor = torch.tensor(az_list, dtype=torch.float32)
    el_tensor = torch.tensor(el_list, dtype=torch.float32)

    # Use the global AngleVectorizer instance
    # angle_vectorizer_degrees is initialized with angle_units='degrees'
    targets_vector = angle_vectorizer_degrees.angles_to_vector(az_tensor, el_tensor)  # Output: [B, 3]

    return inputs, targets_vector


if __name__ == "__main__":
    load_dotenv()
    insert_run_date_comment(script_path=__file__)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Ensure angle_vectorizer_degrees is initialized (done globally)
    if 'angle_vectorizer_degrees' not in globals() or angle_vectorizer_degrees is None:
        angle_vectorizer_degrees = AngleVectorizer(angle_units='degrees')
        print("Re-initialized global angle_vectorizer_degrees in main.")

    # W&B Configuration (used for logging and can be saved with model)
    # Keep training hyperparams separate from data/model fixed params
    training_config = {
        'batch_size': 32,
        'gradient_accumulation_steps': 2,
        'lr': 1e-4,  # Start with a slightly lower LR for potentially more complex target
        'epochs': 50,  # Reduced for quicker test runs
    }
    model_params = {
        'n_mics': 16,
        'fs': 16000,
        'n_samples_in_frame': 4096,  # Matches 'n_samples' in original config
        'max_tau': 0.001,
        'interp_factor': 4
    }
    # Combine for W&B logging
    wandb_config = {**training_config, **model_params, "architecture": "CNN_VectorOutput"}

    initialize_wandb(project_name="SSL_Vectorized_Output")  # New W&B project name
    if wandb.run:
        wandb.config.update(wandb_config)

    # Load data
    from s10_src.p20_ml_model.dl01_time_series_16_channels import create_data_loaders

    script_path = Path(__file__)  # Use consistent naming
    project_root = script_path.parents[2]
    data_config_yaml_path = project_root / "05_config/c12_t10_training_split.yaml"
    print(f'Loading data config from {data_config_yaml_path}')

    train_loader, test_loader, data_info = create_data_loaders(
        config_path=data_config_yaml_path,
        batch_size=training_config['batch_size'],
        num_workers=os.cpu_count() // 2 if os.cpu_count() else 4,  # Dynamic num_workers
        pin_memory=True if device == 'cuda' else False,  # Pin memory only if using CUDA
        shuffle_train=True,
        shuffle_test=False,
        collate_fn=collate_fn  # IMPORTANT: Pass the new collate_fn
    )
    print(f"Data info: {data_info}")

    # Initialize model
    model = SoundSourceLocalizationCNN(
        n_mics=model_params['n_mics'],
        fs=model_params['fs'],
        n_samples_in_frame=model_params['n_samples_in_frame'],
        max_tau=model_params['max_tau'],
        interp_factor=model_params['interp_factor']
    ).to(device)

    print("\nModel architecture:")
    # print(model) # Can be very verbose
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Train the model
    checkpoint_dir_path = Path('/home/tj/99_tmp/11 - synthetic mic array data/02_training_data/checkpoints_vectorized/')
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        early_stopping_strategy=PatienceEarlyStopping(patience=10, verbose=True),  # Monitor angular error or test loss
        epochs=training_config['epochs'],
        lr=training_config['lr'],
        device=device,
        checkpoint_dir=str(checkpoint_dir_path),
        grad_accum_steps=training_config['gradient_accumulation_steps'],
        wandb_config=wandb_config  # Pass config for saving with checkpoint
    )

    if wandb.run:
        wandb.finish()