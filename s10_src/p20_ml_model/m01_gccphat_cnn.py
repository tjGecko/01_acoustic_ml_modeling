# RUN DATE: 2025-05-31 13:18:23
# RUN DATE: 2025-05-31 13:17:14
# RUN DATE: 2025-05-31 13:14:30
# RUN DATE: 2025-05-31 13:13:11
# RUN DATE: 2025-05-31 13:05:34
# RUN DATE: 2025-05-31 12:29:21
# RUN DATE: 2025-05-31 12:28:07
# RUN DATE: 2025-05-31 12:26:51
# RUN DATE: 2025-05-31 12:26:07
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from pathlib import Path
from dotenv import load_dotenv
import wandb

from s10_src.p20_ml_model.m02_gcc_phat_features import GCCPHATFeatures
from s10_src.p55_util.f02_script_comments import insert_run_date_comment
from s10_src.p55_util.f03_auto_git import auto_commit_and_get_hash


# from p10_ml_model.p05_training_strategies.s01_Patience import PatienceEarlyStopping
# from p55_util.f02_script_comments import insert_run_date_comment
# from p55_util.f03_auto_git import auto_commit_and_get_hash

# todo add data loader
# todo add min/max normalization for input data

def initialize_wandb(project_name="my_project"):
    # Initialize the WandB run
    wandb.init(project=project_name)

    # Auto-commit the current code and get the commit hash
    git_commit_hash = auto_commit_and_get_hash(script_path=__file__)

    # Log the commit hash in WandB
    if git_commit_hash:
        wandb.config.update({"git_commit_hash": git_commit_hash})
    else:
        print("Git commit hash not available.")





class SoundSourceLocalizationCNN(nn.Module):
    def __init__(self, n_mics: int, fs: int = 16000, n_samples_in_frame: int = 4096,
                 max_tau: float = 0.001, interp_factor: int = 4):
        super().__init__()

        # GCC-PHAT Feature Extractor
        self.gcc_phat_extractor = GCCPHATFeatures(
            n_mics=n_mics,
            fs=fs,
            n_samples_in_frame=n_samples_in_frame,
            max_tau=max_tau,
            interp_factor=interp_factor
        )

        # From GCCPHATFeatures, we know:
        # n_pairs = n_mics * (n_mics - 1) // 2
        # gcc_feature_length (N_c) = 2 * ceil(max_tau * fs * interp_factor) + 1
        self.n_pairs = self.gcc_phat_extractor.n_pairs
        self.gcc_feature_length = self.gcc_phat_extractor.gcc_feature_length

        # CNN Architecture (based on Figure 2 of the paper)
        # Input to Conv2d: [batch_size, in_channels, height, width]
        # Here, in_channels=1, height=n_pairs, width=gcc_feature_length

        # Conv1: 64 filters, 3x3 kernel, stride 1x1. Paper doesn't state padding.
        # To maintain dimensions with 3x3 kernel and stride 1, padding should be 1.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output size after pool1: H_out = floor((H_in - 2)/2 + 1), W_out = floor((W_in - 2)/2 + 1)
        # H_out1 = self.n_pairs // 2
        # W_out1 = self.gcc_feature_length // 2

        # Conv2: 128 filters, 3x3 kernel, stride 1x1, padding 1.
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # H_out2 = H_out1 // 2
        # W_out2 = W_out1 // 2

        # Calculate flattened size dynamically
        # Create a dummy input to pass through conv layers to get the flattened size
        # This is more robust than manual calculation if padding/strides change.
        with torch.no_grad():
            dummy_input_height = self.n_pairs
            dummy_input_width = self.gcc_feature_length
            dummy_conv_input = torch.zeros(1, 1, dummy_input_height, dummy_input_width)
            dummy_after_conv1 = self.pool1(self.relu1(self.conv1(dummy_conv_input)))
            dummy_after_conv2 = self.pool2(self.relu2(self.conv2(dummy_after_conv1)))
            self.flattened_size = dummy_after_conv2.numel()  # numel gives total number of elements

        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128, 128)
        self.relu_fc2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        # Output Layer: 2 units (azimuth, elevation) with Tanh activation
        self.fc_out = nn.Linear(128, 2)
        self.tanh_out = nn.Tanh()  # Output between -1 and 1

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        - signals: [batch_size, n_mics, n_samples] time-domain signals

        Returns:
        - predictions: [batch_size, 2] tensor of (azimuth, elevation) estimates, scaled by tanh.
                       You'll need to rescale these to your actual angle ranges.
        """
        # 1. Extract GCC-PHAT features
        # gcc_features: [batch_size, n_pairs, gcc_feature_length]
        gcc_features = self.gcc_phat_extractor(signals)

        # 2. Reshape for Conv2D: [batch_size, 1, n_pairs, gcc_feature_length]
        x = gcc_features.unsqueeze(1)

        # 3. Pass through CNN layers
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        # 4. Flatten
        x = torch.flatten(x, start_dim=1)  # Flatten all dims except batch

        # 5. Pass through FC layers
        x = self.dropout1(self.relu_fc1(self.fc1(x)))
        x = self.dropout2(self.relu_fc2(self.fc2(x)))
        x = self.tanh_out(self.fc_out(x))

        return x


# Training function
def train_model(model, train_loader, test_loader, early_stopping_strategy=None, epochs=200, lr=1e-3, device='cpu'):
    # model.apply(init_he_weights)
    model.to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5  # L2 regularization
    )
    criterion = nn.MSELoss()

    # Initialize the scheduler after setting up the optimizer
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    print(f"Initial learning rate: {optimizer.param_groups[0]['lr']}")

    # Log training metrics to Weights and Biases
    wandb.watch(model, log="all")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets) / config['gradient_accumulation_steps']
            
            # Backward pass with gradient accumulation
            loss.backward()
            
            # Update weights every gradient_accumulation_steps
            if (i + 1) % config['gradient_accumulation_steps'] == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item() * config['gradient_accumulation_steps']
            
            # Print memory usage
            if i % 10 == 0:
                print(f"Batch {i}/{len(train_loader)} - Loss: {loss.item() * config['gradient_accumulation_steps']:.4f}")
                if torch.cuda.is_available():
                    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
                    print(f"GPU Memory Cached: {torch.cuda.memory_reserved()/1e9:.2f}GB")
        
        avg_train_loss = running_loss / len(train_loader)

        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)

        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, "
              )

        # Log metrics to Weights and Biases
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "test_loss": avg_test_loss,
            "learning_rate": current_lr,
        })

        # Step the scheduler based on the test loss
        scheduler.step(avg_test_loss)

        # Early stopping check
        if early_stopping_strategy and early_stopping_strategy.should_stop(avg_test_loss):
            break

    print("Training complete.")


class AngleNormalizer:
    """Normalize angles to [-1, 1] range for training."""
    def __init__(self, az_range=(-90, 90), el_range=(-90, 90)):
        self.az_range = az_range
        self.el_range = el_range

    def normalize(self, az, el):
        """Normalize angles to [-1, 1] range."""
        az_norm = 2 * (az - self.az_range[0]) / (self.az_range[1] - self.az_range[0]) - 1
        el_norm = 2 * (el - self.el_range[0]) / (self.el_range[1] - self.el_range[0]) - 1
        return az_norm, el_norm

    def denormalize(self, az_norm, el_norm):
        """Convert normalized angles back to original range."""
        az = (az_norm + 1) * (self.az_range[1] - self.az_range[0]) / 2 + self.az_range[0]
        el = (el_norm + 1) * (self.el_range[1] - self.el_range[0]) / 2 + self.el_range[0]
        return az, el


def collate_fn(batch):
    """Custom collate function to handle the dictionary structure of our dataset."""
    inputs = torch.stack([item[0] for item in batch])  # Stack audio tensors
    
    # Extract and normalize angles
    az = torch.tensor([item[1]['azimuth'] for item in batch], dtype=torch.float32)
    el = torch.tensor([item[1]['elevation'] for item in batch], dtype=torch.float32)
    
    # Normalize angles to [-1, 1] range
    normalizer = AngleNormalizer(az_range=(0, 360), el_range=(-90, 90))
    az_norm, el_norm = normalizer.normalize(az, el)
    
    # Stack normalized angles
    targets = torch.stack([az_norm, el_norm], dim=1)
    
    return inputs, targets


if __name__ == "__main__":
    load_dotenv()

    # Insert a unique RUN DATE comment into the script
    insert_run_date_comment(script_path=__file__)

    # Set up device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize Weights and Biases
    initialize_wandb(project_name="sound_source_localization")

    # Configuration - Modified for quick test
    config = {
        'batch_size': 8,  # Reduced batch size to fit in GPU memory
        'gradient_accumulation_steps': 4,  # Simulate larger batch size
        'lr': 1e-4,
        'epochs': 2,  # Only run 2 epochs for testing
        'n_mics': 16,
        'fs': 16000,
        'n_samples': 4096,
        'max_tau': 0.001,  # 1ms delay
        'interp_factor': 4
    }

    # Load data
    from s10_src.p20_ml_model.dl01_time_series_16_channels import create_data_loaders

    script = Path(__file__)
    project_root = script.parents[3]
    config_path = project_root / "05_config/c12_t10_training_split.yaml"
    print(f'Loading config from {config_path}')

    # Get data loaders
    train_loader, test_loader, _ = create_data_loaders(
        config_path=config_path,
        batch_size=config['batch_size'],
        num_workers=4,
        pin_memory=True,
        shuffle_train=True,
        shuffle_test=False
    )

    # Update data loaders to use our custom collate function
    train_loader.collate_fn = collate_fn
    test_loader.collate_fn = collate_fn

    # Initialize model
    model = SoundSourceLocalizationCNN(
        n_mics=config['n_mics'],
        fs=config['fs'],
        n_samples_in_frame=config['n_samples'],
        max_tau=config['max_tau'],
        interp_factor=config['interp_factor']
    ).to(device)

    # Log model architecture and config to wandb
    wandb.config.update(config)
    
    # Print model summary
    print("\nModel architecture:")
    print(model)
    print(f"\nNumber of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Initialize early stopping
    class PatienceEarlyStopping:
        def __init__(self, patience=10, min_delta=0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_loss = float('inf')
            self.early_stop = False

        def should_stop(self, val_loss):
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            return self.early_stop

    # Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        early_stopping_strategy=PatienceEarlyStopping(patience=10),
        epochs=config['epochs'],
        lr=config['lr'],
        device=device
    )

# if __name__ == '__main__':
#     # --- Parameters ---
#     BATCH_SIZE = 32  # As requested
#     N_MICS = 16  # As requested
#     FS = 16000  # As requested
#     N_SAMPLES = 4096  # As requested (frame length)
#
#     # Parameters from the paper / for GCC
#     MAX_TAU_GCC = 0.001  # s (1ms, as suggested in paper for their array size)
#     # Original code had 0.002s, which is also fine.
#     # This affects N_c (gcc_feature_length)
#     INTERP_FACTOR_GCC = 4  # L=4, as in paper
#
#     # --- Device ---
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     # --- Test GCCPHATFeatures module ---
#     print("\n--- Testing GCCPHATFeatures ---")
#     gcc_module = GCCPHATFeatures(
#         n_mics=N_MICS,
#         fs=FS,
#         n_samples_in_frame=N_SAMPLES,
#         max_tau=MAX_TAU_GCC,
#         interp_factor=INTERP_FACTOR_GCC
#     ).to(device)
#
#     print(f"Number of microphone pairs (N_p): {gcc_module.n_pairs}")
#     print(f"GCC feature length (N_c): {gcc_module.gcc_feature_length}")
#
#     # Create dummy batch of signals
#     dummy_signals = torch.randn(BATCH_SIZE, N_MICS, N_SAMPLES).to(device)
#     print(f"Input signals shape: {dummy_signals.shape}")
#
#     gcc_output = gcc_module(dummy_signals)
#     print(f"Output GCC features shape: {gcc_output.shape}")
#     # Expected: [BATCH_SIZE, n_pairs, gcc_feature_length]
#     # e.g. N_MICS=16 -> n_pairs = 16*15/2 = 120
#     # N_c for max_tau=0.001, fs=16k, interp=4:
#     # max_shift_samples = ceil(0.001 * 16000 * 4) = ceil(64) = 64
#     # gcc_feature_length = 2 * 64 + 1 = 129
#     # Expected shape: [32, 120, 129]
#     assert gcc_output.shape == (BATCH_SIZE, gcc_module.n_pairs, gcc_module.gcc_feature_length)
#     print("GCCPHATFeatures test passed.")
#
#     # --- Test SoundSourceLocalizationCNN ---
#     print("\n--- Testing SoundSourceLocalizationCNN ---")
#     ssl_cnn_model = SoundSourceLocalizationCNN(
#         n_mics=N_MICS,
#         fs=FS,
#         n_samples_in_frame=N_SAMPLES,
#         max_tau=MAX_TAU_GCC,
#         interp_factor=INTERP_FACTOR_GCC
#     ).to(device)
#
#     print(f"CNN Model:\n{ssl_cnn_model}")
#
#     # Test with dummy signals
#     predictions = ssl_cnn_model(dummy_signals)
#     print(f"Output predictions shape: {predictions.shape}")
#     # Expected: [BATCH_SIZE, 2]
#     assert predictions.shape == (BATCH_SIZE, 2)
#     print(f"Sample predictions (first 2 from batch):\n{predictions[:2]}")
#     print("SoundSourceLocalizationCNN test passed.")
#
#     # --- Example Training Loop Snippet (conceptual) ---
#     # optimizer = torch.optim.Adam(ssl_cnn_model.parameters(), lr=1e-4)
#     # criterion = nn.MSELoss() # Or a custom angular loss
#
#     # # Dummy targets (azimuth, elevation) scaled to [-1, 1] like Tanh output
#     # # In a real scenario, your ground truth angles (e.g., in degrees/radians)
#     # # would need to be normalized to the [-1, 1] range if using Tanh.
#     # # E.g., if azimuth is [-180, 180] -> target_az_norm = target_az_deg / 180.0
#     # # E.g., if elevation is [-90, 90] -> target_el_norm = target_el_deg / 90.0
#     # dummy_targets = torch.rand(BATCH_SIZE, 2).to(device) * 2 - 1 # Random values in [-1, 1]
#
#     # ssl_cnn_model.train()
#     # optimizer.zero_grad()
#     # output_preds = ssl_cnn_model(dummy_signals)
#     # loss = criterion(output_preds, dummy_targets)
#     # loss.backward()
#     # optimizer.step()
#     # print(f"\nConceptual training step: Loss = {loss.item()}")
