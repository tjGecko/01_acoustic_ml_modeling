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


# from p10_ml_model.p05_training_strategies.s01_Patience import PatienceEarlyStopping
# from p55_util.f02_script_comments import insert_run_date_comment
# from p55_util.f03_auto_git import auto_commit_and_get_hash

# todo add data loader
# todo look at normalization for input data

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
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
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
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 256)
        self.relu_fc2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        # Output Layer: 2 units (azimuth, elevation) with Tanh activation
        self.fc_out = nn.Linear(256, 2)
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
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Log training metrics to Weights and Biases
    wandb.watch(model, log="all")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

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


if __name__ == "__main__":
    load_dotenv()

    # Insert a unique RUN DATE comment into the script
    # - This asserts a new commit hash for comparing hyperparam and script configs
    insert_run_date_comment(script_path=__file__)

    # Check if GPU is available and set device accordingly
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize Weights and Biases
    # wandb.init(project="02_ML_Modem_PyTorch")
    # Step 3: Initialize WandB and store the Git commit hash
    initialize_wandb(project_name="02_ML_Modem_PyTorch")

    # Load the dataset
    dataset = get_dataset()

    # Split the dataset into 90% train and 10% test
    train_dataset, test_dataset = split_dataset(dataset)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Peek into the dataset to determine input and output shapes
    sample_input, sample_output = next(iter(train_loader))
    input_size = sample_input.shape[1]
    output_size = sample_output.shape[1]

    # Initialize the model
    # model = DenseNN(input_size=input_size, output_size=output_size)
    model = DenseNN(
        input_size=input_size,
        output_size=output_size,
        hidden_size=512,
        num_hidden_layers=3,
        dropout_rate=0.3
    )

    # Train the model
    train_model(
        model,
        train_loader,
        test_loader,
        early_stopping_strategy=PatienceEarlyStopping(patience=10),
        epochs=200,
        lr=1e-4,
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
