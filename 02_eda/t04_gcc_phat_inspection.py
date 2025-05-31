import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent

from s10_src.p20_ml_model.dl01_time_series_16_channels import TimeSeries16ChannelDataset
from s10_src.p20_ml_model.m02_gcc_phat_features import GCCPHATFeatures


def load_sample_dataset():
    """Load a single sample from the dataset."""
    # Update these paths according to your dataset structure
    manifest_path = Path('/home/tj/99_tmp/11 - synthetic mic array data/02_training_data/manifest.json')

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found.")
    
    dataset = TimeSeries16ChannelDataset(
        manifest_path=manifest_path,
        data_dir=None,
        shuffle=False
    )
    
    # Get first sample
    sample, _ = dataset[0]
    return sample.unsqueeze(0)  # Add batch dimension

def plot_gcc_phat_features(sample, fs=16000, n_samples=4096, max_tau=0.001, interp_factor=4):
    """
    Compute and plot GCC-PHAT features for specified microphone pairs.
    
    Args:
        sample: Input tensor of shape (batch_size, n_channels, n_samples)
        fs: Sampling rate in Hz
        n_samples: Number of samples per channel
        max_tau: Maximum expected time delay in seconds
        interp_factor: Upsampling factor for cross-correlation
    """
    # Define microphone pairs: mic 2 with mics [15, 9, 8, 14] (0-based indexing)
    mic_pairs = [(2, 15), (2, 9), (2, 8), (2, 14)]
    
    # Initialize GCC-PHAT feature extractor for 2 microphones
    gcc_phat = GCCPHATFeatures(
        n_mics=2,  # We'll process one pair at a time
        fs=fs,
        n_samples_in_frame=n_samples,
        max_tau=max_tau,
        interp_factor=interp_factor
    )
    
    # Compute time lags
    max_shift_samples = int(np.ceil(max_tau * fs * interp_factor))
    taus = np.linspace(-max_tau * 1e6, max_tau * 1e6, 2 * max_shift_samples + 1)  # in microseconds
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, (i, j) in enumerate(mic_pairs):
        # Extract the specific pair (shape: [batch_size, 2, n_samples])
        pair_tensor = sample[:, [i, j], :]
        
        # Compute GCC-PHAT for this pair
        with torch.no_grad():
            gcc = gcc_phat(pair_tensor)
        
        # Plot - ensure we're working with 1D numpy arrays
        ax = axes[idx]
        gcc_np = gcc.squeeze().numpy()  # Remove any singleton dimensions
        ax.plot(taus, gcc_np, 'b-', linewidth=1.5)
        ax.axvline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_title(f'GCC-PHAT: Mic {i+1} vs Mic {j+1}')
        ax.set_xlabel('Time Delay (µs)')
        ax.set_ylabel('Correlation')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    print("Loading sample data...")
    try:
        sample = load_sample_dataset()
        print(f"Sample loaded with shape: {sample.shape}")
        
        print("Computing GCC-PHAT features...")
        fig = plot_gcc_phat_features(sample)
        
        # Save figure
        output_dir = project_root / '02_eda' / 'figures'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'gcc_phat_features.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
