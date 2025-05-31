import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns

# Set style for better looking plots
plt.style.use('seaborn-whitegrid')
sns.set_palette("Set2")

# Load the tensor data
input_tensor_path = Path("/home/tj/99_tmp/11 - synthetic mic array data/02_training_data/02_test_split/0afbfc45400a5cedba38fdbef067688c.npy")
tensor_data = np.load(input_tensor_path)

# Extract the first channel (assuming shape is [channels, time_steps, features])
if len(tensor_data.shape) == 3:  # If 3D tensor [channels, time, features]
    channel_data = tensor_data[0, :, :].flatten()
else:  # If 2D tensor [time, features]
    channel_data = tensor_data.flatten()

# Calculate statistics
stats = {
    'Min': np.min(channel_data),
    'Max': np.max(channel_data),
    'Mean': np.mean(channel_data),
    'Median': np.median(channel_data),
    'Std Dev': np.std(channel_data),
    '25th Percentile': np.percentile(channel_data, 25),
    '75th Percentile': np.percentile(channel_data, 75)
}

# Create figure and axis
plt.figure(figsize=(8, 10))

# Create boxplot with points
sns.boxplot(y=channel_data, color='lightblue', width=0.3, fliersize=0)
sns.stripplot(y=channel_data, color='#4e79a7', alpha=0.2, size=3, jitter=0.2)

# Add title and labels
plt.title('Channel 1 Data Distribution', pad=20, fontsize=14, fontweight='bold')
plt.ylabel('Value')

# Add statistics as text
stats_text = (
    f"Min: {stats['Min']:.4f}\n"
    f"Max: {stats['Max']:.4f}\n"
    f"Mean: {stats['Mean']:.4f}\n"
    f"Median: {stats['Median']:.4f}\n"
    f"Std Dev: {stats['Std Dev']:.4f}\n"
    f"25th %ile: {stats['25th Percentile']:.4f}\n"
    f"75th %ile: {stats['75th Percentile']:.4f}"
)

# Add a nice background for the stats
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
         verticalalignment='top', bbox=props, fontfamily='monospace',
         fontsize=10, color='black')

# Adjust layout
plt.tight_layout()

# Save the plot
output_file_path = input_tensor_path.parent / 'channel1_boxplot.png'
plt.savefig(output_file_path, dpi=300, bbox_inches='tight')
print(f"Box plot saved to: {output_file_path}")

# Print statistics to console
print("\nChannel 1 Statistics:")
for stat, value in stats.items():
    print(f"{stat}: {value:.4f}")

# Show the plot
plt.show()