import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- AngleNormalizer Class ---
class AngleNormalizer:
    """Normalize angles to [-1, 1] range for training."""
    def __init__(self, az_range=(-90, 90), el_range=(-90, 90)):
        self.az_range = az_range
        self.el_range = el_range
        self.az_span = az_range[1] - az_range[0]
        self.el_span = el_range[1] - el_range[0]

    def normalize(self, az, el):
        """Normalize angles to [-1, 1] range."""
        az_norm = 2 * (az - self.az_range[0]) / self.az_span - 1
        el_norm = 2 * (el - self.el_range[0]) / self.el_span - 1
        return az_norm, el_norm

    def denormalize(self, az_norm, el_norm):
        """Convert normalized angles back to original range."""
        az = (az_norm + 1) * self.az_span / 2 + self.az_range[0]
        el = (el_norm + 1) * self.el_span / 2 + self.el_range[0]
        return az, el

    def denormalize_error(self, norm_error_az, norm_error_el):
        """Convert normalized error magnitudes back to degrees."""
        error_deg_az = norm_error_az * self.az_span / 2
        error_deg_el = norm_error_el * self.el_span / 2
        return error_deg_az, error_deg_el

# --- Configuration ---
CSV_FILE_PATH = "/home/tj/99_tmp/11 - synthetic mic array data/02_training_data/prediction_results/eval_20250601_122048/evaluation_results.csv"
OUTPUT_PLOT_DIR = "/home/tj/99_tmp/11 - synthetic mic array data/02_training_data/prediction_results/eval_20250601_122048/" # Directory to save plots
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

# Define the angle ranges used during normalization/training
# !!! UPDATE THESE TO MATCH YOUR ACTUAL TRAINING SETUP !!!
AZIMUTH_RANGE_DEGREES = (0, 360)    # Example: Full circle azimuth
ELEVATION_RANGE_DEGREES = (-90, 90) # Example: Standard elevation

# Initialize the normalizer
normalizer = AngleNormalizer(az_range=AZIMUTH_RANGE_DEGREES, el_range=ELEVATION_RANGE_DEGREES)

# --- Load the Data ---
try:
    df = pd.read_csv(CSV_FILE_PATH)
    print("CSV loaded successfully. Here's a preview:")
    print(df.head())
except FileNotFoundError:
    print(f"Error: CSV file not found at {CSV_FILE_PATH}")
    exit()
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# --- Denormalize Data for Plotting ---
# Denormalize true angles
df['true_azimuth_deg'], df['true_elevation_deg'] = normalizer.denormalize(
    df['true_azimuth_norm_dec'],
    df['true_elevation_norm_dec']
)

# Predicted angles are already in degrees in the CSV ('pred_azimuth_deg', 'pred_elevation_deg')
# If they weren't, you'd denormalize them:
# df['pred_azimuth_deg_calc'], df['pred_elevation_deg_calc'] = normalizer.denormalize(
#     df['pred_az_norm_dec'],
#     df['pred_el_norm_dec']
# )
# We'll use the existing 'pred_azimuth_deg' and 'pred_elevation_deg' columns.

# Denormalize error magnitudes
df['azimuth_error_deg'], df['elevation_error_deg'] = normalizer.denormalize_error(
    df['azimuth_error_norm_dec'],
    df['elevation_error_norm_dec']
)

# Calculate total error in degrees (this assumes a simple Euclidean distance on denormalized errors,
# which might not be perfectly accurate for spherical coordinates but is often used as an approximation)
df['total_error_deg'] = np.sqrt(df['azimuth_error_deg']**2 + df['elevation_error_deg']**2)


print("\nDataFrame with denormalized values (degrees):")
print(df[['true_azimuth_deg', 'pred_azimuth_deg', 'azimuth_error_deg',
          'true_elevation_deg', 'pred_elevation_deg', 'elevation_error_deg']].head())


# --- Plotting Function (Modified for Degrees) ---
def plot_ground_truth_vs_prediction_with_error_degrees(
    true_values_deg, pred_values_deg, errors_deg,
    angle_range_deg, title, xlabel, ylabel, output_filename
):
    plt.figure(figsize=(10, 8))

    plt.errorbar(
        x=true_values_deg,
        y=pred_values_deg,
        yerr=errors_deg,
        fmt='o',
        ecolor='lightcoral',
        elinewidth=1,
        capsize=3,
        alpha=0.6,
        label='Prediction with Error Margin (Degrees)'
    )

    min_val_plot = angle_range_deg[0] - 0.05 * (angle_range_deg[1] - angle_range_deg[0])
    max_val_plot = angle_range_deg[1] + 0.05 * (angle_range_deg[1] - angle_range_deg[0])

    plt.plot([angle_range_deg[0], angle_range_deg[1]],
             [angle_range_deg[0], angle_range_deg[1]],
             'k--', lw=2, label='Perfect Prediction (y=x)')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(min_val_plot, max_val_plot)
    plt.ylim(min_val_plot, max_val_plot)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOT_DIR, output_filename))
    print(f"Plot saved to {os.path.join(OUTPUT_PLOT_DIR, output_filename)}")
    plt.show()


# --- Create Azimuth Plot (Degrees) ---
print("\nGenerating Azimuth plot (Degrees)...")
plot_ground_truth_vs_prediction_with_error_degrees(
    true_values_deg=df['true_azimuth_deg'],
    pred_values_deg=df['pred_azimuth_deg'], # Using the pre-calculated prediction in degrees
    errors_deg=df['azimuth_error_deg'],
    angle_range_deg=AZIMUTH_RANGE_DEGREES,
    title='Ground Truth Azimuth vs. Predicted Azimuth (Degrees)',
    xlabel='True Azimuth (Degrees)',
    ylabel='Predicted Azimuth (Degrees)',
    output_filename='azimuth_gt_vs_pred_error_degrees.png'
)

# --- Create Elevation Plot (Degrees) ---
print("\nGenerating Elevation plot (Degrees)...")
plot_ground_truth_vs_prediction_with_error_degrees(
    true_values_deg=df['true_elevation_deg'],
    pred_values_deg=df['pred_elevation_deg'], # Using the pre-calculated prediction in degrees
    errors_deg=df['elevation_error_deg'],
    angle_range_deg=ELEVATION_RANGE_DEGREES,
    title='Ground Truth Elevation vs. Predicted Elevation (Degrees)',
    xlabel='True Elevation (Degrees)',
    ylabel='Predicted Elevation (Degrees)',
    output_filename='elevation_gt_vs_pred_error_degrees.png'
)


# --- Error Distributions in Degrees (Histograms) ---
def plot_error_distribution_degrees(errors_deg, angle_type, output_filename):
    plt.figure(figsize=(8, 6))
    plt.hist(errors_deg, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    mean_error = errors_deg.mean()
    median_error = errors_deg.median()
    plt.axvline(mean_error, color='red', linestyle='dashed', linewidth=2, label=f'Mean Error: {mean_error:.2f}°')
    plt.axvline(median_error, color='green', linestyle='dashed', linewidth=2, label=f'Median Error: {median_error:.2f}°')
    plt.title(f'Distribution of {angle_type} Error (Degrees)')
    plt.xlabel(f'{angle_type} Error (Degrees)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOT_DIR, output_filename))
    print(f"Plot saved to {os.path.join(OUTPUT_PLOT_DIR, output_filename)}")
    plt.show()

print("\nGenerating Azimuth Error Distribution plot (Degrees)...")
plot_error_distribution_degrees(
    errors_deg=df['azimuth_error_deg'],
    angle_type='Azimuth',
    output_filename='azimuth_error_distribution_degrees.png'
)

print("\nGenerating Elevation Error Distribution plot (Degrees)...")
plot_error_distribution_degrees(
    errors_deg=df['elevation_error_deg'],
    angle_type='Elevation',
    output_filename='elevation_error_distribution_degrees.png'
)

print("\nGenerating Total Error Distribution plot (Degrees)...")
plot_error_distribution_degrees(
    errors_deg=df['total_error_deg'],
    angle_type='Total Angular',
    output_filename='total_error_distribution_degrees.png'
)

print("\nAll plots generated.")