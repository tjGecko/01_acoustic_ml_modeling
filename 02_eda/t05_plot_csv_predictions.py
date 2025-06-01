import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # For calculating error bar limits

# --- Configuration ---
# Replace with the actual path to your CSV file
CSV_FILE_PATH = "/home/tj/99_tmp/11 - synthetic mic array data/02_training_data/prediction_results/eval_20250601_122048/evaluation_results.csv"
OUTPUT_PLOT_DIR = "/home/tj/99_tmp/11 - synthetic mic array data/02_training_data/prediction_results/eval_20250601_122048/" # Directory to save plots

# Create output directory if it doesn't exist
import os
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

# --- Load the Data ---
try:
    df = pd.read_csv(CSV_FILE_PATH)
    print("CSV loaded successfully. Here's a preview:")
    print(df.head())
    print(f"\nNumber of records: {len(df)}")
    print(f"\nColumns: {df.columns.tolist()}")
except FileNotFoundError:
    print(f"Error: CSV file not found at {CSV_FILE_PATH}")
    exit()
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# --- Data Sanity Check (Optional but Recommended) ---
required_columns = [
    'true_azimuth_norm_dec', 'true_elevation_norm_dec',
    'azimuth_error_norm_dec', 'elevation_error_norm_dec',
    'pred_az_norm_dec', 'pred_el_norm_dec'
]
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    print(f"Error: Missing required columns in CSV: {missing_cols}")
    exit()

# --- Plotting Function ---
def plot_ground_truth_vs_prediction_with_error(
    true_values, pred_values, errors, title, xlabel, ylabel, output_filename
):
    """
    Creates a scatter plot of true vs. predicted values with error bars.

    Args:
        true_values (pd.Series): Series of true normalized values.
        pred_values (pd.Series): Series of predicted normalized values.
        errors (pd.Series): Series of error values (absolute difference).
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        output_filename (str): Filename to save the plot.
    """
    plt.figure(figsize=(10, 8))

    # Error bars represent the range [pred - error, pred + error]
    # Matplotlib's errorbar yerr can take a 2xN array for asymmetric errors,
    # or a 1xN array for symmetric errors. Since our 'error' is absolute,
    # it's symmetric around the prediction.
    # However, to visualize the range of where the true value *could* be given the prediction and error,
    # we often plot the true value on x and prediction on y.
    # The error bars then show the uncertainty of the prediction.

    # For each true_value (x-axis), plot the pred_value (y-axis dot)
    # with error bars extending from pred_value - error to pred_value + error.
    plt.errorbar(
        x=true_values,
        y=pred_values,
        yerr=errors,  # Symmetric error
        fmt='o',      # Format for the marker (o for dot)
        ecolor='lightcoral',
        elinewidth=1,
        capsize=3,    # Length of the error bar caps
        alpha=0.6,    # Transparency for better visualization if points overlap
        label='Prediction with Error Margin'
    )

    # Add a y=x line for reference (perfect prediction)
    min_val = min(true_values.min(), pred_values.min()) - 0.1 # Add a small margin
    max_val = max(true_values.max(), pred_values.max()) + 0.1 # Add a small margin
    # Ensure the limits cover the [-1, 1] normalized range if applicable
    min_val = max(min_val, -1.1)
    max_val = min(max_val, 1.1)

    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction (y=x)')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOT_DIR, output_filename))
    print(f"Plot saved to {os.path.join(OUTPUT_PLOT_DIR, output_filename)}")
    plt.show() # Display the plot


# --- Create Azimuth Plot ---
print("\nGenerating Azimuth plot...")
plot_ground_truth_vs_prediction_with_error(
    true_values=df['true_azimuth_norm_dec'],
    pred_values=df['pred_az_norm_dec'],
    errors=df['azimuth_error_norm_dec'],
    title='Ground Truth Azimuth vs. Predicted Azimuth (Normalized)',
    xlabel='True Normalized Azimuth (Decimal)',
    ylabel='Predicted Normalized Azimuth (Decimal)',
    output_filename='azimuth_gt_vs_pred_error.png'
)

# --- Create Elevation Plot ---
print("\nGenerating Elevation plot...")
plot_ground_truth_vs_prediction_with_error(
    true_values=df['true_elevation_norm_dec'],
    pred_values=df['pred_el_norm_dec'],
    errors=df['elevation_error_norm_dec'],
    title='Ground Truth Elevation vs. Predicted Elevation (Normalized)',
    xlabel='True Normalized Elevation (Decimal)',
    ylabel='Predicted Normalized Elevation (Decimal)',
    output_filename='elevation_gt_vs_pred_error.png'
)

# --- Additional Analysis: Error Distributions (Histograms) ---
def plot_error_distribution(errors, angle_type, output_filename):
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    mean_error = errors.mean()
    median_error = errors.median()
    plt.axvline(mean_error, color='red', linestyle='dashed', linewidth=2, label=f'Mean Error: {mean_error:.4f}')
    plt.axvline(median_error, color='green', linestyle='dashed', linewidth=2, label=f'Median Error: {median_error:.4f}')
    plt.title(f'Distribution of Normalized {angle_type} Error')
    plt.xlabel(f'Normalized {angle_type} Error (Decimal)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOT_DIR, output_filename))
    print(f"Plot saved to {os.path.join(OUTPUT_PLOT_DIR, output_filename)}")
    plt.show()

print("\nGenerating Azimuth Error Distribution plot...")
plot_error_distribution(
    errors=df['azimuth_error_norm_dec'],
    angle_type='Azimuth',
    output_filename='azimuth_error_distribution.png'
)

print("\nGenerating Elevation Error Distribution plot...")
plot_error_distribution(
    errors=df['elevation_error_norm_dec'],
    angle_type='Elevation',
    output_filename='elevation_error_distribution.png'
)

print("\nGenerating Total Error Distribution plot...")
plot_error_distribution(
    errors=df['total_error_norm_dec'],
    angle_type='Total Angular',
    output_filename='total_error_distribution.png'
)

print("\nAll plots generated.")