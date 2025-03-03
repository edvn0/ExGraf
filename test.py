from multiprocessing import Pool, cpu_count
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.gridspec as gridspec

matplotlib.use("Agg")  # Ensure compatibility with headless environments

# File locations
CONFUSION_FILE = "confusion_matrices/all_confusions.csv"
OUTPUT_DIR = "confusion_images"


def load_confusion_matrices():
    """Loads confusion matrices from a single CSV file into a structured format."""
    if not os.path.exists(CONFUSION_FILE):
        print(f"File {CONFUSION_FILE} not found.")
        return []

    df = pd.read_csv(CONFUSION_FILE, header=None)
    num_classes = int(np.sqrt(df.shape[1] - 1))  # Infer number of classes
    if num_classes * num_classes != df.shape[1] - 1:
        raise ValueError(
            "Matrix dimensions do not match expected square shape.")

    # Extract epoch numbers and reshape matrix values
    epochs = df.iloc[:, 0].astype(int)  # First column is epoch
    matrices = df.iloc[:, 1:].values.reshape(len(df), num_classes, num_classes)

    return list(zip(epochs, matrices))


def generate_plot(epoch, matrix):
    """Generates a confusion matrix plot and returns the figure and output path."""
    vmin, vmax = np.min(matrix), np.max(matrix)
    off_diag_mask = np.eye(*matrix.shape, dtype=bool)  # Mask for diagonal

    fig = plt.figure(figsize=(9, 7))
    gs0 = gridspec.GridSpec(1, 2, width_ratios=[20, 2], wspace=0.2)
    gs00 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs0[1], wspace=0)

    ax = fig.add_subplot(gs0[0])
    cax1 = fig.add_subplot(gs00[0])
    cax2 = fig.add_subplot(gs00[1])

    # Heatmaps with proper formatting
    sns.heatmap(matrix, annot=True, fmt=".0f", annot_kws={"size": 10},
                mask=~off_diag_mask, cmap='Blues', vmin=vmin, vmax=vmax, ax=ax, cbar_ax=cax2)
    sns.heatmap(matrix, annot=True, fmt=".0f", annot_kws={"size": 10},
                mask=off_diag_mask, cmap='OrRd', vmin=vmin, vmax=vmax, ax=ax, cbar_ax=cax1, cbar_kws={"ticks": []})

    ax.set_title(f"Confusion Matrix - Epoch {epoch}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    output_path = os.path.join(
        OUTPUT_DIR, f"confusion_matrix_epoch_{epoch}.png")

    return fig, output_path


def save_plot(fig, output_path):
    """Saves a figure to disk."""
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)  # Free memory
    print(f"Saved: {output_path}")


def save_confusion_matrices():
    """Processes matrices in parallel, generates plots, and batches I/O operations."""
    matrices = load_confusion_matrices()

    if not matrices:
        print("No confusion matrices found for visualization.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with Pool(cpu_count()) as pool:
        results = pool.starmap(generate_plot, matrices)

    with Pool(cpu_count()) as pool:
        pool.starmap(save_plot, results)


if __name__ == "__main__":
    save_confusion_matrices()
