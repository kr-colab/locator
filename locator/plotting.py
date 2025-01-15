"""Plotting functionality for locator predictions"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path


def kde_predict(x_coords, y_coords, n_points=500):
    """Calculate kernel density estimate of predictions

    Args:
        x_coords: Array of x coordinates
        y_coords: Array of y coordinates
        n_points: Number of points for density estimation grid

    Returns:
        Tuple of (x_grid, y_grid, density)
    """
    try:
        # Calculate kernel density
        positions = np.vstack([x_coords, y_coords])
        kernel = gaussian_kde(positions)

        # Create grid of points
        x_grid = np.linspace(min(x_coords), max(x_coords), n_points)
        y_grid = np.linspace(min(y_coords), max(y_coords), n_points)
        xx, yy = np.meshgrid(x_grid, y_grid)

        # Evaluate kernel on grid
        positions = np.vstack([xx.ravel(), yy.ravel()])
        density = np.reshape(kernel(positions).T, xx.shape)

        return x_grid, y_grid, density

    except Exception:
        # Return mean coordinates if KDE fails
        return (np.mean(x_coords), np.mean(y_coords), None)


def plot_predictions(
    predictions,  # Can be path to file/dir or DataFrame from run functions
    sample_data,
    out_prefix,
    samples=None,
    n_samples=9,
    n_cols=3,
    plot_map=True,
    width=5,
    height=4,
    dpi=300,
):
    """Plot locator predictions

    Args:
        predictions: Path to prediction file/dir or DataFrame from run functions
        sample_data: Path to sample data file
        out_prefix: Prefix for output files
        samples: List of sample IDs to plot (default: None)
        n_samples: Number of random samples to plot if samples not specified
        n_cols: Number of columns in multi-panel plot
        plot_map: Whether to plot background map
        width: Figure width in inches
        height: Figure height in inches
        dpi: Figure resolution
    """
    # Load predictions
    if isinstance(predictions, (str, Path)):
        pred_path = Path(predictions)
        if pred_path.is_file():
            preds = pd.read_csv(pred_path)
        else:
            # Load multiple prediction files
            pred_files = list(pred_path.glob("*predlocs.txt"))
            preds = pd.concat([pd.read_csv(f) for f in pred_files])
    else:
        # Use provided DataFrame
        preds = predictions

    # Load sample data
    if isinstance(sample_data, (str, Path)):
        samples_df = pd.read_csv(sample_data, sep="\t")
    else:
        # Assume sample_data is already a DataFrame
        samples_df = sample_data

    # Select samples to plot
    if samples is None:
        samples = np.random.choice(preds["sampleID"].unique(), n_samples, replace=False)
    elif isinstance(samples, str):
        samples = samples.split(",")

    # Create figure
    n_rows = int(np.ceil(len(samples) / n_cols))
    fig = plt.figure(figsize=(width, height), dpi=dpi)

    # Plot each sample
    for i, sample in enumerate(samples, 1):
        ax = fig.add_subplot(n_rows, n_cols, i, projection=ccrs.PlateCarree())

        sample_preds = preds[preds["sampleID"] == sample]
        sample_true = samples_df[samples_df["sampleID"] == sample].iloc[0]

        if plot_map:
            ax.add_feature(cfeature.LAND, facecolor="lightgray")
            ax.add_feature(cfeature.OCEAN, facecolor="white")
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

        # Plot training locations
        ax.scatter(
            samples_df["x"],
            samples_df["y"],
            c="dodgerblue",
            marker="o",
            s=20,
            alpha=0.5,
        )

        # Get x and y coordinates from prediction columns
        x_cols = [col for col in sample_preds.columns if col.startswith("x_")]
        y_cols = [col for col in sample_preds.columns if col.startswith("y_")]

        x_preds = sample_preds[x_cols].values.flatten()
        y_preds = sample_preds[y_cols].values.flatten()

        # Plot predictions
        ax.scatter(x_preds, y_preds, c="black", s=10, alpha=0.7)

        # Plot true location
        ax.scatter(sample_true["x"], sample_true["y"], c="red", marker="*", s=100)

        # Plot KDE contours
        x_grid, y_grid, density = kde_predict(x_preds, y_preds)
        if density is not None:
            levels = np.percentile(density, [10, 50, 95])
            ax.contour(
                x_grid, y_grid, density, levels=levels, colors="black", alpha=0.5
            )

        ax.set_title(sample)

    plt.tight_layout()
    plt.savefig(f"{out_prefix}_predictions.png")
    plt.close()


def plot_error_summary(
    pred_file,
    sample_data,
    out_prefix,
    plot_map=True,
    width=6,
    height=4,
    dpi=300,
):
    """Plot summary of prediction errors

    Args:
        pred_file: Path to prediction file
        sample_data: Path to sample data file
        out_prefix: Prefix for output files
        plot_map: Whether to plot background map
        width: Figure width in inches
        height: Figure height in inches
        dpi: Figure resolution
    """
    # Load data
    preds = pd.read_csv(pred_file)
    samples = pd.read_csv(sample_data, sep="\t")

    # Calculate errors
    errors = []
    for _, row in samples.iterrows():
        sample_preds = preds[preds["sampleID"] == row["sampleID"]]
        if len(sample_preds) > 0:
            x_grid, y_grid, density = kde_predict(sample_preds["x"], sample_preds["y"])
            if density is not None:
                pred_x = x_grid[density.argmax() // density.shape[1]]
                pred_y = y_grid[density.argmax() % density.shape[1]]
                error = np.sqrt((pred_x - row["x"]) ** 2 + (pred_y - row["y"]) ** 2)
                errors.append(
                    {
                        "sampleID": row["sampleID"],
                        "true_x": row["x"],
                        "true_y": row["y"],
                        "pred_x": pred_x,
                        "pred_y": pred_y,
                        "error": error,
                    }
                )

    errors = pd.DataFrame(errors)

    # Plot error summary
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, height), dpi=dpi)

    if plot_map:
        ax1 = plt.subplot(121, projection=ccrs.PlateCarree())
        ax1.add_feature(cfeature.LAND, facecolor="lightgray")
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)

    # Plot errors on map
    scatter = ax1.scatter(
        errors["true_x"], errors["true_y"], c=errors["error"], cmap="RdYlBu_r", s=50
    )
    plt.colorbar(scatter, ax=ax1, label="Error")

    # Plot error connections
    for _, row in errors.iterrows():
        ax1.plot(
            [row["true_x"], row["pred_x"]],
            [row["true_y"], row["pred_y"]],
            "k-",
            linewidth=0.5,
            alpha=0.5,
        )

    # Plot error histogram
    sns.histplot(data=errors, x="error", ax=ax2)
    ax2.set_xlabel("Error")
    ax2.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(f"{out_prefix}_error_summary.png")
    plt.close()
