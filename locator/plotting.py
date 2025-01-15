"""Plotting functionality for locator predictions"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

__all__ = ["kde_predict", "plot_predictions", "plot_error_summary"]


def kde_predict(x_coords, y_coords, xlim=(0, 50), ylim=(0, 50), n_points=100):
    """Calculate kernel density estimate of predictions

    Args:
        x_coords: Array of x coordinates
        y_coords: Array of y coordinates
        xlim: Tuple of (min, max) x values for grid
        ylim: Tuple of (min, max) y values for grid
        n_points: Number of points for density estimation grid

    Returns:
        Tuple of (x_grid, y_grid, density)
    """
    try:
        # Calculate kernel density
        positions = np.vstack([x_coords, y_coords])
        kernel = gaussian_kde(positions)

        # Create grid of points using full plot range
        x_grid = np.linspace(xlim[0], xlim[1], n_points)
        y_grid = np.linspace(ylim[0], ylim[1], n_points)
        xx, yy = np.meshgrid(x_grid, y_grid)

        # Evaluate kernel on grid
        positions = np.vstack([xx.ravel(), yy.ravel()])
        density = np.reshape(kernel(positions).T, xx.shape)

        return xx, yy, density

    except Exception as e:
        print(f"KDE failed: {e}")
        return None, None, None


def plot_predictions(
    predictions,
    locator,
    out_prefix,
    samples=None,
    n_samples=9,
    n_cols=3,
    plot_map=False,
    width=5,
    height=4,
    dpi=300,
    xlim=(0, 50),
    ylim=(0, 50),
    n_levels=3,
):
    """Plot locator predictions from jacknife, bootstrap, or windows analyses.

    This function visualizes predictions from any of locator's prediction methods:
    - run_jacknife()
    - run_bootstraps()
    - run_windows()

    The function expects prediction data with:
    - A 'sampleID' column
    - Multiple prediction columns ('x_0', 'x_1'... and 'y_0', 'y_1'...)

    For each sample, the plot shows:
    - KDE contours of predictions (blue lines)
    - True location if known (red star)
    - All training sample locations (gray circles)

    Args:
        predictions: DataFrame or path to predictions file. Output from any of:
            - locator.run_jacknife(return_df=True)
            - locator.run_bootstraps(return_df=True)
            - locator.run_windows(return_df=True)
        locator: Locator instance containing training data configuration
        out_prefix: Prefix for output files
        samples: List of sample IDs to plot. If None, randomly selects n_samples
        n_samples: Number of samples to plot if samples not specified
        n_cols: Number of columns in plot grid
        plot_map: Whether to plot on a map (requires cartopy)
        width: Width of each subplot
        height: Height of each subplot
        dpi: DPI for output figure
        xlim: x-axis limits (min, max)
        ylim: y-axis limits (min, max)
        n_levels: Number of KDE contour levels to plot

    Returns:
        matplotlib figure object

    Example:
        >>> # For jacknife analysis
        >>> predictions = locator.run_jacknife(genotypes, samples, return_df=True)
        >>> plot_predictions(predictions, locator, "jacknife_example")

        >>> # For bootstrap analysis
        >>> predictions = locator.run_bootstraps(genotypes, samples, return_df=True)
        >>> plot_predictions(predictions, locator, "bootstrap_example")

        >>> # For windows analysis
        >>> predictions = locator.run_windows(genotypes, samples, return_df=True)
        >>> plot_predictions(predictions, locator, "windows_example")
    """
    # Load predictions
    if isinstance(predictions, (str, Path)):
        pred_path = Path(predictions)
        if pred_path.is_file():
            preds = pd.read_csv(pred_path)
        else:
            pred_files = list(pred_path.glob("*predlocs.txt"))
            preds = pd.concat([pd.read_csv(f) for f in pred_files])
    else:
        preds = predictions

    # Get sample data from locator
    samples_df = pd.read_csv(
        locator.config["sample_data"], sep="\t", na_values="NA", quotechar='"'
    )
    samples_df.columns = samples_df.columns.str.strip('"')
    if "sampleID" in samples_df.columns:
        samples_df["sampleID"] = samples_df["sampleID"].str.strip('"')

    # Select samples to plot if not provided
    if samples is None:
        available_samples = preds["sampleID"].unique()
        samples = np.random.choice(
            available_samples,
            size=min(n_samples, len(available_samples)),
            replace=False,
        )

    # Create figure
    n_rows = int(np.ceil(len(samples) / n_cols))
    fig = plt.figure(figsize=(width * n_cols, height * n_rows), dpi=dpi)

    # Plot each sample
    for i, sample in enumerate(samples, 1):
        ax = fig.add_subplot(
            n_rows, n_cols, i, projection=ccrs.PlateCarree() if plot_map else None
        )

        sample_preds = preds[preds["sampleID"] == sample]
        sample_true = samples_df[samples_df["sampleID"] == sample]

        if plot_map:
            ax.add_feature(cfeature.LAND, facecolor="lightgray")
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        else:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        # Plot all training sample locations as background
        # Only plot samples that have true locations (not NA)
        training_locs = samples_df[
            pd.notna(samples_df["x"]) & pd.notna(samples_df["y"])
        ]
        if not training_locs.empty:
            ax.scatter(
                training_locs["x"],
                training_locs["y"],
                c="gray",
                marker="o",
                s=20,
                facecolors="none",
                alpha=0.5,
                linewidth=0.5,
                label="Training samples",
            )

        # Plot predictions using KDE
        if any(col.startswith("x_") for col in preds.columns):
            # Multiple predictions per sample (e.g., jacknife)
            x_cols = [col for col in preds.columns if col.startswith("x_")]
            y_cols = [col for col in preds.columns if col.startswith("y_")]

            # Collect all predictions
            x_preds = sample_preds[x_cols].values.ravel()
            y_preds = sample_preds[y_cols].values.ravel()

            # Calculate KDE using plot limits
            xx, yy, density = kde_predict(x_preds, y_preds, xlim=xlim, ylim=ylim)
            if density is not None:
                # Calculate percentile-based contour levels
                density_flat = density.ravel()
                levels = np.percentile(density_flat[density_flat > 0], [85, 90, 95, 99])

                # Plot contour lines
                ax.contour(
                    xx,
                    yy,
                    density,
                    levels=levels,
                    colors="blue",
                    alpha=0.8,
                    linewidths=0.5,
                )

        # Plot true location if it exists and is not NA
        if len(sample_true) > 0 and pd.notna(sample_true.iloc[0]["x"]):
            ax.scatter(
                sample_true.iloc[0]["x"],
                sample_true.iloc[0]["y"],
                c="red",
                marker="*",
                s=100,
                label="True",
            )

        ax.set_title(f"Sample {sample}")

    plt.tight_layout()
    if out_prefix:
        plt.savefig(f"{out_prefix}_predictions.pdf")
    return fig


def plot_error_summary(
    predictions,
    sample_data,
    out_prefix=None,
    plot_map=True,
    width=20,
    height=10,
    dpi=300,
):
    """Plot summary of prediction errors from holdout analysis"""
    # Set larger font sizes globally
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    # Load sample data if path provided
    if isinstance(sample_data, (str, Path)):
        samples = pd.read_csv(sample_data, sep="\t")
    else:
        samples = sample_data

    # Merge predictions with true locations
    merged = predictions.merge(
        samples[["sampleID", "x", "y"]], on="sampleID", suffixes=("_pred", "_true")
    )

    # Calculate errors
    merged["error"] = np.sqrt(
        (merged["x_pred"] - merged["x_true"]) ** 2
        + (merged["y_pred"] - merged["y_true"]) ** 2
    )

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, height), dpi=dpi)

    if plot_map:
        ax1 = plt.subplot(121, projection=ccrs.PlateCarree())
        ax1.add_feature(cfeature.LAND, facecolor="lightgray")
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)

    # Plot errors on map with larger colorbar font
    scatter = ax1.scatter(
        merged["x_true"],
        merged["y_true"],
        c=merged["error"],
        cmap="RdYlBu_r",
        s=20,
    )
    plt.colorbar(scatter, ax=ax1, label="Error").ax.tick_params(labelsize=12)

    # Plot error connections
    for _, row in merged.iterrows():
        ax1.plot(
            [row["x_true"], row["x_pred"]],
            [row["y_true"], row["y_pred"]],
            "k-",
            linewidth=0.5,
            alpha=0.5,
        )

    # Plot error histogram with larger fonts
    sns.histplot(data=merged, x="error", ax=ax2)
    ax2.set_xlabel("Error", fontsize=14)
    ax2.set_ylabel("Count", fontsize=14)

    # Add summary statistics as text with larger font
    stats_text = (
        f"Mean error: {merged['error'].mean():.2f}\n"
        f"Median error: {merged['error'].median():.2f}\n"
        f"Max error: {merged['error'].max():.2f}\n"
        f"R² (x): {np.corrcoef(merged['x_pred'], merged['x_true'])[0,1]**2:.3f}\n"
        f"R² (y): {np.corrcoef(merged['y_pred'], merged['y_true'])[0,1]**2:.3f}"
    )
    ax2.text(
        0.95,
        0.95,
        stats_text,
        transform=ax2.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.8),
        fontsize=12,
    )

    plt.tight_layout()

    if out_prefix:
        plt.savefig(f"{out_prefix}_error_summary.png")

    plt.show()
    plt.close()
    return None
