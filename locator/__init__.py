"""Locator: A tool for predicting geographic location from genetic variation"""

from .core import Locator
from .plotting import plot_predictions, plot_error_summary
from .models import create_network, euclidean_distance_loss

__version__ = "0.1.0"

# Make the package namespace clean and complete
__all__ = [
    # Main class
    "Locator",
    # Plotting functions
    "plot_predictions",
    "plot_error_summary",
    # Model functions
    "create_network",
    "euclidean_distance_loss",
]
