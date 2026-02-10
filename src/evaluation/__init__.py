"""
Evaluation metrics and visualization package.
"""

from .metrics import (
    Accuracy, Precision, Recall, F1Score,
    ConfusionMatrix, ClassificationReport
)
from .segmentation_metrics import (
    IoU, DiceCoefficient, PixelAccuracy,
    BoundaryF1Score
)
from .visualization import (
    plot_confusion_matrix, plot_roc_curve,
    plot_precision_recall, visualize_segmentation
)

__all__ = [
    "Accuracy", "Precision", "Recall", "F1Score",
    "ConfusionMatrix", "ClassificationReport",
    "IoU", "DiceCoefficient", "PixelAccuracy", "BoundaryF1Score",
    "plot_confusion_matrix", "plot_roc_curve",
    "plot_precision_recall", "visualize_segmentation"
]
