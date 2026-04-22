"""Visualization utilities for model evaluation."""

from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def plot_confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    labels: Optional[list[str]] = None,
    normalize: bool = False,
    save_path: Optional[str] = None,
):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_roc_curve(
    y_true: Iterable[int],
    y_score: Iterable[float],
    save_path: Optional[str] = None,
):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_precision_recall(
    y_true: Iterable[int],
    y_score: Iterable[float],
    save_path: Optional[str] = None,
):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def visualize_segmentation(
    image: np.ndarray,
    true_mask: np.ndarray,
    pred_mask: np.ndarray,
    save_path: Optional[str] = None,
):
    """Visualize image, ground truth mask, and predicted mask side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].imshow(image, cmap=None if image.ndim == 3 else "gray")
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(true_mask, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
