"""
Plotting utilities for egg fertility detection.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import seaborn as sns


def plot_training_curve(
    train_losses: List[float],
    val_losses: List[float],
    train_ious: List[float],
    val_ious: List[float],
    train_dices: List[float],
    val_dices: List[float],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot training and validation curves."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Loss curve
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_title('Loss Curve')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # IoU curve
    axes[1].plot(train_ious, label='Train IoU')
    axes[1].plot(val_ious, label='Val IoU')
    axes[1].set_title('IoU Curve')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('IoU')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Dice curve
    axes[2].plot(train_dices, label='Train Dice')
    axes[2].plot(val_dices, label='Val Dice')
    axes[2].set_title('Dice Coefficient Curve')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Dice Coefficient')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_image_grid(
    images: List[np.ndarray],
    titles: List[str],
    cols: int = 4,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot grid of images."""
    rows = (len(images) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        axes[i].set_title(title, fontsize=10)
        axes[i].axis('off')
    
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_classification_report(report: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
    """Plot classification report."""
    metrics = ['precision', 'recall', 'f1-score']
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 6))
    
    for i, metric in enumerate(metrics):
        classes = list(report.keys())
        values = [report[cls][metric] for cls in classes]
        
        axes[i].bar(classes, values, color=['skyblue', 'lightgreen', 'orange'])
        axes[i].set_title(metric.replace('-', ' ').capitalize())
        axes[i].set_ylabel('Score')
        axes[i].set_ylim(0, 1)
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def create_subplot_figure(num_plots: int, cols: int = 2, figsize: tuple = (12, 8)) -> plt.Figure:
    """Create figure with subplots."""
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    return fig, axes


def save_figure(fig: plt.Figure, file_path: str) -> None:
    """Save figure to file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_histogram(data: np.ndarray, title: str = 'Histogram', save_path: Optional[str] = None) -> plt.Figure:
    """Plot histogram of data."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data.flatten(), bins=256, range=(0, 256), color='skyblue', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_image_with_metric(
    image: np.ndarray,
    mask: np.ndarray,
    metric: float,
    metric_name: str,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot image with mask and metric."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title(f'Mask ({metric_name}: {metric:.3f})')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig
