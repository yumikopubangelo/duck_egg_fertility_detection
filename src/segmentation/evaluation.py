"""Utilities for multiclass U-Net segmentation evaluation and reporting."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch

DEFAULT_CLASS_NAMES = ["background", "vascularization", "embryo"]
DEFAULT_CLASS_COLORS = {
    0: (30, 41, 59),    # slate
    1: (220, 38, 38),   # red
    2: (22, 163, 74),   # green
}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def predict_class_map(logits: torch.Tensor) -> torch.Tensor:
    """Convert model logits to integer class indices."""
    if logits.ndim != 4:
        raise ValueError(f"Expected logits with shape [B, C, H, W], got {tuple(logits.shape)}")
    if logits.shape[1] == 1:
        return (torch.sigmoid(logits).squeeze(1) >= 0.5).long()
    return torch.argmax(logits, dim=1).long()


def confusion_matrix_for_masks(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Build a pixel-level confusion matrix with rows=true, cols=pred."""
    pred_flat = np.asarray(pred_mask, dtype=np.int64).ravel()
    true_flat = np.asarray(true_mask, dtype=np.int64).ravel()
    valid = (true_flat >= 0) & (true_flat < num_classes)
    encoded = num_classes * true_flat[valid] + pred_flat[valid]
    counts = np.bincount(encoded, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def metrics_from_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Sequence[str] | None = None,
    background_index: int = 0,
) -> dict:
    """Compute multiclass segmentation metrics from a confusion matrix."""
    cm = np.asarray(confusion_matrix, dtype=np.float64)
    num_classes = cm.shape[0]
    class_names = list(class_names or DEFAULT_CLASS_NAMES[:num_classes])

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (tp + fp + fn)
        dice = (2 * tp) / (2 * tp + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)

    support = cm.sum(axis=1)
    total_pixels = int(cm.sum())
    pixel_accuracy = float(tp.sum() / total_pixels) if total_pixels > 0 else 0.0

    foreground_indices = [idx for idx in range(num_classes) if idx != background_index]

    def _safe_mean(values: np.ndarray, indices: Iterable[int] | None = None) -> float:
        selected = values[list(indices)] if indices is not None else values
        selected = selected[~np.isnan(selected)]
        return float(selected.mean()) if selected.size else 0.0

    per_class = []
    for idx in range(num_classes):
        per_class.append(
            {
                "class_id": idx,
                "name": class_names[idx] if idx < len(class_names) else f"class_{idx}",
                "support_pixels": int(support[idx]),
                "support_pct": round(float(support[idx] / total_pixels * 100), 4)
                if total_pixels > 0
                else 0.0,
                "iou": float(0.0 if np.isnan(iou[idx]) else iou[idx]),
                "dice": float(0.0 if np.isnan(dice[idx]) else dice[idx]),
                "precision": float(0.0 if np.isnan(precision[idx]) else precision[idx]),
                "recall": float(0.0 if np.isnan(recall[idx]) else recall[idx]),
                "specificity": float(0.0 if np.isnan(specificity[idx]) else specificity[idx]),
            }
        )

    return {
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": _safe_mean(iou),
        "mean_dice": _safe_mean(dice),
        "foreground_mean_iou": _safe_mean(iou, foreground_indices) if foreground_indices else _safe_mean(iou),
        "foreground_mean_dice": _safe_mean(dice, foreground_indices) if foreground_indices else _safe_mean(dice),
        "per_class": per_class,
        "confusion_matrix": cm.astype(np.int64).tolist(),
        "total_pixels": total_pixels,
        "num_classes": num_classes,
        "class_names": class_names,
    }


def per_sample_segmentation_metrics(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    class_names: Sequence[str] | None = None,
    background_index: int = 0,
) -> dict:
    """Compute per-sample multiclass metrics."""
    num_classes = int(max(np.max(pred_mask), np.max(true_mask)) + 1)
    report = metrics_from_confusion_matrix(
        confusion_matrix_for_masks(pred_mask, true_mask, num_classes=num_classes),
        class_names=class_names,
        background_index=background_index,
    )
    return {
        "mean_iou": report["mean_iou"],
        "mean_dice": report["mean_dice"],
        "foreground_mean_iou": report["foreground_mean_iou"],
        "foreground_mean_dice": report["foreground_mean_dice"],
        "per_class": report["per_class"],
    }


def denormalize_image_tensor(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized CHW tensor to uint8 RGB image."""
    img = image_tensor.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def colorize_mask(mask: np.ndarray, class_colors: dict[int, tuple[int, int, int]] | None = None) -> np.ndarray:
    """Convert class-index mask to RGB visualization."""
    colors = class_colors or DEFAULT_CLASS_COLORS
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, color in colors.items():
        rgb[np.asarray(mask) == cls] = np.asarray(color, dtype=np.uint8)
    return rgb


def overlay_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    class_colors: dict[int, tuple[int, int, int]] | None = None,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a predicted mask on top of the RGB image."""
    colors = class_colors or DEFAULT_CLASS_COLORS
    image = np.asarray(image_rgb, dtype=np.float32).copy()
    overlay_img = image.copy()
    for cls, color in colors.items():
        if cls == 0:
            continue
        overlay_img[np.asarray(mask) == cls] = np.asarray(color, dtype=np.float32)
    blended = image * (1.0 - alpha) + overlay_img * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)

