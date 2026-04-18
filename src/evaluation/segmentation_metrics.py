"""Segmentation metric implementations."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def _binarize(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.dtype != np.bool_:
        mask = mask > threshold
    return mask.astype(bool)


@dataclass
class IoU:
    """Intersection over Union."""

    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
        true = _binarize(y_true, threshold)
        pred = _binarize(y_pred, threshold)

        intersection = np.logical_and(true, pred).sum()
        union = np.logical_or(true, pred).sum()
        if union == 0:
            return 1.0
        return float(intersection / union)


@dataclass
class DiceCoefficient:
    """Dice coefficient."""

    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
        true = _binarize(y_true, threshold)
        pred = _binarize(y_pred, threshold)

        intersection = np.logical_and(true, pred).sum()
        denom = true.sum() + pred.sum()
        if denom == 0:
            return 1.0
        return float((2.0 * intersection) / denom)


@dataclass
class PixelAccuracy:
    """Pixel-wise accuracy."""

    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
        true = _binarize(y_true, threshold)
        pred = _binarize(y_pred, threshold)
        return float((true == pred).mean())


@dataclass
class BoundaryF1Score:
    """Boundary F1 score using Canny edge approximation."""

    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
        true = (_binarize(y_true, threshold).astype(np.uint8)) * 255
        pred = (_binarize(y_pred, threshold).astype(np.uint8)) * 255

        true_edges = cv2.Canny(true, 100, 200) > 0
        pred_edges = cv2.Canny(pred, 100, 200) > 0

        tp = np.logical_and(true_edges, pred_edges).sum()
        fp = np.logical_and(~true_edges, pred_edges).sum()
        fn = np.logical_and(true_edges, ~pred_edges).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        if precision + recall == 0:
            return 0.0
        return float(2 * precision * recall / (precision + recall))
