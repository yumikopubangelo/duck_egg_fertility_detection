"""Post-processing helpers for binary segmentation masks."""

from __future__ import annotations

import cv2
import numpy as np


def _to_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Normalize a mask to uint8 binary format (0 or 255)."""
    return (np.asarray(mask) > 0).astype(np.uint8) * 255


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes in a binary mask."""
    binary = _to_binary_mask(mask)
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    filled = cv2.bitwise_or(binary, cv2.bitwise_not(flood[1:-1, 1:-1]))
    return _to_binary_mask(filled)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a binary mask."""
    binary = _to_binary_mask(mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    return np.where(labels == largest_label, 255, 0).astype(np.uint8)


def remove_small_components(
    mask: np.ndarray,
    min_area: int = 0,
    min_relative_area: float = 0.0,
) -> np.ndarray:
    """Remove connected components smaller than the chosen area threshold."""
    binary = _to_binary_mask(mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary

    component_areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    largest_area = int(component_areas.max(initial=0))
    threshold = max(int(min_area), int(round(largest_area * float(min_relative_area))))

    cleaned = np.zeros_like(binary)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= threshold:
            cleaned[labels == label] = 255
    return cleaned


def constrain_mask_to_roi(mask: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """Clip a binary mask so pixels survive only inside the ROI."""
    return cv2.bitwise_and(_to_binary_mask(mask), _to_binary_mask(roi_mask))

