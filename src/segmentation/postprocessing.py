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


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """Reduce a binary mask to a one-pixel-wide skeleton."""
    work = _to_binary_mask(mask)
    skeleton = np.zeros_like(work, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while cv2.countNonZero(work) > 0:
        eroded = cv2.erode(work, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(work, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        work = eroded

    return skeleton


def _neighbor_count(binary_mask: np.ndarray) -> np.ndarray:
    """Count 8-neighborhood pixels for every active skeleton pixel."""
    skeleton = (_to_binary_mask(binary_mask) > 0).astype(np.uint8)
    neighbor_map = np.zeros_like(skeleton, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(np.roll(skeleton, dy, axis=0), dx, axis=1)
            if dy == -1:
                shifted[-1, :] = 0
            elif dy == 1:
                shifted[0, :] = 0
            if dx == -1:
                shifted[:, -1] = 0
            elif dx == 1:
                shifted[:, 0] = 0
            neighbor_map = neighbor_map + shifted.astype(np.uint8)
    return neighbor_map


def vascular_morphology_metrics(mask: np.ndarray) -> dict[str, float]:
    """Compute vascular morphology metrics from a binary mask."""
    binary = (_to_binary_mask(mask) > 0).astype(np.uint8)
    area_px = int(binary.sum())
    total_px = int(binary.size)
    if area_px == 0 or total_px == 0:
        return {
            "vascular_area_px": 0.0,
            "vascular_area_pct": 0.0,
            "skeleton_length_px": 0.0,
            "skeleton_length_norm": 0.0,
            "node_count": 0.0,
            "branch_density": 0.0,
        }

    skeleton = (skeletonize_mask(binary) > 0).astype(np.uint8)
    skeleton_length = float(skeleton.sum())
    neighbor_map = _neighbor_count(skeleton)
    node_count = float(np.logical_and(skeleton > 0, neighbor_map >= 3).sum())
    branch_density = node_count / (skeleton_length + 1e-8)
    diagonal = float(np.hypot(*binary.shape[:2]))

    return {
        "vascular_area_px": float(area_px),
        "vascular_area_pct": float(area_px / total_px * 100.0),
        "skeleton_length_px": skeleton_length,
        "skeleton_length_norm": float(skeleton_length / max(diagonal, 1.0)),
        "node_count": node_count,
        "branch_density": float(branch_density),
    }


def postprocess_multiclass_mask(mask: np.ndarray) -> np.ndarray:
    """
    Clean a 3-class segmentation mask.

    Class convention:
    0 = background
    1 = vascularization
    2 = embryo
    """
    raw = np.asarray(mask, dtype=np.uint8)
    if raw.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {raw.shape}")

    foreground = keep_largest_component(raw > 0)
    embryo = fill_mask_holes(raw == 2)
    embryo = keep_largest_component(remove_small_components(embryo, min_area=24, min_relative_area=0.08))
    embryo = constrain_mask_to_roi(embryo, foreground)

    vascular = remove_small_components(raw == 1, min_area=6, min_relative_area=0.01)
    vascular = cv2.morphologyEx(vascular, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    vascular = constrain_mask_to_roi(vascular, foreground)

    cleaned = np.zeros_like(raw, dtype=np.uint8)
    cleaned[vascular > 0] = 1
    cleaned[embryo > 0] = 2
    return cleaned
