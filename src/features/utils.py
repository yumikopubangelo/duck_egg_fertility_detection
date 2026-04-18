"""Utility helpers for feature engineering workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


class FeatureUtils:
    """Collection of utility methods for feature matrices."""

    @staticmethod
    def to_2d(features: Iterable[np.ndarray]) -> np.ndarray:
        """Convert iterable of vectors into 2D numpy array."""
        vectors = [np.asarray(v, dtype=np.float32).ravel() for v in features]
        if not vectors:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack(vectors).astype(np.float32)

    @staticmethod
    def standardize(features: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
        """Standardize feature matrix to zero mean and unit variance."""
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        return scaled.astype(np.float32), scaler

    @staticmethod
    def select_by_variance(features: np.ndarray, threshold: float = 0.0) -> Tuple[np.ndarray, VarianceThreshold]:
        """Remove low-variance features."""
        selector = VarianceThreshold(threshold=threshold)
        reduced = selector.fit_transform(features)
        return reduced.astype(np.float32), selector

    @staticmethod
    def save_features(path: str | Path, features: np.ndarray, labels: np.ndarray | None = None) -> None:
        """Save feature matrix (and optional labels) as compressed npz."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if labels is None:
            np.savez_compressed(path, features=features)
        else:
            np.savez_compressed(path, features=features, labels=labels)

    @staticmethod
    def load_features(path: str | Path) -> Tuple[np.ndarray, np.ndarray | None]:
        """Load feature matrix and optional labels from npz."""
        data = np.load(Path(path), allow_pickle=False)
        features = data["features"].astype(np.float32)
        labels = data["labels"] if "labels" in data.files else None
        return features, labels
