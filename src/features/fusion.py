"""Feature fusion helpers for combining classical and deep features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureFusionConfig:
    """Configuration for feature fusion pipeline."""

    standardize: bool = True
    pca_components: Optional[int] = None
    classical_weight: float = 1.0
    deep_weight: float = 1.0


class FeatureFusion:
    """Fuse two feature spaces into a single training-ready matrix."""

    def __init__(self, config: FeatureFusionConfig | None = None):
        self.config = config or FeatureFusionConfig()
        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[PCA] = None

    def _weighted_concat(self, classical_features: np.ndarray, deep_features: np.ndarray) -> np.ndarray:
        if classical_features.ndim != 2 or deep_features.ndim != 2:
            raise ValueError("Input features harus 2D: (n_samples, n_features)")
        if classical_features.shape[0] != deep_features.shape[0]:
            raise ValueError("Jumlah sample classical dan deep features harus sama")

        classical = classical_features * float(self.config.classical_weight)
        deep = deep_features * float(self.config.deep_weight)
        return np.concatenate([classical, deep], axis=1).astype(np.float32)

    def fit_transform(self, classical_features: np.ndarray, deep_features: np.ndarray) -> np.ndarray:
        """
        Fit scaler/PCA and transform fused features.
        """
        fused = self._weighted_concat(classical_features, deep_features)

        if self.config.standardize:
            self._scaler = StandardScaler()
            fused = self._scaler.fit_transform(fused)

        if self.config.pca_components is not None:
            self._pca = PCA(n_components=self.config.pca_components, random_state=42)
            fused = self._pca.fit_transform(fused)

        return fused.astype(np.float32)

    def transform(self, classical_features: np.ndarray, deep_features: np.ndarray) -> np.ndarray:
        """
        Transform fused features with previously fitted scaler/PCA.
        """
        fused = self._weighted_concat(classical_features, deep_features)

        if self._scaler is not None:
            fused = self._scaler.transform(fused)

        if self._pca is not None:
            fused = self._pca.transform(fused)

        return fused.astype(np.float32)
