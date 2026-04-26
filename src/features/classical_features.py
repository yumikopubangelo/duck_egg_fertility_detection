"""Classical feature extraction for duck egg fertility images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import cv2
import numpy as np

GLCM_OFFSETS = ((0, 1), (1, 0), (1, 1), (-1, 1))
GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")


@dataclass
class ClassicalFeatureConfig:
    """Configuration for classical feature extraction."""

    histogram_bins: int = 32
    lbp_radius: int = 1
    lbp_points: int = 8
    include_glcm: bool = True
    glcm_levels: int = 16
    include_edge_stats: bool = True


class ClassicalFeatureExtractor:
    """
    Extract handcrafted features from grayscale egg images.

    Features:
    - Intensity statistics (mean/std/min/max/median)
    - Normalized histogram
    - LBP histogram
    - GLCM spatial texture descriptors
    - Edge statistics (optional)
    """

    def __init__(self, config: ClassicalFeatureConfig | None = None):
        self.config = config or ClassicalFeatureConfig()
        self._feature_names = self._build_feature_names()

    def _build_feature_names(self) -> List[str]:
        names = ["mean", "std", "min", "max", "median"]
        names.extend([f"hist_{i}" for i in range(self.config.histogram_bins)])
        names.extend([f"lbp_{i}" for i in range(self.config.lbp_points + 2)])
        if self.config.include_glcm:
            for direction_idx, _ in enumerate(GLCM_OFFSETS):
                for prop in GLCM_PROPS:
                    names.append(f"glcm_{prop}_dir{direction_idx}")
        if self.config.include_edge_stats:
            names.extend(["edge_density", "edge_mean", "edge_std"])
        return names

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    @property
    def group_map(self) -> Dict[str, List[int]]:
        """Return contiguous index ranges grouped by feature family."""
        groups: Dict[str, List[int]] = {}
        start = 0

        groups["Statistik Intensitas"] = list(range(start, start + 5))
        start += 5

        groups["Histogram"] = list(range(start, start + self.config.histogram_bins))
        start += self.config.histogram_bins

        lbp_bins = self.config.lbp_points + 2
        groups["Tekstur LBP"] = list(range(start, start + lbp_bins))
        start += lbp_bins

        if self.config.include_glcm:
            glcm_len = len(GLCM_OFFSETS) * len(GLCM_PROPS)
            groups["Tekstur GLCM"] = list(range(start, start + glcm_len))
            start += glcm_len

        if self.config.include_edge_stats:
            groups["Tepi (Edge)"] = list(range(start, start + 3))

        return groups

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image.astype(np.uint8, copy=False)
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        raise ValueError(f"Unsupported image shape: {image.shape}")

    def _extract_lbp(self, gray: np.ndarray) -> np.ndarray:
        """
        Compute a simple uniform-like LBP histogram using 8-neighborhood.
        """
        gray = gray.astype(np.int16, copy=False)
        center = gray
        neighbors = [
            np.roll(np.roll(gray, -1, axis=0), -1, axis=1),
            np.roll(gray, -1, axis=0),
            np.roll(np.roll(gray, -1, axis=0), 1, axis=1),
            np.roll(gray, 1, axis=1),
            np.roll(np.roll(gray, 1, axis=0), 1, axis=1),
            np.roll(gray, 1, axis=0),
            np.roll(np.roll(gray, 1, axis=0), -1, axis=1),
            np.roll(gray, -1, axis=1),
        ]

        lbp = np.zeros_like(center, dtype=np.uint8)
        for bit, n in enumerate(neighbors):
            lbp |= ((n >= center).astype(np.uint8) << bit)

        # Uniform pattern approximation: map to number of set bits (0..8), others -> 9
        bit_count = np.unpackbits(lbp[..., None], axis=2).sum(axis=2)
        transitions = np.zeros_like(lbp, dtype=np.uint8)
        for i in range(8):
            b1 = (lbp >> i) & 1
            b2 = (lbp >> ((i + 1) % 8)) & 1
            transitions += (b1 != b2).astype(np.uint8)
        uniform = transitions <= 2
        mapped = np.where(uniform, bit_count, 9).astype(np.uint8)

        hist = cv2.calcHist([mapped], [0], None, [10], [0, 10]).flatten()
        hist /= hist.sum() + 1e-8
        # Keep first lbp_points+2 bins to match naming.
        return hist[: self.config.lbp_points + 2]

    def _extract_glcm(self, gray: np.ndarray) -> np.ndarray:
        levels = int(self.config.glcm_levels)
        quantized = np.floor(gray.astype(np.float32) / (256.0 / levels)).astype(np.int32)
        quantized = np.clip(quantized, 0, levels - 1)
        ii, jj = np.indices((levels, levels))
        outputs: List[float] = []

        for dy, dx in GLCM_OFFSETS:
            y0 = max(0, -dy)
            y1 = min(gray.shape[0], gray.shape[0] - dy)
            x0 = max(0, -dx)
            x1 = min(gray.shape[1], gray.shape[1] - dx)

            src = quantized[y0:y1, x0:x1]
            dst = quantized[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
            matrix = np.zeros((levels, levels), dtype=np.float64)
            if src.size > 0:
                np.add.at(matrix, (src.ravel(), dst.ravel()), 1)
                np.add.at(matrix, (dst.ravel(), src.ravel()), 1)
            matrix /= matrix.sum() + 1e-8

            contrast = float(np.sum(((ii - jj) ** 2) * matrix))
            dissimilarity = float(np.sum(np.abs(ii - jj) * matrix))
            homogeneity = float(np.sum(matrix / (1.0 + (ii - jj) ** 2)))
            energy = float(np.sqrt(np.sum(matrix ** 2)))

            mean_i = float(np.sum(ii * matrix))
            mean_j = float(np.sum(jj * matrix))
            std_i = float(np.sqrt(np.sum(((ii - mean_i) ** 2) * matrix)))
            std_j = float(np.sqrt(np.sum(((jj - mean_j) ** 2) * matrix)))
            if std_i > 0 and std_j > 0:
                correlation = float(np.sum(((ii - mean_i) * (jj - mean_j) * matrix)) / (std_i * std_j))
            else:
                correlation = 0.0

            outputs.extend([contrast, dissimilarity, homogeneity, energy, correlation])

        return np.asarray(outputs, dtype=np.float32)

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract 1D feature vector from a single image."""
        gray = self._to_gray(image)

        stats = np.array(
            [
                float(gray.mean()),
                float(gray.std()),
                float(gray.min()),
                float(gray.max()),
                float(np.median(gray)),
            ],
            dtype=np.float32,
        )

        hist = cv2.calcHist([gray], [0], None, [self.config.histogram_bins], [0, 256]).flatten()
        hist = hist.astype(np.float32)
        hist /= hist.sum() + 1e-8

        lbp_hist = self._extract_lbp(gray).astype(np.float32)

        features = [stats, hist, lbp_hist]

        if self.config.include_glcm:
            features.append(self._extract_glcm(gray))

        if self.config.include_edge_stats:
            edges = cv2.Canny(gray, 50, 150)
            edge_values = edges.astype(np.float32) / 255.0
            edge_stats = np.array(
                [edge_values.mean(), edges.mean(), edges.std()],
                dtype=np.float32,
            )
            features.append(edge_stats)

        return np.concatenate(features, axis=0).astype(np.float32)

    def extract_batch(self, images: Iterable[np.ndarray]) -> np.ndarray:
        """Extract feature matrix with shape (n_samples, n_features)."""
        vectors = [self.extract(img) for img in images]
        if not vectors:
            return np.empty((0, len(self.feature_names)), dtype=np.float32)
        return np.vstack(vectors).astype(np.float32)
