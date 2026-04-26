"""Hybrid feature extraction and metadata helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np

from src.features.classical_features import ClassicalFeatureConfig, ClassicalFeatureExtractor
from src.features.deep_features import DeepFeatureExtractor
from src.preprocessing import DuckEggPreprocessor


@dataclass
class HybridFeatureConfig:
    """Configuration for the hybrid feature pipeline."""

    include_classical: bool = True
    include_glcm: bool = True
    include_morphology: bool = True
    include_deep: bool = True
    preprocess_input: bool = False
    deep_backbone: str = "unet_bottleneck"
    deep_pretrained: bool = False
    deep_image_size: int = 256
    unet_checkpoint: str = ""
    unet_lightweight: bool = True
    unet_n_channels: int = 3
    unet_n_classes: int = 3
    unet_bilinear: bool = True
    unet_dropout_rate: float = 0.3
    glcm_levels: int = 16


class HybridFeatureExtractor:
    """Extract richer hybrid features aligned with the research claim."""

    _GLCM_OFFSETS = ((0, 1), (1, 0), (1, 1), (-1, 1))
    _GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")
    _MORPH_FEATURES = (
        "mask_area_ratio",
        "mask_perimeter_ratio",
        "mask_compactness",
        "mask_extent",
        "mask_solidity",
        "mask_aspect_ratio",
        "mask_eccentricity",
        "mask_centroid_x",
        "mask_centroid_y",
    )

    def __init__(self, config: HybridFeatureConfig | None = None):
        self.config = config or HybridFeatureConfig()
        self.classical_extractor = ClassicalFeatureExtractor(
            ClassicalFeatureConfig(include_glcm=False)
        )
        self.preprocessor = DuckEggPreprocessor() if self.config.preprocess_input else None
        self.deep_extractor = None
        if self.config.include_deep:
            checkpoint_path = self.config.unet_checkpoint or None
            self.deep_extractor = DeepFeatureExtractor(
                backbone=self.config.deep_backbone,
                pretrained=self.config.deep_pretrained,
                image_size=self.config.deep_image_size,
                checkpoint_path=checkpoint_path,
                lightweight=self.config.unet_lightweight,
                n_channels=self.config.unet_n_channels,
                n_classes=self.config.unet_n_classes,
                bilinear=self.config.unet_bilinear,
                dropout_rate=self.config.unet_dropout_rate,
            )
        self._feature_names = self._build_feature_names()
        self._group_map = self._build_group_map()

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    @property
    def group_map(self) -> Dict[str, List[int]]:
        return self._group_map

    def _build_feature_names(self) -> List[str]:
        names: List[str] = []
        if self.config.include_classical:
            names.extend(self.classical_extractor.feature_names)
        if self.config.include_glcm:
            for direction_idx, _ in enumerate(self._GLCM_OFFSETS):
                for prop in self._GLCM_PROPS:
                    names.append(f"glcm_{prop}_dir{direction_idx}")
        if self.config.include_morphology:
            names.extend(self._MORPH_FEATURES)
        if self.config.include_deep and self.deep_extractor is not None:
            names.extend([f"deep_{idx:03d}" for idx in range(self.deep_extractor.feature_dim)])
        return names

    def _build_group_map(self) -> Dict[str, List[int]]:
        groups: Dict[str, List[int]] = {}
        start = 0

        if self.config.include_classical:
            for name, idxs in self.classical_extractor.group_map.items():
                groups[name] = list(range(start, start + len(idxs)))
                start += len(idxs)

        if self.config.include_glcm:
            glcm_len = len(self._GLCM_OFFSETS) * len(self._GLCM_PROPS)
            groups["Tekstur GLCM"] = list(range(start, start + glcm_len))
            start += glcm_len

        if self.config.include_morphology:
            morph_len = len(self._MORPH_FEATURES)
            groups["Morfologi Mask"] = list(range(start, start + morph_len))
            start += morph_len

        if self.config.include_deep and self.deep_extractor is not None:
            deep_len = self.deep_extractor.feature_dim
            groups["Deep Bottleneck"] = list(range(start, start + deep_len))

        return groups

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image.astype(np.uint8, copy=False)
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        raise ValueError(f"Unsupported image shape: {image.shape}")

    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if num_labels <= 1:
            return mask.astype(np.uint8)
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return (labels == largest_label).astype(np.uint8)

    def _resolve_mask(self, gray: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        if mask is not None and np.any(mask):
            resolved = (mask > 0).astype(np.uint8)
            if resolved.shape != gray.shape:
                resolved = cv2.resize(resolved, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
            return self._largest_component(resolved)

        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = (otsu > 0).astype(np.uint8)
        area_ratio = float(binary.mean())
        if area_ratio < 0.01 or area_ratio > 0.90:
            binary = 1 - binary
        return self._largest_component(binary)

    def _extract_glcm_features(self, gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
        levels = int(self.config.glcm_levels)
        quantized = np.floor(gray.astype(np.float32) / (256.0 / levels)).astype(np.int32)
        quantized = np.clip(quantized, 0, levels - 1)
        ii, jj = np.indices((levels, levels))
        outputs: List[float] = []

        for dy, dx in self._GLCM_OFFSETS:
            y0 = max(0, -dy)
            y1 = min(gray.shape[0], gray.shape[0] - dy)
            x0 = max(0, -dx)
            x1 = min(gray.shape[1], gray.shape[1] - dx)

            src = quantized[y0:y1, x0:x1]
            dst = quantized[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
            valid = mask[y0:y1, x0:x1] & mask[y0 + dy : y1 + dy, x0 + dx : x1 + dx]

            src_vals = src[valid > 0]
            dst_vals = dst[valid > 0]
            matrix = np.zeros((levels, levels), dtype=np.float64)
            if src_vals.size > 0:
                np.add.at(matrix, (src_vals, dst_vals), 1)
                np.add.at(matrix, (dst_vals, src_vals), 1)
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

    def _extract_morphology(self, mask: np.ndarray) -> np.ndarray:
        mask_u8 = (mask > 0).astype(np.uint8)
        h, w = mask_u8.shape[:2]
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros(len(self._MORPH_FEATURES), dtype=np.float32)

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        _, _, bw, bh = cv2.boundingRect(contour)
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0
        moments = cv2.moments(contour)
        centroid_x = float(moments["m10"] / moments["m00"] / max(w, 1)) if moments["m00"] else 0.0
        centroid_y = float(moments["m01"] / moments["m00"] / max(h, 1)) if moments["m00"] else 0.0

        compactness = float((4.0 * np.pi * area) / (perimeter ** 2 + 1e-8)) if perimeter > 0 else 0.0
        extent = float(area / (bw * bh + 1e-8)) if bw > 0 and bh > 0 else 0.0
        solidity = float(area / (hull_area + 1e-8)) if hull_area > 0 else 0.0
        aspect_ratio = float(bw / max(bh, 1))
        eccentricity = 0.0
        if len(contour) >= 5:
            (_, _), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
            major_axis = max(float(axis_a), float(axis_b))
            minor_axis = min(float(axis_a), float(axis_b))
            if major_axis > 0:
                eccentricity = float(np.sqrt(max(0.0, 1.0 - (minor_axis ** 2 / (major_axis ** 2 + 1e-8)))))

        values = np.asarray(
            [
                area / (h * w + 1e-8),
                perimeter / (2 * (h + w) + 1e-8),
                compactness,
                extent,
                solidity,
                aspect_ratio,
                eccentricity,
                centroid_x,
                centroid_y,
            ],
            dtype=np.float32,
        )
        return values

    def extract(self, image: np.ndarray) -> np.ndarray:
        working = self.preprocessor.preprocess(image) if self.preprocessor is not None else image
        gray = self._to_gray(working)

        deep_features = np.empty((0,), dtype=np.float32)
        mask = None
        if self.deep_extractor is not None:
            deep_features, mask = self.deep_extractor.extract_with_mask(working)

        roi_mask = self._resolve_mask(gray, mask)

        parts: List[np.ndarray] = []
        if self.config.include_classical:
            parts.append(self.classical_extractor.extract(working))
        if self.config.include_glcm:
            parts.append(self._extract_glcm_features(gray, roi_mask))
        if self.config.include_morphology:
            parts.append(self._extract_morphology(roi_mask))
        if self.config.include_deep:
            parts.append(np.asarray(deep_features, dtype=np.float32))

        return np.concatenate(parts, axis=0).astype(np.float32)

    def extract_batch(self, images: Iterable[np.ndarray]) -> np.ndarray:
        vectors = [self.extract(image) for image in images]
        if not vectors:
            return np.empty((0, len(self.feature_names)), dtype=np.float32)
        return np.vstack(vectors).astype(np.float32)

    def metadata(self) -> Dict[str, object]:
        return {
            "mode": "hybrid",
            "feature_names": self.feature_names,
            "group_map": self.group_map,
            "feature_count": len(self.feature_names),
            "config": asdict(self.config),
        }


def default_feature_metadata_path(root: str | Path | None = None) -> Path:
    base = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    return base / "data" / "features" / "feature_metadata.json"


def save_feature_metadata(path: str | Path, metadata: Dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_feature_metadata(path: str | Path | None = None) -> Dict[str, object] | None:
    metadata_path = Path(path) if path is not None else default_feature_metadata_path()
    if not metadata_path.exists() or metadata_path.stat().st_size == 0:
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def build_extractor_from_metadata(
    metadata: Dict[str, object] | None,
    preprocess_override: bool | None = None,
):
    if not metadata:
        return ClassicalFeatureExtractor()

    mode = str(metadata.get("mode", "classical")).lower()
    if mode == "hybrid":
        raw_config = dict(metadata.get("config", {}))
        if preprocess_override is not None:
            raw_config["preprocess_input"] = preprocess_override
        config = HybridFeatureConfig(**raw_config)
        return HybridFeatureExtractor(config)

    raw_config = dict(metadata.get("config", {}))
    return ClassicalFeatureExtractor(ClassicalFeatureConfig(**raw_config))


def build_default_feature_extractor(
    metadata_path: str | Path | None = None,
    preprocess_override: bool | None = None,
):
    metadata = load_feature_metadata(metadata_path)
    return build_extractor_from_metadata(metadata, preprocess_override=preprocess_override)


def build_classical_metadata() -> Dict[str, object]:
    extractor = ClassicalFeatureExtractor()
    return {
        "mode": "classical",
        "feature_names": extractor.feature_names,
        "group_map": extractor.group_map,
        "feature_count": len(extractor.feature_names),
        "config": {
            "histogram_bins": extractor.config.histogram_bins,
            "lbp_radius": extractor.config.lbp_radius,
            "lbp_points": extractor.config.lbp_points,
            "include_glcm": extractor.config.include_glcm,
            "glcm_levels": extractor.config.glcm_levels,
            "include_edge_stats": extractor.config.include_edge_stats,
        },
    }
