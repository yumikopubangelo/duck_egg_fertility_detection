"""Image-to-fertility prediction service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from src.features.classical_features import ClassicalFeatureExtractor
from src.features.hybrid_features import build_default_feature_extractor
from src.preprocessing import DuckEggPreprocessor
from src.web.model_manager import AWCModelManager, FeaturePrediction, get_default_model_manager


@dataclass(frozen=True)
class ImagePrediction:
    """Prediction result for an uploaded image."""

    label: str
    label_id: int
    confidence: float
    cluster_id: int
    cluster_probability: float
    cluster_purity: float
    distances: list[float]
    label_scores: Dict[str, float]
    feature_count: int
    preprocessed_shape: tuple[int, ...]


class PredictionService:
    """Runs the same preprocessing and feature extraction path used for AWC training."""

    def __init__(
        self,
        model_manager: AWCModelManager | None = None,
        preprocessor: DuckEggPreprocessor | None = None,
        extractor=None,
    ) -> None:
        self.model_manager = model_manager or get_default_model_manager()
        self.preprocessor = preprocessor or DuckEggPreprocessor()
        self.extractor = extractor or self._default_extractor()

    def _default_extractor(self):
        metadata_path = self.model_manager.train_features_path.parent / "feature_metadata.json"
        if metadata_path.exists():
            return build_default_feature_extractor(metadata_path, preprocess_override=False)
        return ClassicalFeatureExtractor()

    def predict_file(self, image_path: str | Path) -> ImagePrediction:
        path = Path(image_path)
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        return self.predict_image(image)

    def predict_image(self, image: np.ndarray) -> ImagePrediction:
        preprocessed = self.preprocessor.preprocess(image)
        features = self.extractor.extract(preprocessed)
        feature_prediction = self.model_manager.predict_features(features)
        return self._build_image_prediction(feature_prediction, features, preprocessed)

    @staticmethod
    def _build_image_prediction(
        prediction: FeaturePrediction,
        features: np.ndarray,
        preprocessed: np.ndarray,
    ) -> ImagePrediction:
        return ImagePrediction(
            label=prediction.label,
            label_id=prediction.label_id,
            confidence=prediction.confidence,
            cluster_id=prediction.cluster_id,
            cluster_probability=prediction.cluster_probability,
            cluster_purity=prediction.cluster_purity,
            distances=prediction.distances,
            label_scores=prediction.label_scores,
            feature_count=int(features.shape[0]),
            preprocessed_shape=tuple(int(value) for value in preprocessed.shape),
        )

    def model_info(self) -> Dict[str, object]:
        return self.model_manager.info()


_default_service: PredictionService | None = None


def get_default_prediction_service() -> PredictionService:
    """Return the process-wide prediction service."""
    global _default_service
    if _default_service is None:
        _default_service = PredictionService()
    return _default_service
