"""Image-to-fertility prediction service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from src.classification.mobilenet_v3_baseline import MobileNetV3Baseline
from src.clustering.fuzzy_cmeans import FuzzyCMeans
from src.clustering.kmeans_baseline import KMeansBaseline
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
        self._kmeans_model = None
        self._fcm_model = None
        self._mobilenet_model = None

    def _default_extractor(self):
        metadata_path = self.model_manager.train_features_path.parent / "feature_metadata.json"
        if metadata_path.exists():
            return build_default_feature_extractor(metadata_path, preprocess_override=False)
        return ClassicalFeatureExtractor()

    def _prepare_prediction(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, FeaturePrediction]:
        preprocessed = self.preprocessor.preprocess(image)
        features = self.extractor.extract(preprocessed)
        feature_prediction = self.model_manager.predict_features(features)
        return preprocessed, features, feature_prediction

    def predict_file(self, image_path: str | Path) -> ImagePrediction:
        path = Path(image_path)
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        return self.predict_image(image)

    def predict_file_with_context(self, image_path: str | Path) -> tuple[ImagePrediction, np.ndarray, np.ndarray]:
        path = Path(image_path)
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        return self.predict_image_with_context(image)

    def predict_image(self, image: np.ndarray) -> ImagePrediction:
        preprocessed, features, feature_prediction = self._prepare_prediction(image)
        return self._build_image_prediction(feature_prediction, features, preprocessed)

    def predict_image_with_context(self, image: np.ndarray) -> tuple[ImagePrediction, np.ndarray, np.ndarray]:
        preprocessed, features, feature_prediction = self._prepare_prediction(image)
        prediction = self._build_image_prediction(feature_prediction, features, preprocessed)
        return prediction, preprocessed, features

    def explain_prediction(self, features: np.ndarray, prediction: ImagePrediction) -> dict[str, object]:
        self.model_manager.load()

        group_map = getattr(self.extractor, "group_map", {}) or {}
        feature_names = list(getattr(self.extractor, "feature_names", []))
        importances = np.zeros(len(features), dtype=np.float32)
        model = self.model_manager.model
        selected = getattr(model, "feature_indices_", None)
        stored = getattr(model, "feature_importance", None)
        if selected is None:
            selected = getattr(model, "feature_indices", None)

        if stored is not None:
            if selected is not None and len(stored) == len(selected):
                importances[np.asarray(selected, dtype=int)] = np.asarray(stored, dtype=np.float32)
            elif len(stored) == len(importances):
                importances = np.asarray(stored, dtype=np.float32)

        top_groups = []
        if group_map:
            for name, idxs in group_map.items():
                score = float(importances[idxs].sum()) if len(importances) == len(features) else 0.0
                magnitude = float(np.mean(np.abs(features[idxs]))) if idxs else 0.0
                top_groups.append(
                    {
                        "name": name,
                        "importance": round(score, 6),
                        "mean_activation": round(magnitude, 4),
                    }
                )
            top_groups.sort(key=lambda item: (item["importance"], item["mean_activation"]), reverse=True)

        reasons = [
            (
                f"AWC memilih label {prediction.label} karena skor label "
                f"{prediction.label_scores.get(prediction.label, 0.0):.2f} "
                f"dengan purity cluster {prediction.cluster_purity:.2f}."
            )
        ]
        if top_groups:
            reasons.append(
                "Kelompok fitur yang paling dominan: "
                + ", ".join(group["name"] for group in top_groups[:3])
                + "."
            )

        top_features = []
        if feature_names and len(importances) == len(feature_names):
            ranked = np.argsort(importances)[::-1][:5]
            for idx in ranked:
                top_features.append(
                    {
                        "name": feature_names[int(idx)],
                        "index": int(idx),
                        "importance": round(float(importances[int(idx)]), 6),
                        "value": round(float(features[int(idx)]), 4),
                    }
                )

        return {
            "summary": reasons[0],
            "reasons": reasons,
            "top_groups": top_groups[:5],
            "top_features": top_features,
            "selected_feature_count": (
                int(len(selected)) if selected is not None else int(len(features))
            ),
        }

    def _load_kmeans_model(self):
        if self._kmeans_model is None:
            path = Path(__file__).resolve().parents[2] / "models" / "baselines" / "kmeans_model.pkl"
            if path.exists() and path.stat().st_size > 0:
                self._kmeans_model = KMeansBaseline.load(path)
        return self._kmeans_model

    def _load_fcm_model(self):
        if self._fcm_model is None:
            path = Path(__file__).resolve().parents[2] / "models" / "baselines" / "fcm_model.pkl"
            if path.exists() and path.stat().st_size > 0:
                self._fcm_model = FuzzyCMeans.load(path)
        return self._fcm_model

    def _load_mobilenet_model(self):
        if self._mobilenet_model is None:
            path = Path(__file__).resolve().parents[2] / "models" / "baselines" / "mobilenetv3_small.pth"
            if path.exists() and path.stat().st_size > 0:
                self._mobilenet_model = MobileNetV3Baseline.load(path)
        return self._mobilenet_model

    def compare_models(self, image: np.ndarray, features: np.ndarray) -> dict[str, dict[str, object]]:
        comparisons: dict[str, dict[str, object]] = {}

        kmeans = self._load_kmeans_model()
        if kmeans is not None:
            proba = kmeans.predict_proba(features.reshape(1, -1))[0]
            label = int(np.argmax(proba))
            comparisons["KMeans"] = {
                "label": "fertile" if label == 1 else "infertile",
                "label_id": label,
                "confidence": round(float(proba[label]), 4),
            }

        fcm = self._load_fcm_model()
        if fcm is not None:
            proba = fcm.predict_proba(features.reshape(1, -1))[0]
            label = int(np.argmax(proba))
            comparisons["FCM"] = {
                "label": "fertile" if label == 1 else "infertile",
                "label_id": label,
                "confidence": round(float(proba[label]), 4),
            }

        try:
            mobilenet = self._load_mobilenet_model()
        except Exception:
            mobilenet = None
        if mobilenet is not None:
            label, fertile_prob = mobilenet.predict_image(image)
            comparisons["MobileNetV3"] = {
                "label": "fertile" if label == 1 else "infertile",
                "label_id": int(label),
                "confidence": round(float(max(fertile_prob, 1.0 - fertile_prob)), 4),
                "fertile_probability": round(float(fertile_prob), 4),
            }

        return comparisons

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
