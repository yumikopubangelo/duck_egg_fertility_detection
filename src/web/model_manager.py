"""Model loading and prediction utilities for the web application."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Dict, Iterable, Mapping

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


LABEL_NAMES = {
    0: "infertile",
    1: "fertile",
}


@dataclass(frozen=True)
class ClusterStats:
    """Training-derived metadata for one AWC cluster."""

    label: int
    label_name: str
    purity: float
    counts: Dict[int, int]


@dataclass(frozen=True)
class FeaturePrediction:
    """Prediction result for a single feature vector."""

    label: str
    label_id: int
    confidence: float
    cluster_id: int
    cluster_probability: float
    cluster_purity: float
    distances: list[float]
    label_scores: Dict[str, float]


class AWCModelManager:
    """
    Lazy loader for the AWC artifacts used by the web inference path.

    The stored AWC centroids are in StandardScaler space because the training
    implementation standardizes features before fitting. For reliable inference
    we rebuild the same scaler from saved training features, then map clusters
    to fertility labels from the saved training labels.
    """

    def __init__(
        self,
        project_root: str | Path | None = None,
        model_path: str | Path | None = None,
        train_features_path: str | Path | None = None,
        train_labels_path: str | Path | None = None,
        test_features_path: str | Path | None = None,
        test_labels_path: str | Path | None = None,
    ) -> None:
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
        self.model_path = self._resolve_path(model_path, self._default_model_path())
        self.train_features_path = self._resolve_path(
            train_features_path,
            self.project_root / "data" / "features" / "awc_features.npy",
        )
        self.train_labels_path = self._resolve_path(
            train_labels_path,
            self.project_root / "data" / "features" / "awc_labels.npy",
        )
        self.test_features_path = self._resolve_path(
            test_features_path,
            self.project_root / "data" / "features" / "awc_test_features.npy",
        )
        self.test_labels_path = self._resolve_path(
            test_labels_path,
            self.project_root / "data" / "features" / "awc_test_labels.npy",
        )

        self._loaded = False
        self.model = None
        self.scaler: StandardScaler | None = None
        self.centroids: np.ndarray | None = None
        self.feature_indices: np.ndarray | None = None
        self.cluster_stats: Dict[int, ClusterStats] = {}
        self.cluster_label_map: Dict[int, str] = {}
        self.evaluation: Dict[str, object] = {}

    def _resolve_path(self, path: str | Path | None, default: Path) -> Path:
        resolved = Path(path) if path is not None else default
        if not resolved.is_absolute():
            resolved = self.project_root / resolved
        return resolved

    def _default_model_path(self) -> Path:
        primary = self.project_root / "models" / "awc" / "awc_model.pkl"
        if primary.exists():
            return primary
        return self.project_root / "models" / "awc" / "model.pkl"

    def load(self) -> None:
        """Load model artifacts once."""
        if self._loaded:
            return

        self._ensure_exists(self.model_path, "AWC model")
        self._ensure_exists(self.train_features_path, "AWC training features")
        self._ensure_exists(self.train_labels_path, "AWC training labels")

        with self.model_path.open("rb") as model_file:
            self.model = pickle.load(model_file)

        if not hasattr(self.model, "centroids_") or self.model.centroids_ is None:
            raise ValueError(f"AWC model is not fitted: {self.model_path}")

        X_train = np.load(self.train_features_path)
        y_train = np.load(self.train_labels_path)
        if X_train.ndim != 2:
            raise ValueError(f"Training features must be 2D, got shape {X_train.shape}")
        if len(X_train) != len(y_train):
            raise ValueError(
                "Training feature/label length mismatch: "
                f"{len(X_train)} features vs {len(y_train)} labels"
            )

        indices = getattr(self.model, "feature_indices_", None)
        if indices is None:
            indices = getattr(self.model, "feature_indices", None)
        self.feature_indices = None if indices is None else np.asarray(indices, dtype=int)

        X_model = self._model_features(X_train)
        self.centroids = np.asarray(self.model.centroids_, dtype=np.float32)
        if self.centroids.ndim != 2:
            raise ValueError(f"AWC centroids must be 2D, got shape {self.centroids.shape}")
        if X_model.shape[1] != self.centroids.shape[1]:
            raise ValueError(
                "Feature dimension mismatch: "
                f"training has {X_model.shape[1]}, model expects {self.centroids.shape[1]}"
            )

        self.scaler = getattr(self.model, "scaler_", None)
        if self.scaler is None:
            self.scaler = StandardScaler().fit(X_model)
        train_clusters = self._nearest_clusters(X_train)
        self.cluster_stats = self._build_cluster_stats(train_clusters, y_train)
        self.cluster_label_map = {
            cluster_id: stats.label_name for cluster_id, stats in self.cluster_stats.items()
        }
        self._loaded = True
        self.evaluation = self._evaluate_if_available()

    @staticmethod
    def _ensure_exists(path: Path, label: str) -> None:
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{label} not found or empty: {path}")

    def _nearest_clusters(self, features: np.ndarray) -> np.ndarray:
        if self.scaler is None or self.centroids is None:
            raise RuntimeError("AWC artifacts are not loaded")
        model_features = self._model_features(np.asarray(features, dtype=np.float32))
        scaled = self.scaler.transform(model_features)
        distances = np.linalg.norm(scaled[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis=2)
        return np.argmin(distances, axis=1)

    def _model_features(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if self.feature_indices is None:
            return matrix
        if matrix.shape[1] == len(self.feature_indices):
            return matrix
        if int(self.feature_indices.max()) >= matrix.shape[1]:
            raise ValueError(
                f"Feature matrix has {matrix.shape[1]} values, "
                f"but selected feature index {int(self.feature_indices.max())} is required"
            )
        return matrix[:, self.feature_indices]

    def _build_cluster_stats(self, clusters: np.ndarray, labels: np.ndarray) -> Dict[int, ClusterStats]:
        stats: Dict[int, ClusterStats] = {}
        n_clusters = int(getattr(self.model, "n_clusters", self.centroids.shape[0]))

        for cluster_id in range(n_clusters):
            cluster_labels = labels[clusters == cluster_id].astype(int)
            if len(cluster_labels) == 0:
                fallback_label = self._majority_label(labels)
                stats[cluster_id] = ClusterStats(
                    label=fallback_label,
                    label_name=LABEL_NAMES.get(fallback_label, str(fallback_label)),
                    purity=0.0,
                    counts={},
                )
                continue

            unique, counts = np.unique(cluster_labels, return_counts=True)
            majority_idx = int(np.argmax(counts))
            majority_label = int(unique[majority_idx])
            total = int(counts.sum())
            stats[cluster_id] = ClusterStats(
                label=majority_label,
                label_name=LABEL_NAMES.get(majority_label, str(majority_label)),
                purity=float(counts[majority_idx] / total),
                counts={int(label): int(count) for label, count in zip(unique, counts)},
            )

        return stats

    @staticmethod
    def _majority_label(labels: Iterable[int]) -> int:
        labels_array = np.asarray(list(labels), dtype=int)
        unique, counts = np.unique(labels_array, return_counts=True)
        return int(unique[int(np.argmax(counts))])

    def _evaluate_if_available(self) -> Dict[str, object]:
        if not self.test_features_path.exists() or not self.test_labels_path.exists():
            return {}
        if self.test_features_path.stat().st_size == 0 or self.test_labels_path.stat().st_size == 0:
            return {}

        X_test = np.load(self.test_features_path)
        y_test = np.load(self.test_labels_path).astype(int)
        if X_test.ndim != 2 or len(X_test) != len(y_test):
            return {}

        predictions = [self.predict_features(row).label_id for row in X_test]
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        return {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "confusion_matrix": matrix.astype(int).tolist(),
            "test_samples": int(len(y_test)),
        }

    @staticmethod
    def _softmax_negative_distances(distances: np.ndarray) -> np.ndarray:
        neg_distances = -distances
        exp_distances = np.exp(neg_distances - np.max(neg_distances))
        return exp_distances / (np.sum(exp_distances) + 1e-12)

    def predict_features(self, features: np.ndarray) -> FeaturePrediction:
        """Predict a fertility label from one feature vector."""
        self.load()
        if self.scaler is None or self.centroids is None:
            raise RuntimeError("AWC artifacts are not loaded")

        vector = np.asarray(features, dtype=np.float32).reshape(1, -1)
        model_vector = self._model_features(vector)
        if model_vector.shape[1] != self.centroids.shape[1]:
            raise ValueError(
                f"Feature vector has {model_vector.shape[1]} model values, expected {self.centroids.shape[1]}"
            )

        scaled = self.scaler.transform(model_vector)
        distances = np.linalg.norm(scaled - self.centroids, axis=1)
        cluster_probs = self._softmax_negative_distances(distances)
        cluster_id = int(np.argmin(distances))
        stats = self.cluster_stats[cluster_id]

        label_scores = self._label_scores(cluster_probs)
        label_score = label_scores.get(stats.label_name, 0.0)
        confidence = float(np.clip(label_score * stats.purity, 0.0, 1.0))

        return FeaturePrediction(
            label=stats.label_name,
            label_id=stats.label,
            confidence=confidence,
            cluster_id=cluster_id,
            cluster_probability=float(cluster_probs[cluster_id]),
            cluster_purity=stats.purity,
            distances=[float(distance) for distance in distances],
            label_scores=label_scores,
        )

    def _label_scores(self, cluster_probs: np.ndarray) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for cluster_id, probability in enumerate(cluster_probs):
            label = self.cluster_stats[cluster_id].label_name
            scores[label] = scores.get(label, 0.0) + float(probability)
        return scores

    def info(self) -> Dict[str, object]:
        """Return model metadata suitable for the API."""
        self.load()
        return {
            "model_name": "AWC (Adaptive Weighted Clustering)",
            "model_version": "1.0.0",
            "model_path": str(self.model_path),
            "train_features_path": str(self.train_features_path),
            "n_clusters": int(getattr(self.model, "n_clusters", len(self.cluster_stats))),
            "selected_feature_count": (
                int(len(self.feature_indices)) if self.feature_indices is not None else None
            ),
            "selected_feature_indices": (
                self.feature_indices.astype(int).tolist() if self.feature_indices is not None else None
            ),
            "cluster_label_map": self.cluster_label_map,
            "cluster_stats": {
                cluster_id: {
                    "label": stats.label_name,
                    "label_id": stats.label,
                    "purity": round(stats.purity, 4),
                    "counts": stats.counts,
                }
                for cluster_id, stats in self.cluster_stats.items()
            },
            "silhouette_score": self._safe_float(getattr(self.model, "silhouette_", None)),
            "evaluation": self.evaluation,
        }

    @staticmethod
    def _safe_float(value: object) -> float | None:
        if value is None:
            return None
        return float(value)


_default_manager: AWCModelManager | None = None


def get_default_model_manager(config: Mapping[str, object] | None = None) -> AWCModelManager:
    """Return the process-wide model manager."""
    global _default_manager
    if _default_manager is None:
        model_path = None
        if config is not None:
            model_folder = config.get("MODEL_FOLDER")
            if model_folder:
                root = Path(__file__).resolve().parents[2]
                candidate = Path(str(model_folder)) / "awc" / "awc_model.pkl"
                if not candidate.is_absolute():
                    candidate = root / candidate
                model_path = candidate
        _default_manager = AWCModelManager(model_path=model_path)
    return _default_manager
