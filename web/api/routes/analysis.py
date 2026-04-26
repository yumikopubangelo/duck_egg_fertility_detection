"""Analysis API — feature importance, cluster visualisation, confusion matrix."""

from __future__ import annotations

import traceback

import numpy as np
from flask import Blueprint, current_app, jsonify
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.features.classical_features import ClassicalFeatureExtractor
from src.features.hybrid_features import load_feature_metadata
from src.web.model_manager import get_default_model_manager

analysis_bp = Blueprint("analysis", __name__)

_viz_cache: dict = {}


def _manager():
    return get_default_model_manager(current_app.config)


def _feature_schema() -> tuple[list[str], dict[str, list[int]]]:
    mgr = _manager()
    metadata_path = mgr.train_features_path.parent / "feature_metadata.json"
    metadata = load_feature_metadata(metadata_path)
    if metadata:
        feature_names = list(metadata.get("feature_names", []))
        group_map = {
            str(name): [int(idx) for idx in indices]
            for name, indices in dict(metadata.get("group_map", {})).items()
        }
        if feature_names and group_map:
            return feature_names, group_map

    extractor = ClassicalFeatureExtractor()
    return extractor.feature_names, extractor.group_map


def _feature_importance(model, X: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    n_features = X.shape[1]
    stored = getattr(model, "feature_importance", None)
    feature_indices = getattr(model, "feature_indices_", None)
    if feature_indices is None:
        feature_indices = getattr(model, "feature_indices", None)

    if stored is not None and feature_indices is not None and len(stored) == len(feature_indices):
        imp = np.zeros(n_features, dtype=float)
        imp[np.asarray(feature_indices, dtype=int)] = np.array(stored, dtype=float)
    elif stored is not None and len(stored) == n_features:
        imp = np.array(stored, dtype=float)
    else:
        unique = np.unique(clusters)
        if len(unique) < 2:
            imp = np.ones(n_features, dtype=float)
        else:
            c0 = X[clusters == unique[0]]
            c1 = X[clusters == unique[1]]
            mean_diff = np.abs(c0.mean(axis=0) - c1.mean(axis=0))
            pooled_std = np.sqrt((c0.std(axis=0) ** 2 + c1.std(axis=0) ** 2) / 2 + 1e-8)
            imp = mean_diff / pooled_std

    imp = np.clip(imp, 0, None)
    total = imp.sum()
    return imp / total if total > 0 else np.ones(n_features) / n_features


@analysis_bp.route("/analysis/feature-importance", methods=["GET"])
def feature_importance():
    try:
        mgr = _manager()
        mgr.load()

        feature_names, group_map = _feature_schema()
        n_features = len(feature_names)

        X_train = np.load(mgr.train_features_path)
        clusters = mgr._nearest_clusters(X_train)
        imp = _feature_importance(mgr.model, X_train, clusters)
        if len(imp) != n_features or not group_map:
            imp = np.ones(len(imp)) / max(len(imp), 1)
            feature_names = feature_names or [f"feature_{idx:03d}" for idx in range(len(imp))]
            group_map = {"Semua Fitur": list(range(len(imp)))}

        groups = {name: round(float(imp[idxs].sum()) * 100, 2) for name, idxs in group_map.items()}

        sorted_idx = np.argsort(imp)[::-1]
        top_features = []
        for rank, idx in enumerate(sorted_idx[:15], 1):
            group = next((g for g, idxs in group_map.items() if int(idx) in idxs), "Lainnya")
            top_features.append(
                {
                    "rank": rank,
                    "index": int(idx),
                    "name": feature_names[idx],
                    "importance": round(float(imp[idx]), 6),
                    "importance_pct": round(float(imp[idx]) * 100, 3),
                    "group": group,
                }
            )

        return jsonify(
            {
                "feature_names": feature_names,
                "importance": imp.tolist(),
                "top_features": top_features,
                "groups": groups,
                "total_features": n_features,
            }
        ), 200

    except Exception as exc:
        current_app.logger.exception("feature-importance failed")
        return jsonify({"error": str(exc), "trace": traceback.format_exc()}), 500


@analysis_bp.route("/analysis/cluster-visualization", methods=["GET"])
def cluster_visualization():
    if "data" in _viz_cache:
        return jsonify(_viz_cache["data"]), 200

    try:
        mgr = _manager()
        mgr.load()

        X_train = np.load(mgr.train_features_path).astype(np.float64)
        y_train = np.load(mgr.train_labels_path).astype(int)
        X_model = mgr._model_features(X_train.astype(np.float32))
        X_scaled = mgr.scaler.transform(X_model).astype(np.float64)
        clusters = mgr._nearest_clusters(X_train.astype(np.float32))
        labels = [mgr.cluster_label_map.get(int(c), str(c)) for c in clusters]

        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        var_explained = [round(float(v) * 100, 1) for v in pca.explained_variance_ratio_]

        centroids_scaled = np.asarray(mgr.centroids, dtype=np.float64)
        pca_centroids = pca.transform(centroids_scaled)

        perplexity = min(30, max(5, len(X_scaled) - 1))
        tsne = TSNE(
            n_components=2, random_state=42, perplexity=perplexity, max_iter=1000, init="random"
        )
        X_tsne = tsne.fit_transform(X_scaled)

        def _points(coords):
            return [
                {
                    "x": round(float(coords[i, 0]), 4),
                    "y": round(float(coords[i, 1]), 4),
                    "label": labels[i],
                    "cluster_id": int(clusters[i]),
                    "true_label": "fertile" if int(y_train[i]) == 1 else "infertile",
                }
                for i in range(len(X_train))
            ]

        result = {
            "pca": {
                "points": _points(X_pca),
                "variance_explained": var_explained,
                "centroids": [
                    {
                        "x": round(float(pca_centroids[ci, 0]), 4),
                        "y": round(float(pca_centroids[ci, 1]), 4),
                        "cluster_id": ci,
                        "label": mgr.cluster_label_map.get(ci, str(ci)),
                    }
                    for ci in range(len(pca_centroids))
                ],
            },
            "tsne": {
                "points": _points(X_tsne),
            },
            "n_samples": int(len(X_train)),
            "n_fertile": int((y_train == 1).sum()),
            "n_infertile": int((y_train == 0).sum()),
        }

        _viz_cache["data"] = result
        return jsonify(result), 200

    except Exception as exc:
        current_app.logger.exception("cluster-visualization failed")
        return jsonify({"error": str(exc), "trace": traceback.format_exc()}), 500


@analysis_bp.route("/analysis/confusion-matrix", methods=["GET"])
def confusion_matrix_data():
    try:
        mgr = _manager()
        mgr.load()
        ev = mgr.evaluation

        if not ev or "confusion_matrix" not in ev:
            return jsonify(
                {"error": "Evaluation data not available. Run model evaluation first."}
            ), 404

        cm = ev["confusion_matrix"]
        tn, fp = int(cm[0][0]), int(cm[0][1])
        fn, tp = int(cm[1][0]), int(cm[1][1])

        total = tn + fp + fn + tp
        accuracy = (tn + tp) / total if total > 0 else 0

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = (
            2 * precision * sensitivity / (precision + sensitivity)
            if (precision + sensitivity) > 0
            else 0
        )
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        return jsonify(
            {
                "matrix": {
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                },
                "metrics": {
                    "accuracy": round(accuracy, 4),
                    "sensitivity": round(sensitivity, 4),
                    "specificity": round(specificity, 4),
                    "precision": round(precision, 4),
                    "npv": round(npv, 4),
                    "f1_score": round(f1, 4),
                },
                "test_samples": int(ev.get("test_samples", total)),
            }
        ), 200

    except Exception as exc:
        current_app.logger.exception("confusion-matrix failed")
        return jsonify({"error": str(exc)}), 500


@analysis_bp.route("/analysis/cache/clear", methods=["POST"])
def clear_cache():
    _viz_cache.clear()
    return jsonify({"message": "Cache cleared"}), 200
