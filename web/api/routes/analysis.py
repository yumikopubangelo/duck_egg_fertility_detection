"""Analysis API — feature importance, cluster visualisation, confusion matrix."""

from __future__ import annotations

import base64
import io
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from flask import Blueprint, current_app, jsonify
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import transforms

from src.features.classical_features import ClassicalFeatureExtractor
from src.features.hybrid_features import load_feature_metadata
from src.segmentation.data_loader import EggDataset
from src.segmentation.evaluation import (
    DEFAULT_CLASS_NAMES,
    colorize_mask,
    confusion_matrix_for_masks,
    denormalize_image_tensor,
    metrics_from_confusion_matrix,
    overlay_mask,
    per_sample_segmentation_metrics,
    predict_class_map,
)
from src.segmentation.unet import create_unet_for_eggs
from src.web.model_manager import get_default_model_manager

analysis_bp = Blueprint("analysis", __name__)

_viz_cache: dict = {}


def _manager():
    return get_default_model_manager(current_app.config)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_unet_config() -> dict:
    config_path = _project_root() / "configs" / "unet_config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_unet_model_path(config: dict) -> Path | None:
    root = _project_root()
    candidates = [
        Path(current_app.config.get("MODEL_FOLDER", "models")) / "unet" / "model.pth",
        root / "models" / "unet" / "model.pth",
        Path(current_app.config.get("MODEL_FOLDER", "models")) / "unet" / "unet_best.pth",
        root / "models" / "unet" / "unet_best.pth",
    ]

    training_cfg = dict(config.get("training", {}))
    output_dir = training_cfg.get("output_dir")
    if output_dir:
        ckpt_dir = (root / output_dir / "checkpoints").resolve()
        best_ckpts = sorted(
            ckpt_dir.glob("best_epoch_*.pth"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        candidates.extend(best_ckpts[:3])

    for path in candidates:
        resolved = path if path.is_absolute() else (root / path)
        if resolved.exists() and resolved.stat().st_size > 0:
            return resolved
    return None


def _load_unet_for_report() -> tuple[torch.nn.Module, dict, Path]:
    cfg = _resolve_unet_config()
    model_cfg = dict(cfg.get("model", {}))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = _resolve_unet_model_path(cfg)
    if model_path is None:
        raise FileNotFoundError("Checkpoint U-Net tidak ditemukan.")

    checkpoint = torch.load(str(model_path), map_location=device)
    state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    ckpt_model_cfg = dict(checkpoint.get("config", {}).get("model", {}))
    model_cfg = {**model_cfg, **ckpt_model_cfg}

    model = create_unet_for_eggs(
        n_channels=model_cfg.get("n_channels", 3),
        n_classes=model_cfg.get("n_classes", 3),
        bilinear=model_cfg.get("bilinear", True),
        dropout_rate=model_cfg.get("dropout_rate", 0.3),
        lightweight=model_cfg.get("lightweight", True),
    )
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    model._report_device = device
    model._n_classes = model_cfg.get("n_classes", 3)
    return model, cfg, model_path


def _png_data_url(image_rgb: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _round_float(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


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


def _build_segmentation_report() -> dict:
    if "segmentation_report" in _viz_cache:
        return _viz_cache["segmentation_report"]

    model, cfg, model_path = _load_unet_for_report()
    data_cfg = dict(cfg.get("data", {}))
    eval_cfg = dict(cfg.get("evaluation", {}))
    image_size = tuple(data_cfg.get("image_size", [256, 256]))
    test_image_dir = data_cfg.get("test_image_dir", "data/segmentation/test/images")
    test_mask_dir = data_cfg.get("test_mask_dir", "data/segmentation/test/masks")
    class_names = DEFAULT_CLASS_NAMES[: getattr(model, "_n_classes", 3)]

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = EggDataset(
        image_size=image_size,
        transform=transform,
        image_dir=test_image_dir,
        mask_dir=test_mask_dir,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=min(int(eval_cfg.get("batch_size", 4)), max(len(dataset), 1)),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    total_cm = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    sample_rows = []
    sample_index = 0
    device = getattr(model, "_report_device", torch.device("cpu"))

    with torch.no_grad():
        for images, targets in dataloader:
            outputs = model(images.to(device))
            preds = predict_class_map(outputs).cpu().numpy()
            targets_np = targets.numpy()
            images_cpu = images.cpu()

            for i in range(len(images_cpu)):
                pred_mask = preds[i].astype(np.uint8)
                true_mask = targets_np[i].astype(np.uint8)
                total_cm += confusion_matrix_for_masks(pred_mask, true_mask, len(class_names))

                sample_metrics = per_sample_segmentation_metrics(
                    pred_mask,
                    true_mask,
                    class_names=class_names,
                    background_index=0,
                )
                file_name = (
                    dataset.samples[sample_index][0].name
                    if sample_index < len(dataset.samples)
                    else f"sample_{sample_index:03d}"
                )
                image_rgb = denormalize_image_tensor(images_cpu[i])
                sample_rows.append(
                    {
                        "sample_index": sample_index,
                        "file_name": file_name,
                        "mean_iou": sample_metrics["mean_iou"],
                        "mean_dice": sample_metrics["mean_dice"],
                        "foreground_mean_iou": sample_metrics["foreground_mean_iou"],
                        "foreground_mean_dice": sample_metrics["foreground_mean_dice"],
                        "per_class": sample_metrics["per_class"],
                        "original_b64": _png_data_url(image_rgb),
                        "truth_b64": _png_data_url(colorize_mask(true_mask)),
                        "prediction_b64": _png_data_url(overlay_mask(image_rgb, pred_mask)),
                    }
                )
                sample_index += 1

    report = metrics_from_confusion_matrix(total_cm, class_names=class_names, background_index=0)
    sorted_rows = sorted(sample_rows, key=lambda row: row["foreground_mean_dice"])

    def _choose_example(kind: str) -> dict | None:
        if not sorted_rows:
            return None
        if kind == "best":
            chosen = sorted_rows[-1]
        elif kind == "worst":
            chosen = sorted_rows[0]
        else:
            chosen = sorted_rows[len(sorted_rows) // 2]
        return {
            "sample_index": chosen["sample_index"],
            "file_name": chosen["file_name"],
            "mean_iou": _round_float(chosen["mean_iou"]),
            "mean_dice": _round_float(chosen["mean_dice"]),
            "foreground_mean_iou": _round_float(chosen["foreground_mean_iou"]),
            "foreground_mean_dice": _round_float(chosen["foreground_mean_dice"]),
            "per_class": [
                {
                    **metric,
                    "iou": _round_float(metric["iou"]),
                    "dice": _round_float(metric["dice"]),
                    "precision": _round_float(metric["precision"]),
                    "recall": _round_float(metric["recall"]),
                    "specificity": _round_float(metric["specificity"]),
                }
                for metric in chosen["per_class"]
            ],
            "original_b64": chosen["original_b64"],
            "truth_b64": chosen["truth_b64"],
            "prediction_b64": chosen["prediction_b64"],
        }

    def _resolved_path(path_str: str) -> str:
        path = Path(path_str)
        if path.is_absolute():
            return str(path)
        return str((_project_root() / path).resolve())

    payload = {
        "checkpoint_path": str(model_path),
        "dataset": {
            "image_dir": _resolved_path(test_image_dir),
            "mask_dir": _resolved_path(test_mask_dir),
            "image_size": list(image_size),
            "test_samples": len(dataset),
        },
        "summary": {
            "pixel_accuracy": _round_float(report["pixel_accuracy"]),
            "mean_iou": _round_float(report["mean_iou"]),
            "mean_dice": _round_float(report["mean_dice"]),
            "foreground_mean_iou": _round_float(report["foreground_mean_iou"]),
            "foreground_mean_dice": _round_float(report["foreground_mean_dice"]),
            "total_pixels": int(report["total_pixels"]),
        },
        "classes": [
            {
                **metric,
                "iou": _round_float(metric["iou"]),
                "dice": _round_float(metric["dice"]),
                "precision": _round_float(metric["precision"]),
                "recall": _round_float(metric["recall"]),
                "specificity": _round_float(metric["specificity"]),
            }
            for metric in report["per_class"]
        ],
        "confusion_matrix": report["confusion_matrix"],
        "examples": {
            "best": _choose_example("best"),
            "median": _choose_example("median"),
            "worst": _choose_example("worst"),
        },
    }

    _viz_cache["segmentation_report"] = payload
    return payload


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


@analysis_bp.route("/analysis/segmentation-report", methods=["GET"])
def segmentation_report():
    try:
        return jsonify(_build_segmentation_report()), 200
    except Exception as exc:
        current_app.logger.exception("segmentation-report failed")
        return jsonify({"error": str(exc), "trace": traceback.format_exc()}), 500


@analysis_bp.route("/analysis/cache/clear", methods=["POST"])
def clear_cache():
    _viz_cache.clear()
    return jsonify({"message": "Cache cleared"}), 200
