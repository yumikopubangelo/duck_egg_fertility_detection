"""
Full-dataset evaluation script for statistical significance testing.

Uses ALL labelled images in data/preprocessed/ (train + val + test = 230+ images)
to produce per-sample predictions from AWC, K-Means, and FCM. Then runs the
complete statistical test suite (Wilcoxon, McNemar, Friedman) on real data.

Workflow:
  1. Collect all images from data/preprocessed/{train,val,test}/{fertile,infertile}/
  2. Extract classical features (including GLCM texture descriptors)
  3. Re-fit K-Means and FCM on train split, AWC loaded from checkpoint
  4. Predict on test + val split (held-out)
  5. Save per-sample predictions to results/evaluation/
  6. Run statistical tests and save report

Usage:
    python scripts/12_full_statistical_evaluation.py
    python scripts/12_full_statistical_evaluation.py --splits test val
    python scripts/12_full_statistical_evaluation.py --splits test val train --alpha 0.01
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.classical_features import ClassicalFeatureExtractor
from src.clustering.awc import AdaptiveWeightedClustering
from src.clustering.kmeans_baseline import KMeansBaseline
from src.clustering.fuzzy_cmeans import FuzzyCMeans
from src.evaluation.statistical_tests import StatisticalTestSuite
from src.utils.logger import setup_logger

logger = setup_logger("full_eval", level=logging.INFO)

PREPROCESSED = Path("data/preprocessed")
OUT_EVAL     = Path("results/evaluation")
OUT_STAT     = Path("results/statistical_tests")
AWC_CKPT     = Path("models/awc/awc_model.pkl")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def collect_images(splits: List[str]) -> Tuple[np.ndarray, np.ndarray, List[Path]]:
    """Return X (N, F), y (N,), paths for the requested splits."""
    extractor = ClassicalFeatureExtractor()
    X_list, y_list, paths = [], [], []

    for split in splits:
        for label_name, label_id in [("fertile", 1), ("infertile", 0)]:
            folder = PREPROCESSED / split / label_name
            if not folder.exists():
                logger.warning(f"Folder not found: {folder}")
                continue
            imgs = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
            logger.info(f"  {split}/{label_name}: {len(imgs)} images")

            for img_path in imgs:
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    logger.warning(f"  Cannot read {img_path.name}, skipping")
                    continue
                feat = extractor.extract(bgr)
                X_list.append(feat)
                y_list.append(label_id)
                paths.append(img_path)

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=int)
    return X, y, paths


# ---------------------------------------------------------------------------
# Model loading / fitting
# ---------------------------------------------------------------------------

def load_awc() -> AdaptiveWeightedClustering | None:
    if AWC_CKPT.exists() and AWC_CKPT.stat().st_size > 0:
        try:
            m = AdaptiveWeightedClustering.load(str(AWC_CKPT))
            logger.info(f"AWC loaded from {AWC_CKPT}")
            return m
        except Exception as e:
            logger.warning(f"AWC load failed ({e}), will refit from training data")
    return None


def _build_cluster_label_map(model, X_ref: np.ndarray, y_ref: np.ndarray) -> dict:
    """Majority-vote mapping: cluster_id -> binary label, using labelled reference data."""
    raw = model.predict(X_ref).astype(int)
    cluster_map = {}
    for cid in np.unique(raw):
        mask = raw == cid
        majority = int(np.bincount(y_ref[mask].astype(int)).argmax())
        cluster_map[cid] = majority
    return cluster_map


def fit_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    awc_pretrained=None,
) -> Dict:
    models = {}
    cluster_maps = {}   # only needed for AWC (K-Means/FCM already do label mapping)

    # AWC ----------------------------------------------------------------
    if awc_pretrained is not None:
        models["AWC"] = awc_pretrained
    else:
        logger.info("Fitting AWC on train split ...")
        awc = AdaptiveWeightedClustering(n_clusters=2, random_state=42)
        awc.fit(X_train)
        models["AWC"] = awc

    # Build cluster->label map for AWC using training labels (majority vote)
    logger.info("Building AWC cluster->label mapping from training data ...")
    cluster_maps["AWC"] = _build_cluster_label_map(models["AWC"], X_train, y_train)
    logger.info(f"  AWC cluster map: {cluster_maps['AWC']}")

    # K-Means ------------------------------------------------------------
    logger.info("Fitting K-Means on train split ...")
    km = KMeansBaseline(n_clusters=2, random_state=42)
    km.fit(X_train, y_train)
    models["KMeans"] = km

    # FCM ----------------------------------------------------------------
    logger.info("Fitting Fuzzy C-Means on train split ...")
    fcm = FuzzyCMeans(c=2, random_state=42)
    fcm.fit(X_train, y_train)
    models["FCM"] = fcm

    return models, cluster_maps


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _binary_predict(model, X: np.ndarray, cluster_map: dict | None = None) -> np.ndarray:
    """Return binary int predictions (0/1).

    For AWC: uses a pre-built cluster_map (majority vote from training data).
    For KMeans/FCM: their predict() already returns binary labels directly.
    """
    raw = model.predict(X).astype(int)

    if cluster_map is not None:
        # Apply explicit majority-vote mapping
        preds = np.array([cluster_map.get(c, 0) for c in raw], dtype=int)
    else:
        preds = raw

    return np.clip(preds, 0, 1)


def compute_per_sample_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, List[float]]:
    """Compute per-sample binary metrics."""
    metrics: Dict[str, List[float]] = {
        "accuracy": [],
        "correct":  [],
    }
    for yt, yp in zip(y_true, y_pred):
        correct = int(yt == yp)
        metrics["accuracy"].append(float(correct))
        metrics["correct"].append(float(correct))
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Full dataset evaluation + statistical significance tests"
    )
    p.add_argument(
        "--splits", nargs="+",
        default=["test", "val"],
        choices=["train", "val", "test"],
        help="Splits to use for evaluation (default: test val)"
    )
    p.add_argument(
        "--train-splits", nargs="+",
        default=["train"],
        choices=["train", "val", "test"],
        help="Splits to use for model fitting (default: train)"
    )
    p.add_argument("--alpha", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT_EVAL.mkdir(parents=True, exist_ok=True)
    OUT_STAT.mkdir(parents=True, exist_ok=True)

    extractor = ClassicalFeatureExtractor()
    feature_names = extractor.feature_names

    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    logger.info("=== Loading training data ===")
    X_train, y_train, _ = collect_images(args.train_splits)
    logger.info(f"Train: {len(X_train)} images  (fertile={y_train.sum()}, infertile={(y_train==0).sum()})")

    logger.info("=== Loading evaluation data ===")
    X_eval, y_eval, eval_paths = collect_images(args.splits)
    logger.info(f"Eval:  {len(X_eval)} images  (fertile={y_eval.sum()}, infertile={(y_eval==0).sum()})")

    if len(X_eval) < 6:
        logger.error("Not enough evaluation images. Need at least 6.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 2. Load / fit models
    # ------------------------------------------------------------------ #
    logger.info("=== Fitting models ===")
    awc_pretrained = load_awc()
    models, cluster_maps = fit_models(X_train, y_train, awc_pretrained)

    # ------------------------------------------------------------------ #
    # 3. Predict on eval set
    # ------------------------------------------------------------------ #
    logger.info("=== Running predictions ===")
    predictions: Dict[str, np.ndarray] = {}
    metrics_per_model: Dict[str, Dict[str, List[float]]] = {}

    for name, model in models.items():
        t0 = time.time()
        preds = _binary_predict(model, X_eval, cluster_map=cluster_maps.get(name))
        elapsed = time.time() - t0

        predictions[name] = preds
        metrics_per_model[name] = compute_per_sample_metrics(y_eval, preds)

        accuracy = float((preds == y_eval).mean())
        logger.info(f"  {name:8s}  accuracy={accuracy:.3f}  ({elapsed:.1f}s)")

    # ------------------------------------------------------------------ #
    # 4. Save predictions
    # ------------------------------------------------------------------ #
    np.save(str(OUT_EVAL / "true_labels.npy"), y_eval)
    for name, preds in predictions.items():
        np.save(str(OUT_EVAL / f"{name.lower()}_predictions.npy"), preds)

    # Save human-readable CSV
    import csv
    csv_path = OUT_EVAL / "full_evaluation_predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["image", "split", "true_label"] + list(predictions.keys())
        writer.writerow(header)
        for i, path in enumerate(eval_paths):
            split = path.parts[-3]      # train / val / test
            true  = "fertile" if y_eval[i] == 1 else "infertile"
            row   = [path.name, split, true] + [
                "fertile" if predictions[m][i] == 1 else "infertile"
                for m in predictions
            ]
            writer.writerow(row)
    logger.info(f"Predictions saved to {csv_path}")

    # ------------------------------------------------------------------ #
    # 5. Statistical tests
    # ------------------------------------------------------------------ #
    logger.info("=== Running statistical tests ===")

    suite = StatisticalTestSuite(alpha=args.alpha)

    # Wilcoxon: AWC vs each baseline
    if "AWC" in metrics_per_model:
        for baseline in ("KMeans", "FCM"):
            if baseline in metrics_per_model:
                suite.add_wilcoxon(
                    metrics_a=metrics_per_model["AWC"],
                    metrics_b=metrics_per_model[baseline],
                    model_a_name="AWC",
                    model_b_name=baseline,
                )

    # McNemar: all pairwise
    model_names = list(predictions.keys())
    y_true_list = y_eval.tolist()
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            ma, mb = model_names[i], model_names[j]
            suite.add_mcnemar(
                y_true=y_true_list,
                y_pred_a=predictions[ma].tolist(),
                y_pred_b=predictions[mb].tolist(),
                model_a_name=ma,
                model_b_name=mb,
            )

    # Friedman: all three models on accuracy
    if len(predictions) >= 3:
        suite.add_friedman(
            model_scores={
                name: metrics_per_model[name]["accuracy"]
                for name in model_names
            },
            metric_name="accuracy",
        )

    results = suite.run()
    suite.print_summary()

    # Save outputs
    suite.save(str(OUT_STAT / "report_full.json"))
    suite.save_latex_table(str(OUT_STAT / "wilcoxon_table_full.tex"))

    # Also save per-model accuracy summary
    summary = {
        "n_eval_images": int(len(X_eval)),
        "splits_used": args.splits,
        "class_distribution": {
            "fertile":   int(y_eval.sum()),
            "infertile": int((y_eval == 0).sum()),
        },
        "model_accuracy": {
            name: float((predictions[name] == y_eval).mean())
            for name in predictions
        },
        "feature_names": feature_names,
        "n_features": len(feature_names),
    }
    (OUT_STAT / "eval_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    logger.info(f"Full report saved to {OUT_STAT}/report_full.json")
    logger.info(f"LaTeX table saved  to {OUT_STAT}/wilcoxon_table_full.tex")
    logger.info("Done.")


if __name__ == "__main__":
    main()
