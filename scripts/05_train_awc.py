"""Train Adaptive Weighted Clustering (AWC) model from extracted features."""

from __future__ import annotations

import argparse
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

import sys
from pathlib import Path

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

sys.path.append(str(Path(__file__).parent.parent))

from src.clustering.awc import AdaptiveWeightedClustering, evaluate_clustering
from src.utils.config import load_config
from src.utils.file_utils import create_directories


def _select_feature_indices(config: dict, X_train: np.ndarray, y_train: np.ndarray | None) -> list[int] | None:
    advanced = config.get("advanced", {})
    if not advanced.get("feature_selection", False):
        return None

    method = str(advanced.get("selection_method", "variance")).lower()
    if method != "anova":
        raise ValueError(f"Unsupported AWC feature selection method: {method}")
    if y_train is None:
        raise ValueError("ANOVA feature selection requires training labels")

    k_best = int(advanced.get("k_best", min(20, X_train.shape[1])))
    k_best = max(1, min(k_best, X_train.shape[1]))
    selector = SelectKBest(score_func=f_classif, k=k_best)
    selector.fit(X_train, y_train)
    return selector.get_support(indices=True).astype(int).tolist()


def main():
    parser = argparse.ArgumentParser(description="Train AWC model")
    parser.add_argument("--config", default="configs/awc_config.yaml", help="AWC config path")
    args = parser.parse_args()

    config = load_config(args.config)
    train_data_path = Path(config["data"]["train_data"])
    train_labels_path = Path(config["data"]["train_labels"])
    test_data_path = Path(config["data"]["test_data"])
    test_labels_path = Path(config["data"]["test_labels"])

    X_train = np.load(train_data_path)
    y_train = np.load(train_labels_path) if train_labels_path.exists() else None
    X_test = np.load(test_data_path) if test_data_path.exists() else X_train
    y_test = np.load(test_labels_path) if test_labels_path.exists() else None
    feature_indices = _select_feature_indices(config, X_train, y_train)

    model = AdaptiveWeightedClustering(
        n_clusters=config["data"]["n_clusters"],
        max_iter=config["algorithm"]["max_iter"],
        tol=config["algorithm"]["tol"],
        initial_weights=config["algorithm"]["initial_weights"],
        feature_importance=config["algorithm"]["feature_importance"],
        feature_indices=feature_indices,
        random_state=config["algorithm"]["random_state"],
    )

    model.fit(X_train)
    pred_test = model.predict(X_test)

    eval_metrics = evaluate_clustering(X_test, pred_test, y_test)
    model_info = model.get_cluster_info()

    checkpoint_path = Path(config["training"]["checkpoint_path"])
    create_directories(str(checkpoint_path.parent))
    model.save(str(checkpoint_path))

    results_dir = Path(config["evaluation"]["output_dir"])
    create_directories(str(results_dir))

    metrics_path = Path(config["evaluation"]["metrics_path"])
    create_directories(str(metrics_path.parent))

    output = {
        "evaluation": eval_metrics,
        "cluster_info": model_info,
        "checkpoint_path": str(checkpoint_path),
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "selected_feature_count": len(feature_indices) if feature_indices is not None else X_train.shape[1],
        "selected_feature_indices": feature_indices,
        "train_labels_available": y_train is not None,
        "test_labels_available": y_test is not None,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    print(f"Model saved: {checkpoint_path}")
    if feature_indices is not None:
        print(f"Selected ANOVA features: {len(feature_indices)}/{X_train.shape[1]}")
    print(f"Metrics saved: {metrics_path}")
    print("Evaluation:")
    for key, value in eval_metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
