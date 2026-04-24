"""
SHAP Explainer for AWC / clustering-based fertility classifier.

Uses KernelSHAP (model-agnostic) so it works regardless of whether the
underlying model is AWC, K-Means, or FCM.  Returns per-feature importance
scores and saves a summary bar-chart suitable for a journal figure.

Usage:
    explainer = SHAPExplainer(awc_model, feature_names)
    shap_values = explainer.explain(X_test)
    explainer.save_summary("results/xai/shap_summary.png", X_test)
    explainer.save_waterfall("results/xai/shap_waterfall_0.png", 0)
"""

from __future__ import annotations

import warnings
import numpy as np
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Internal predict wrapper — gives KernelSHAP a probability interface
# ---------------------------------------------------------------------------

class _ClusterPredictor:
    """Wraps any clustering model that exposes predict_proba(X)."""

    def __init__(self, model) -> None:
        self.model = model

    # Always binary: infertile=0, fertile=1
    N_CLASSES = 2

    def __call__(self, X: np.ndarray) -> np.ndarray:
        proba = None

        if hasattr(self.model, "predict_proba"):
            try:
                proba = self.model.predict_proba(X)
            except Exception:
                proba = None  # fall through to predict()

        if proba is None:
            labels = self.model.predict(X).astype(int)
            # Clamp labels so index never exceeds N_CLASSES-1
            labels = np.clip(labels, 0, self.N_CLASSES - 1)
            proba = np.eye(self.N_CLASSES)[labels]      # (N, 2)

        proba = np.asarray(proba, dtype=float)

        # Ensure output is always shape (N, 2): [P(infertile), P(fertile)]
        if proba.ndim == 1 or proba.shape[1] == 1:
            p = proba.ravel()
            return np.column_stack([1.0 - p, p])
        if proba.shape[1] >= 2:
            out = proba[:, :2]
            row_sum = out.sum(axis=1, keepdims=True)
            row_sum = np.where(row_sum == 0, 1.0, row_sum)
            return out / row_sum
        return np.full((len(X), 2), 0.5)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SHAPExplainer:
    """SHAP-based explainer for the AWC fertility classifier.

    Parameters
    ----------
    model:
        Fitted clustering/classification model (AWC, KMeansBaseline, FuzzyCMeans).
        Must expose either ``predict_proba(X)`` or ``predict(X)``.
    feature_names:
        List of feature names (used in plots).
    background_samples:
        Number of background samples for KernelSHAP (higher = more accurate,
        slower).  100–200 is a good default for publication figures.
    """

    FERTILE_CLASS_IDX = 1

    def __init__(
        self,
        model,
        feature_names: Optional[List[str]] = None,
        background_samples: int = 100,
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.background_samples = background_samples
        self._explainer = None
        self._shap_values: Optional[np.ndarray] = None
        self._X_explained: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def fit_explainer(self, X_background: np.ndarray) -> None:
        """Fit KernelSHAP background distribution from training data."""
        try:
            import shap
        except ImportError:
            raise ImportError(
                "shap is not installed. Run:  pip install shap"
            )

        n = min(self.background_samples, len(X_background))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_background), size=n, replace=False)
        background = X_background[idx]

        predictor = _ClusterPredictor(self.model)
        self._explainer = shap.KernelExplainer(predictor, background)

    def explain(
        self,
        X: np.ndarray,
        nsamples: int = 512,
    ) -> np.ndarray:
        """Compute SHAP values for samples in X.

        Returns
        -------
        shap_values : np.ndarray, shape (N, n_features)
            SHAP values for the *fertile* class.
        """
        if self._explainer is None:
            raise RuntimeError("Call fit_explainer(X_background) first.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sv = self._explainer.shap_values(X, nsamples=nsamples)

        # sv is a list [shap_infertile, shap_fertile] or a single array
        if isinstance(sv, list) and len(sv) >= 2:
            values = sv[self.FERTILE_CLASS_IDX]
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            values = sv[:, :, self.FERTILE_CLASS_IDX]
        else:
            values = np.asarray(sv)

        self._shap_values = values
        self._X_explained = X
        return values

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def feature_importance(self) -> Tuple[np.ndarray, List[str]]:
        """Return (mean |SHAP|, feature_names) sorted descending."""
        if self._shap_values is None:
            raise RuntimeError("Run explain() first.")
        importance = np.abs(self._shap_values).mean(axis=0)
        order = np.argsort(importance)[::-1]
        names = self._feature_names_list()
        return importance[order], [names[i] for i in order]

    def top_features(self, k: int = 10) -> List[Tuple[str, float]]:
        """Return top-k (feature_name, mean_|SHAP|) pairs."""
        imp, names = self.feature_importance()
        return list(zip(names[:k], imp[:k].tolist()))

    # ------------------------------------------------------------------
    # Saving figures
    # ------------------------------------------------------------------

    def save_summary(
        self,
        output_path: str,
        X: Optional[np.ndarray] = None,
        max_display: int = 15,
        title: str = "SHAP Feature Importance (fertile class)",
    ) -> None:
        """Save a bar-chart of mean |SHAP| values — publication-ready."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for saving figures.")

        if self._shap_values is None:
            if X is not None:
                self.explain(X)
            else:
                raise RuntimeError("Run explain(X) before saving summary.")

        imp, names = self.feature_importance()
        n = min(max_display, len(names))
        imp_top, names_top = imp[:n][::-1], names[:n][::-1]

        fig, ax = plt.subplots(figsize=(8, max(4, n * 0.35)))
        bars = ax.barh(names_top, imp_top, color="#2196F3", edgecolor="none")
        ax.set_xlabel("Mean |SHAP value|", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Annotate each bar with its value
        for bar, val in zip(bars, imp_top):
            ax.text(
                val + imp_top.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                fontsize=8,
            )

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def save_waterfall(
        self,
        output_path: str,
        sample_idx: int = 0,
        title: str = "",
    ) -> None:
        """Save a waterfall plot for a single sample."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required.")

        if self._shap_values is None:
            raise RuntimeError("Run explain() first.")

        sv = self._shap_values[sample_idx]
        names = self._feature_names_list()
        order = np.argsort(np.abs(sv))[::-1][:20]   # top 20

        sv_top   = sv[order]
        names_top = [names[i] for i in order]
        colours  = ["#F44336" if v > 0 else "#2196F3" for v in sv_top]

        fig, ax = plt.subplots(figsize=(8, max(4, len(order) * 0.4)))
        ax.barh(names_top[::-1], sv_top[::-1], color=colours[::-1], edgecolor="none")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value (impact on fertile prediction)", fontsize=11)
        ax.set_title(title or f"SHAP Waterfall — sample {sample_idx}", fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def save_report(self, output_path: str) -> None:
        """Save a JSON report with feature importances."""
        import json
        imp, names = self.feature_importance()
        report = {
            "method": "KernelSHAP",
            "target_class": "fertile",
            "n_samples_explained": int(len(self._shap_values)),
            "feature_importances": [
                {"feature": name, "mean_abs_shap": float(val)}
                for name, val in zip(names, imp.tolist())
            ],
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _feature_names_list(self) -> List[str]:
        if self.feature_names:
            return list(self.feature_names)
        n = self._shap_values.shape[1]
        return [f"feature_{i}" for i in range(n)]