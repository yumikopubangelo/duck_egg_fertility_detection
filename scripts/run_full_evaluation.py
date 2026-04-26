"""
Pipeline Evaluasi Lengkap — Deteksi Fertilitas Telur Bebek
==========================================================
Menjalankan dan membandingkan tiga model clustering:
  - Adaptive Weight Clustering (AWC)
  - K-Means (baseline)
  - Fuzzy C-Means (baseline)

Output yang dihasilkan (tersimpan di results/evaluation/):
  - Confusion matrix (PNG) per model
  - Combined ROC curve (PNG)
  - Combined Precision-Recall curve (PNG)
  - Tabel perbandingan metrik (CSV + JSON)
  - Hasil uji statistik Wilcoxon (JSON)
  - Laporan teks ringkas (TXT)

Penggunaan:
  python scripts/run_full_evaluation.py
  python scripts/run_full_evaluation.py --features data/features/awc_test_features.npy \
                                          --labels   data/features/awc_test_labels.npy  \
                                          --train-features data/features/awc_features.npy \
                                          --train-labels   data/features/awc_labels.npy  \
                                          --awc-model  models/awc/awc_model.pkl          \
                                          --output-dir results/evaluation/full
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np

# ── path setup ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score,
)
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.stats import wilcoxon

from src.clustering.awc import AdaptiveWeightedClustering
from src.clustering.kmeans_baseline import KMeansBaseline
from src.clustering.fuzzy_cmeans import FuzzyCMeans

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# ── colour palette ──────────────────────────────────────────────────────────
PALETTE = {
    "AWC":    "#1A3A6E",
    "KMeans": "#E07B39",
    "FCM":    "#2E9E5B",
}

# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return super().default(obj)


def compute_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn = cm[0, 0]
    fp = cm[0, 1]
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def compute_all_metrics(y_true, y_pred, y_proba_pos):
    """Return a dict of all classification metrics."""
    return {
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "precision":   float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":      float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(compute_specificity(y_true, y_pred)),
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":     float(roc_auc_score(y_true, y_proba_pos)),
        "brier_score": float(brier_score_loss(y_true, y_proba_pos)),
    }


def select_awc_feature_indices(X_train, y_train, method="anova", k_best=20):
    """Return selected feature indices for the AWC model."""
    method = (method or "none").lower()
    if method in {"none", "off", "false"}:
        return None
    if method != "anova":
        raise ValueError(f"Unsupported AWC feature selection method: {method}")

    k_best = max(1, min(int(k_best), X_train.shape[1]))
    selector = SelectKBest(score_func=f_classif, k=k_best)
    selector.fit(X_train, y_train)
    return selector.get_support(indices=True).astype(int).tolist()


def awc_positive_scores(model, X, cluster_to_label):
    """Distance-softmax probability for the fertile label in AWC's scaled space."""
    cluster_probs = model.predict_proba(X)
    positive = np.zeros(cluster_probs.shape[0], dtype=np.float32)
    for cid, label in cluster_to_label.items():
        if int(label) == 1:
            positive += cluster_probs[:, int(cid)]
    return positive


# ═══════════════════════════════════════════════════════════════════════════
#  Visualisation helpers
# ═══════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, y_pred, model_name, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Count", "Normalised"],
        ["d", ".2f"],
    ):
        im = ax.imshow(data, interpolation="nearest", cmap="Blues",
                       vmin=0, vmax=(1.0 if fmt == ".2f" else cm.max()))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        tick_labels = ["Infertil (0)", "Fertil (1)"]
        ax.set_xticks([0, 1]); ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_yticks([0, 1]); ax.set_yticklabels(tick_labels, fontsize=9, rotation=90, va="center")
        thresh = data.max() / 2.0
        for r in range(2):
            for c in range(2):
                ax.text(c, r, format(data[r, c], fmt),
                        ha="center", va="center", fontsize=12,
                        color="white" if data[r, c] > thresh else "black")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out_path}")


def plot_roc_curves(results_dict, out_path):
    """Plot ROC curves for all models on a single figure."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.50)")

    for name, res in results_dict.items():
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_proba"])
        auc = res["metrics"]["roc_auc"]
        ax.plot(fpr, tpr, color=PALETTE[name], lw=2,
                label=f"{name} (AUC = {auc:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — Perbandingan Model", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out_path}")


def plot_pr_curves(results_dict, out_path):
    """Plot Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, res in results_dict.items():
        prec, rec, _ = precision_recall_curve(res["y_true"], res["y_proba"])
        ap = average_precision_score(res["y_true"], res["y_proba"])
        ax.plot(rec, prec, color=PALETTE[name], lw=2,
                label=f"{name} (AP = {ap:.3f})")

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curves — Perbandingan Model", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out_path}")


def plot_metrics_bar(results_dict, out_path):
    """Grouped bar chart comparing all metrics across models."""
    metric_keys   = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "ROC-AUC"]

    model_names = list(results_dict.keys())
    x = np.arange(len(metric_keys))
    width = 0.25
    offsets = np.linspace(-(width * (len(model_names) - 1) / 2),
                          (width * (len(model_names) - 1) / 2),
                          len(model_names))

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (name, offset) in enumerate(zip(model_names, offsets)):
        vals = [results_dict[name]["metrics"][k] for k in metric_keys]
        bars = ax.bar(x + offset, vals, width, label=name,
                      color=PALETTE[name], alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Perbandingan Metrik Evaluasi — AWC vs K-Means vs FCM",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Statistical tests
# ═══════════════════════════════════════════════════════════════════════════

def run_wilcoxon_tests(results_dict):
    """
    Wilcoxon Signed-Rank Test between every pair of models.
    Uses per-sample binary correctness (1=correct, 0=wrong) as the paired scores.
    """
    model_names = list(results_dict.keys())
    wilcoxon_results = {}

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            n1, n2 = model_names[i], model_names[j]
            y_true = np.array(results_dict[n1]["y_true"])

            # Per-sample correctness
            correct1 = (np.array(results_dict[n1]["y_pred"]) == y_true).astype(float)
            correct2 = (np.array(results_dict[n2]["y_pred"]) == y_true).astype(float)

            pair_key = f"{n1}_vs_{n2}"
            diff = correct1 - correct2
            if np.all(diff == 0):
                wilcoxon_results[pair_key] = {
                    "statistic": None, "p_value": None,
                    "significant": False,
                    "note": "Identical predictions — test not applicable"
                }
            else:
                try:
                    stat, pval = wilcoxon(correct1, correct2,
                                         alternative="two-sided",
                                         zero_method="wilcox")
                    wilcoxon_results[pair_key] = {
                        "statistic": float(stat),
                        "p_value": float(pval),
                        "significant": bool(pval < 0.05),
                        "better_model": n1 if correct1.mean() > correct2.mean() else n2,
                        "note": f"p={'<0.001' if pval < 0.001 else f'{pval:.4f}'} "
                                f"({'significant' if pval < 0.05 else 'not significant'} at α=0.05)"
                    }
                except Exception as e:
                    wilcoxon_results[pair_key] = {
                        "statistic": None, "p_value": None,
                        "significant": False, "note": str(e)
                    }
    return wilcoxon_results


# ═══════════════════════════════════════════════════════════════════════════
#  Report generation
# ═══════════════════════════════════════════════════════════════════════════

def build_comparison_table(results_dict):
    """Return list-of-dicts suitable for CSV."""
    rows = []
    metric_keys = ["accuracy", "precision", "recall", "specificity",
                   "f1", "roc_auc", "brier_score"]
    for name, res in results_dict.items():
        row = {"model": name}
        for k in metric_keys:
            row[k] = round(res["metrics"][k], 4)
        rows.append(row)
    return rows


def save_csv(rows, path):
    import csv
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  [saved] {path}")


def generate_text_report(results_dict, wilcoxon_results, out_path, meta):
    lines = [
        "=" * 65,
        "  LAPORAN EVALUASI MODEL — DETEKSI FERTILITAS TELUR BEBEK",
        "=" * 65,
        f"  Tanggal  : {meta['timestamp']}",
        f"  Dataset  : {meta['n_test']} sampel uji "
        f"(fertil={meta['n_fertile']}, infertil={meta['n_infertile']})",
        f"  Fitur    : {meta['n_features']} dimensi",
        f"  AWC      : {meta.get('awc_selected_features', meta['n_features'])} fitur terpilih",
        "",
        "-" * 65,
        "  METRIK PERBANDINGAN",
        "-" * 65,
        f"  {'Metrik':<15} {'AWC':>8} {'K-Means':>8} {'FCM':>8}",
        f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8}",
    ]
    for metric in ["accuracy","precision","recall","specificity","f1","roc_auc","brier_score"]:
        label = metric.replace("_", " ").title()
        vals = [results_dict[m]["metrics"][metric] for m in ["AWC","KMeans","FCM"]]
        lines.append(f"  {label:<15} {vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f}")

    lines += [
        "",
        "-" * 65,
        "  UJI STATISTIK WILCOXON (α = 0.05)",
        "-" * 65,
    ]
    for pair, res in wilcoxon_results.items():
        sig = "✓ Signifikan" if res.get("significant") else "✗ Tidak signifikan"
        pval = f"{res['p_value']:.4f}" if res.get("p_value") is not None else "N/A"
        lines.append(f"  {pair:<25} p={pval:<10} {sig}")

    # Determine best model by F1
    best = max(results_dict, key=lambda m: results_dict[m]["metrics"]["f1"])
    lines += [
        "",
        "-" * 65,
        f"  Model terbaik (F1-Score): {best}",
        f"  F1 = {results_dict[best]['metrics']['f1']:.4f}",
        f"  Accuracy = {results_dict[best]['metrics']['accuracy']:.4f}",
        f"  ROC-AUC  = {results_dict[best]['metrics']['roc_auc']:.4f}",
        "=" * 65,
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [saved] {out_path}")
    print("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_evaluation(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────
    print("\n[1/5] Memuat data fitur …")
    X_train = np.load(args.train_features).astype(np.float32)
    y_train = np.load(args.train_labels).astype(int)
    X_test  = np.load(args.features).astype(np.float32)
    y_test  = np.load(args.labels).astype(int)

    n_test, n_features = X_test.shape
    n_fertile   = int((y_test == 1).sum())
    n_infertile = int((y_test == 0).sum())
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")
    print(f"  Kelas test → fertil: {n_fertile}, infertil: {n_infertile}")

    # ── Train / load models ────────────────────────────────────────────
    print("\n[2/5] Melatih / memuat model …")
    results_dict = {}

    # -- AWC --
    print("  → AWC …")

    feature_indices = select_awc_feature_indices(
        X_train,
        y_train,
        method=args.awc_feature_selection,
        k_best=args.awc_k_best,
    )
    if feature_indices is not None:
        print(f"     seleksi fitur: ANOVA top-{len(feature_indices)} dari {X_train.shape[1]} fitur")

    # Always retrain AWC with n_clusters=2 for binary classification
    awc = AdaptiveWeightedClustering(
        n_clusters=2,
        max_iter=100,
        tol=1e-4,
        feature_indices=feature_indices,
        random_state=42,
    )
    awc.fit(X_train)
    awc_pred  = awc.predict(X_test)

    # Map AWC cluster IDs to fertility labels using majority-vote on train set
    train_pred = awc.predict(X_train)
    cluster_to_label = {}
    for cid in np.unique(train_pred):
        mask = train_pred == cid
        majority = int(np.bincount(y_train[mask]).argmax())
        cluster_to_label[cid] = majority
    awc_pred_mapped  = np.array([cluster_to_label.get(c, c) for c in awc_pred])
    train_pred_mapped = np.array([cluster_to_label.get(c, c) for c in train_pred])

    awc_proba = awc_positive_scores(awc, X_test, cluster_to_label)

    results_dict["AWC"] = {
        "y_true": y_test.tolist(),
        "y_pred": awc_pred_mapped.tolist(),
        "y_proba": awc_proba.tolist(),
        "metrics": compute_all_metrics(y_test, awc_pred_mapped, awc_proba),
    }
    print(f"     Accuracy={results_dict['AWC']['metrics']['accuracy']:.4f}  "
          f"F1={results_dict['AWC']['metrics']['f1']:.4f}")

    # -- K-Means --
    print("  → K-Means …")
    km = KMeansBaseline(n_clusters=2, random_state=42)
    km.fit(X_train, y_train)
    km_pred  = km.predict(X_test)
    km_proba = km.predict_proba(X_test)[:, 1]
    results_dict["KMeans"] = {
        "y_true":  y_test.tolist(),
        "y_pred":  km_pred.tolist(),
        "y_proba": km_proba.tolist(),
        "metrics": compute_all_metrics(y_test, km_pred, km_proba),
    }
    print(f"     Accuracy={results_dict['KMeans']['metrics']['accuracy']:.4f}  "
          f"F1={results_dict['KMeans']['metrics']['f1']:.4f}")

    # -- FCM --
    print("  → Fuzzy C-Means …")
    fcm = FuzzyCMeans(c=2, m=2.0, error=0.005, max_iter=1000, random_state=42)
    fcm.fit(X_train, y_train)
    fcm_pred  = fcm.predict(X_test)
    fcm_proba = fcm.predict_proba(X_test)[:, 1]
    results_dict["FCM"] = {
        "y_true":  y_test.tolist(),
        "y_pred":  fcm_pred.tolist(),
        "y_proba": fcm_proba.tolist(),
        "metrics": compute_all_metrics(y_test, fcm_pred, fcm_proba),
    }
    print(f"     Accuracy={results_dict['FCM']['metrics']['accuracy']:.4f}  "
          f"F1={results_dict['FCM']['metrics']['f1']:.4f}")

    # ── Visualisations ─────────────────────────────────────────────────
    print("\n[3/5] Membuat visualisasi …")

    for name in ["AWC", "KMeans", "FCM"]:
        plot_confusion_matrix(
            np.array(results_dict[name]["y_true"]),
            np.array(results_dict[name]["y_pred"]),
            model_name=name,
            out_path=figs_dir / f"confusion_matrix_{name.lower()}.png",
        )

    plot_roc_curves(results_dict, figs_dir / "roc_curves_comparison.png")
    plot_pr_curves(results_dict,  figs_dir / "pr_curves_comparison.png")
    plot_metrics_bar(results_dict, figs_dir / "metrics_bar_comparison.png")

    # ── Statistical tests ──────────────────────────────────────────────
    print("\n[4/5] Menjalankan uji statistik Wilcoxon …")
    wilcoxon_results = run_wilcoxon_tests(results_dict)
    for pair, res in wilcoxon_results.items():
        print(f"  {pair}: {res['note']}")

    # ── Save results ───────────────────────────────────────────────────
    print("\n[5/5] Menyimpan hasil …")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {
        "timestamp": timestamp,
        "n_test":     n_test,
        "n_fertile":  n_fertile,
        "n_infertile": n_infertile,
        "n_features": n_features,
        "awc_feature_selection": args.awc_feature_selection,
        "awc_selected_features": len(feature_indices) if feature_indices is not None else n_features,
    }

    # Full JSON results
    full_output = {
        "meta": meta,
        "results": {
            name: {k: v for k, v in res.items() if k != "y_proba"}
            for name, res in results_dict.items()
        },
        "wilcoxon": wilcoxon_results,
    }
    json_path = out_dir / f"full_evaluation_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, cls=NumpyEncoder)
    print(f"  [saved] {json_path}")

    # CSV comparison table
    table_rows = build_comparison_table(results_dict)
    csv_path = out_dir / "comparison_table.csv"
    save_csv(table_rows, csv_path)

    # Text report
    report_path = out_dir / "evaluation_report.txt"
    generate_text_report(results_dict, wilcoxon_results, report_path, meta)

    # Save trained models
    awc.save(ROOT / args.awc_model)
    km.save(ROOT / "models" / "baselines" / "kmeans_model.pkl")
    fcm.save(ROOT / "models" / "baselines" / "fcm_model.pkl")
    print(f"  [saved] {args.awc_model}")
    print(f"  [saved] models/baselines/kmeans_model.pkl")
    print(f"  [saved] models/baselines/fcm_model.pkl")

    print(f"\n✓ Evaluasi selesai. Semua output tersimpan di: {out_dir}")
    return results_dict


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Pipeline Evaluasi Lengkap")
    p.add_argument("--features",       default="data/features/awc_test_features.npy")
    p.add_argument("--labels",         default="data/features/awc_test_labels.npy")
    p.add_argument("--train-features", default="data/features/awc_features.npy")
    p.add_argument("--train-labels",   default="data/features/awc_labels.npy")
    p.add_argument("--awc-model",      default="models/awc/awc_model.pkl")
    p.add_argument("--output-dir",     default="results/evaluation/full")
    p.add_argument("--awc-feature-selection", default="anova", choices=["none", "anova"])
    p.add_argument("--awc-k-best", type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
