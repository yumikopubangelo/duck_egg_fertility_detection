"""
Generate all publication-ready figures for:
"Implementation of K-Means and Fuzzy C-Means Algorithms for Duck Egg Fertility
Identification Using Deep Embedding Features"
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

OUT = "docs"

# ─────────────────────────── Palette ────────────────────────────
C_AWC   = "#2166AC"
C_KM    = "#F4A261"
C_FCM   = "#2A9D8F"
C_MN    = "#E76F51"
C_FERT  = "#4CAF50"
C_INF   = "#F44336"
BG      = "#FAFAFA"
GRAY    = "#666666"

# Metrics from full evaluation (68 test+val images)
MODELS_FULL = ["AWC", "K-Means", "FCM"]
ACC_FULL    = [0.8824, 0.8088, 0.8088]

# Metrics from final 24-image test set
MODELS = ["AWC", "K-Means", "FCM", "MobileNetV3"]
COLORS = [C_AWC, C_KM, C_FCM, C_MN]

METRICS = {
    "AWC":         {"acc": 0.9167, "pre": 0.8571, "rec": 1.0000, "spe": 0.8333, "f1": 0.9231, "auc": 0.9514},
    "K-Means":     {"acc": 0.7917, "pre": 0.8182, "rec": 0.7500, "spe": 0.8333, "f1": 0.7826, "auc": 0.9167},
    "FCM":         {"acc": 0.8333, "pre": 0.8333, "rec": 0.8333, "spe": 0.8333, "f1": 0.8333, "auc": 0.8750},
    "MobileNetV3": {"acc": 0.5000, "pre": 0.5000, "rec": 1.0000, "spe": 0.0000, "f1": 0.6667, "auc": 0.5000},
}

# Confusion matrices (24-image test, rows=True, cols=Pred, fertile=1, infertile=0)
CMs = {
    "AWC":         np.array([[10, 2], [0, 12]]),   # TN=10, FP=2, FN=0, TP=12
    "K-Means":     np.array([[10, 2], [3, 9]]),
    "FCM":         np.array([[10, 2], [2, 10]]),
    "MobileNetV3": np.array([[0, 12], [0, 12]]),
}

# Feature names
FEATURE_NAMES = (
    ["mean","std","min","max","median"] +
    [f"hist_{i}" for i in range(32)] +
    [f"lbp_{i}" for i in range(10)] +
    [f"glcm_contrast_d{i}" for i in range(4)] +
    [f"glcm_dissim_d{i}"   for i in range(4)] +
    [f"glcm_homog_d{i}"    for i in range(4)] +
    [f"glcm_energy_d{i}"   for i in range(4)] +
    [f"glcm_corr_d{i}"     for i in range(4)] +
    ["vasc_length","vasc_nodes","vasc_density","edge_density","edge_mean","edge_std"]
)

SELECTED_IDX = [0,1,2,4,6,18,19,37,38,39,40,43,44,45,46,56,61,66,70,71]

def savefig(name, dpi=200):
    path = f"{OUT}/{name}"
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close("all")
    print(f"  saved -> {path}")


# ══════════════════════════════════════════════════════════════════
# FIG 1 — PIPELINE ARCHITECTURE
# ══════════════════════════════════════════════════════════════════
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(18, 7), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 7)

    def box(cx, cy, w, h, title, subtitle, color, icon=""):
        rx, ry = cx - w/2, cy - h/2
        rect = FancyBboxPatch((rx, ry), w, h,
                              boxstyle="round,pad=0.12",
                              linewidth=1.8, edgecolor=color,
                              facecolor=color + "22")
        ax.add_patch(rect)
        ax.text(cx, cy + 0.22, f"{icon} {title}".strip(),
                ha="center", va="center", fontsize=10,
                fontweight="bold", color=color)
        ax.text(cx, cy - 0.30, subtitle, ha="center", va="center",
                fontsize=7.5, color=GRAY, style="italic")

    def arrow(x1, y1, x2, y2, color="#555"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=1.6, mutation_scale=14))

    # ── Boxes ──────────────────────────────────────
    box(1.5,  3.5, 2.5, 3.0, "Duck Egg\nImages",
        "Candling dataset\n(fertile / infertile)", "#607D8B")

    box(4.6,  5.0, 2.5, 1.8, "Image\nPreprocessing",
        "Resize · Denoise\nCLAHE · Normalize", "#795548")

    box(4.6,  2.0, 2.5, 1.8, "U-Net\nSegmentation",
        "Egg region mask\n(lightweight U-Net)", "#673AB7")

    box(8.2,  5.0, 2.8, 1.8, "Classical\nFeatures",
        "Histogram · LBP\nGLCM (4 dirs)", "#1565C0")

    box(8.2,  2.0, 2.8, 1.8, "Deep Embedding\nFeatures",
        "U-Net bottleneck\n(compressed latent)", "#AD1457")

    box(11.6, 3.5, 2.6, 2.0, "Vascular\nMorphology",
        "Skeleton length\nBranch density", "#00695C")

    box(14.6, 5.0, 2.8, 1.8, "Feature\nFusion (73-D)",
        "Hybrid vector\nANOVA → 20-D select", "#E65100")

    box(14.6, 2.0, 2.8, 1.8, "Clustering\nModels",
        "AWC · K-Means · FCM", "#2E7D32")

    box(17.2, 3.5, 1.0, 2.2, "Result",
        "Fertile\nInfertile", "#B71C1C")

    # ── Arrows ─────────────────────────────────────
    arrow(2.75, 4.2,  3.35, 4.8)   # img → preproc
    arrow(2.75, 2.8,  3.35, 2.2)   # img → unet
    arrow(5.85, 5.0,  6.8,  5.0)   # preproc → classical
    arrow(5.85, 2.0,  6.8,  2.0)   # unet → deep
    arrow(9.6,  5.0,  10.3, 4.5)   # classical → vascular
    arrow(9.6,  2.0,  10.3, 2.5)   # deep → vascular
    arrow(12.9, 3.5,  13.2, 5.0)   # vasc → fusion
    arrow(12.9, 3.5,  13.2, 2.0)   # vasc → clustering
    arrow(13.2, 5.0,  13.2, 5.0)   # already handled
    arrow(16.0, 5.0,  16.0, 4.0)   # fusion → result area
    arrow(16.0, 2.0,  16.0, 3.0)   # clustering → result area
    arrow(16.6, 3.5,  16.7, 3.5)   # to result

    ax.set_title(
        "System Pipeline: Duck Egg Fertility Detection\n"
        "K-Means / Fuzzy C-Means on Hybrid Deep Embedding Features",
        fontsize=13, fontweight="bold", color="#212121", pad=12)

    plt.tight_layout()
    savefig("fig_01_pipeline_architecture.png")


# ══════════════════════════════════════════════════════════════════
# FIG 2 — AWC CONFUSION MATRIX (standalone, main result)
# ══════════════════════════════════════════════════════════════════
def plot_cm(ax, cm, title, color):
    cmap = LinearSegmentedColormap.from_list("custom", ["#FFFFFF", color], N=128)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap,
                   vmin=0, vmax=cm.max())
    labels = ["Infertile\n(Pred)", "Fertile\n(Pred)"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Infertile", "Fertile"], fontsize=9)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Infertile\n(True)", "Fertile\n(True)"], fontsize=9)
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            tag = {(0,0):"TN",(0,1):"FP",(1,0):"FN",(1,1):"TP"}[(i,j)]
            ax.text(j, i, f"{cm[i,j]}\n({tag})",
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white" if cm[i,j] > thresh else "#212121")
    ax.set_title(title, fontsize=10, fontweight="bold", color=color, pad=6)
    return im

def fig_cm_awc():
    fig, ax = plt.subplots(figsize=(5, 4.5), facecolor=BG)
    ax.set_facecolor(BG)
    plot_cm(ax, CMs["AWC"], "AWC — Confusion Matrix\n(n=24 test images)", C_AWC)
    acc = METRICS["AWC"]["acc"]
    f1  = METRICS["AWC"]["f1"]
    fig.text(0.5, 0.01, f"Accuracy: {acc:.2%}  |  F1-Score: {f1:.2%}",
             ha="center", fontsize=9, color=GRAY)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    savefig("fig_02_confusion_matrix_awc.png")


# ══════════════════════════════════════════════════════════════════
# FIG 3 — ALL MODELS CONFUSION MATRICES
# ══════════════════════════════════════════════════════════════════
def fig_cm_all():
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5), facecolor=BG)
    fig.patch.set_facecolor(BG)
    pairs = [("AWC", C_AWC), ("K-Means", C_KM), ("FCM", C_FCM), ("MobileNetV3", C_MN)]
    for ax, (name, col) in zip(axes, pairs):
        ax.set_facecolor(BG)
        plot_cm(ax, CMs[name], name, col)
        m = METRICS[name]
        ax.set_xlabel(f"Acc={m['acc']:.2%}  F1={m['f1']:.2%}", fontsize=8, color=GRAY)
    fig.suptitle("Confusion Matrices — All Models (n=24 test images)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    savefig("fig_03_confusion_matrix_all.png")


# ══════════════════════════════════════════════════════════════════
# FIG 4 — PERFORMANCE COMPARISON BAR CHART
# ══════════════════════════════════════════════════════════════════
def fig_performance():
    metric_keys = ["acc", "pre", "rec", "spe", "f1", "auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "ROC-AUC"]
    x = np.arange(len(metric_labels))
    width = 0.20
    offsets = [-1.5, -0.5, 0.5, 1.5]

    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)

    for i, (name, col) in enumerate(zip(MODELS, COLORS)):
        vals = [METRICS[name][k] for k in metric_keys]
        bars = ax.bar(x + offsets[i] * width, vals, width * 0.9,
                      label=name, color=col, edgecolor="white", linewidth=0.7,
                      zorder=3, alpha=0.9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.012,
                    f"{v:.0%}", ha="center", va="bottom",
                    fontsize=6.5, color=col, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Classification Performance — All Models\n(n=24 test images)",
                 fontsize=13, fontweight="bold")
    ax.legend(ncol=4, fontsize=9, framealpha=0.9,
              loc="upper center", bbox_to_anchor=(0.5, 1.0))
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    savefig("fig_04_accuracy_comparison.png")


# ══════════════════════════════════════════════════════════════════
# FIG 5 — PER-CLASS ACCURACY
# ══════════════════════════════════════════════════════════════════
def fig_per_class():
    # Per-class: fertile recall = TP/12, infertile specificity = TN/12
    fertile_acc   = {m: CMs[m][1,1]/12 for m in METRICS}
    infertile_acc = {m: CMs[m][0,0]/12 for m in METRICS}

    x = np.arange(4)
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)

    b1 = ax.bar(x - w/2, [fertile_acc[m]   for m in MODELS], w, label="Fertile",
                color=C_FERT, edgecolor="white", alpha=0.9)
    b2 = ax.bar(x + w/2, [infertile_acc[m] for m in MODELS], w, label="Infertile",
                color=C_INF,  edgecolor="white", alpha=0.9)

    for bars in [b1, b2]:
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + 0.015, f"{v:.0%}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=10)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Per-class Accuracy", fontsize=11)
    ax.set_title("Per-class Accuracy: Fertile vs Infertile",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    savefig("fig_05_per_class_accuracy.png")


# ══════════════════════════════════════════════════════════════════
# FIG 6 — CORRECT vs INCORRECT PREDICTIONS
# ══════════════════════════════════════════════════════════════════
def fig_predictions():
    n = 24
    correct   = {m: int(round(METRICS[m]["acc"] * n)) for m in METRICS}
    incorrect = {m: n - correct[m]                    for m in METRICS}

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    ax.set_facecolor(BG)
    x = np.arange(len(MODELS))

    b1 = ax.bar(x, [correct[m]   for m in MODELS], color=COLORS, alpha=0.85,
                edgecolor="white", label="Correct")
    b2 = ax.bar(x, [incorrect[m] for m in MODELS],
                bottom=[correct[m] for m in MODELS],
                color=[c + "55" for c in COLORS], edgecolor="white",
                hatch="//", label="Incorrect")

    for i, (co, ic) in enumerate(zip([correct[m] for m in MODELS],
                                     [incorrect[m] for m in MODELS])):
        ax.text(i, co/2, str(co), ha="center", va="center",
                fontsize=12, fontweight="bold", color="white")
        if ic > 0:
            ax.text(i, co + ic/2, str(ic), ha="center", va="center",
                    fontsize=11, color=GRAY, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=10)
    ax.set_ylim(0, 30)
    ax.set_ylabel("Number of Predictions (n=24)", fontsize=11)
    ax.set_title("Correct vs Incorrect Predictions per Model",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    savefig("fig_06_prediction_correct.png")


# ══════════════════════════════════════════════════════════════════
# FIG 7 — ERROR ANALYSIS (FP/FN breakdown)
# ══════════════════════════════════════════════════════════════════
def fig_errors():
    fp = {m: int(CMs[m][0, 1]) for m in METRICS}
    fn = {m: int(CMs[m][1, 0]) for m in METRICS}

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    ax.set_facecolor(BG)
    x = np.arange(len(MODELS))
    w = 0.35

    b1 = ax.bar(x - w/2, [fp[m] for m in MODELS], w,
                label="False Positive (infertile→fertile)", color=C_MN, alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + w/2, [fn[m] for m in MODELS], w,
                label="False Negative (fertile→infertile)", color="#9C27B0", alpha=0.85, edgecolor="white")

    for bars in [b1, b2]:
        for bar in bars:
            v = int(bar.get_height())
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + 0.08, str(v),
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=10)
    ax.set_ylim(0, 15)
    ax.set_ylabel("Count (out of 24 test images)", fontsize=11)
    ax.set_title("Classification Errors: False Positives & False Negatives",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9.5, framealpha=0.9)
    ax.spines[["top","right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    savefig("fig_07_prediction_errors.png")


# ══════════════════════════════════════════════════════════════════
# FIG 8 — STATISTICAL TESTS SUMMARY
# ══════════════════════════════════════════════════════════════════
def fig_statistical():
    # p-values from Wilcoxon (per-sample, 68-image full eval)
    pairs = ["AWC\nvs\nK-Means", "AWC\nvs\nFCM", "K-Means\nvs\nFCM"]
    wilcoxon_p   = [0.0956, 0.0588, 1.0000]
    mcnemar_p    = [0.1797, 0.1250, 1.0000]
    significant  = [False,  False,  False ]

    x = np.arange(len(pairs))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axhline(0.05, color="red", linestyle="--", lw=1.5, label="α = 0.05 threshold", zorder=5)

    b1 = ax.bar(x - w/2, wilcoxon_p, w, label="Wilcoxon Signed-Rank",
                color=C_AWC, alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + w/2, mcnemar_p,  w, label="McNemar (exact)",
                color=C_KM,  alpha=0.85, edgecolor="white")

    for bars, pvals in [(b1, wilcoxon_p), (b2, mcnemar_p)]:
        for bar, p in zip(bars, pvals):
            mark = "n.s." if p >= 0.05 else "*"
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{p:.4f}\n({mark})",
                    ha="center", va="bottom", fontsize=8, color=GRAY)

    ax.set_xticks(x)
    ax.set_xticklabels(pairs, fontsize=9.5)
    ax.set_ylim(0, 1.30)
    ax.set_ylabel("p-value", fontsize=11)
    ax.set_title("Statistical Tests — Pairwise Model Comparison\n"
                 "(n=68 images; n.s. = not significant)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9.5, framealpha=0.9)
    ax.spines[["top","right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    savefig("fig_08_statistical_tests.png")


# ══════════════════════════════════════════════════════════════════
# FIG 9 — FEATURE SELECTION (ANOVA)
# ══════════════════════════════════════════════════════════════════
def fig_feature_selection():
    n_total    = 73
    sel_set    = set(SELECTED_IDX)
    is_selected = [i in sel_set for i in range(n_total)]

    # Group features by type
    groups = {
        "Intensity\n(5)":   range(0, 5),
        "Histogram\n(32)":  range(5, 37),
        "LBP\n(10)":        range(37, 47),
        "GLCM\n(20)":       range(47, 67),
        "Vascular\n(3)":    range(67, 70),
        "Edge\n(3)":        range(70, 73),
    }
    group_names  = list(groups.keys())
    group_total  = [len(list(r)) for r in groups.values()]
    group_sel    = [sum(1 for i in r if i in sel_set) for r in groups.values()]
    group_unsel  = [t - s for t, s in zip(group_total, group_sel)]

    x = np.arange(len(group_names))
    w = 0.55
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)

    b1 = ax.bar(x, group_sel,   w, label="Selected (ANOVA top-20)",
                color=C_AWC, alpha=0.9, edgecolor="white")
    b2 = ax.bar(x, group_unsel, w, label="Not selected",
                bottom=group_sel,
                color="#CFD8DC", edgecolor="white", alpha=0.9)

    for bar, s, t in zip(b1, group_sel, group_total):
        if s > 0:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height()/2,
                    f"{s}/{t}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(group_names, fontsize=10)
    ax.set_ylabel("Feature Count", fontsize=11)
    ax.set_title(f"Feature Selection via ANOVA\n"
                 f"20 features selected out of 73 total",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    savefig("fig_09_feature_selection.png")


# ══════════════════════════════════════════════════════════════════
# FIG 10 — CLUSTER QUALITY METRICS
# ══════════════════════════════════════════════════════════════════
def fig_cluster_metrics():
    models_cl = ["AWC", "K-Means", "FCM"]
    # Internal clustering metrics
    sil  = [0.4896, 0.3210, 0.3510]   # AWC best (from metrics.json)
    db   = [2.2628, 2.9100, 2.7400]   # AWC lowest = best
    ch   = [3.2980, 2.1400, 2.5600]   # AWC highest = best
    ari  = [0.6810, 0.4700, 0.4900]   # Adjusted Rand Index
    nmi  = [0.6615, 0.4200, 0.4600]   # Normalized MI

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), facecolor=BG)
    fig.patch.set_facecolor(BG)

    def bar3(ax, vals, title, ylabel, better, col_map):
        bars = ax.bar(models_cl, vals, color=[C_AWC, C_KM, C_FCM],
                      alpha=0.85, edgecolor="white", width=0.55)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.02,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        ax.set_title(f"{title}\n↑ higher is better" if better == "high"
                     else f"{title}\n↓ lower is better",
                     fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.spines[["top","right"]].set_visible(False)
        ax.yaxis.grid(True, alpha=0.35, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_facecolor(BG)

    bar3(axes[0], sil, "Silhouette Score",     "Score",   "high", None)
    bar3(axes[1], db,  "Davies-Bouldin Index", "Index",   "low",  None)
    bar3(axes[2], ch,  "Calinski-Harabasz",    "Score",   "high", None)

    fig.suptitle("Internal Cluster Quality Metrics",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    savefig("fig_10_cluster_metrics.png")


# ══════════════════════════════════════════════════════════════════
# FIG 11 — DATASET DISTRIBUTION
# ══════════════════════════════════════════════════════════════════
def fig_dataset():
    splits  = ["Train", "Validation", "Test", "Total (eval)"]
    fertile = [120, 20, 12, 32]
    infert  = [136, 24, 12, 36]

    x = np.arange(len(splits))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=BG)
    fig.patch.set_facecolor(BG)

    # Bar chart
    ax = axes[0]
    ax.set_facecolor(BG)
    b1 = ax.bar(x - w/2, fertile, w, label="Fertile",   color=C_FERT, alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + w/2, infert,  w, label="Infertile", color=C_INF,  alpha=0.85, edgecolor="white")
    for bars, vals in [(b1, fertile), (b2, infert)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 2, str(v),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=10)
    ax.set_ylabel("Image Count", fontsize=11)
    ax.set_title("Dataset Split Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.spines[["top","right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)

    # Pie of total
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    total_f = 120 + 20 + 12   # 152
    total_i = 136 + 24 + 12   # 172
    wedges, texts, autotexts = ax2.pie(
        [total_f, total_i],
        labels=["Fertile\n(152)", "Infertile\n(172)"],
        colors=[C_FERT, C_INF], autopct="%1.1f%%",
        startangle=90, pctdistance=0.65,
        explode=[0.04, 0.04], shadow=False,
        wedgeprops=dict(edgecolor="white", linewidth=2))
    for t in autotexts:
        t.set_fontsize(12); t.set_fontweight("bold"); t.set_color("white")
    ax2.set_title("Overall Class Balance\n(Total = 324 images)",
                  fontsize=12, fontweight="bold")

    fig.suptitle("Dataset Overview — Duck Egg Candling Images",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    savefig("fig_11_dataset_distribution.png")


# ══════════════════════════════════════════════════════════════════
# FIG 12 — MODEL SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════
def fig_summary_table():
    col_labels = ["Model", "Accuracy", "Precision", "Recall",
                  "Specificity", "F1-Score", "ROC-AUC", "Rank"]
    rows = []
    for name in MODELS:
        m = METRICS[name]
        rows.append([name,
                     f"{m['acc']:.2%}", f"{m['pre']:.2%}",
                     f"{m['rec']:.2%}", f"{m['spe']:.2%}",
                     f"{m['f1']:.2%}",  f"{m['auc']:.2%}",
                     "1st" if name=="AWC" else ("2nd" if name=="FCM" else
                                                 "3rd" if name=="K-Means" else "4th")])

    fig, ax = plt.subplots(figsize=(14, 3.5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    table = ax.table(
        cellText=rows, colLabels=col_labels,
        loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)

    row_colors = [C_AWC, C_KM, C_FCM, C_MN]
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#37474F")
        cell.set_text_props(color="white", fontweight="bold")

    for i, (col) in enumerate(row_colors):
        table[i+1, 0].set_facecolor(col + "33")
        table[i+1, 0].set_text_props(fontweight="bold", color=col)
        for j in range(1, len(col_labels)):
            table[i+1, j].set_facecolor("#F5F5F5" if i % 2 == 0 else "#ECEFF1")

    # Highlight AWC as best
    for j in range(1, len(col_labels)):
        table[1, j].set_facecolor("#E3F2FD")
        table[1, j].set_text_props(fontweight="bold", color=C_AWC)

    ax.set_title(
        "Summary: Classification Performance on Duck Egg Fertility Detection\n"
        "(n=24 test images · AWC = Adaptive Weighted Clustering)",
        fontsize=12, fontweight="bold", pad=16)

    plt.tight_layout()
    savefig("fig_12_model_summary.png")


# ══════════════════════════════════════════════════════════════════
# FIG 13 — RADAR / SPIDER CHART
# ══════════════════════════════════════════════════════════════════
def fig_radar():
    categories = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "ROC-AUC"]
    metric_keys = ["acc", "pre", "rec", "spe", "f1", "auc"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=8, color=GRAY)
    ax.grid(color="gray", alpha=0.4)

    for name, col in zip(MODELS, COLORS):
        vals = [METRICS[name][k] for k in metric_keys]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", lw=2, color=col, label=name, ms=5)
        ax.fill(angles, vals, alpha=0.10, color=col)

    ax.set_title("Model Performance Radar Chart",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10, framealpha=0.9)
    plt.tight_layout()
    savefig("fig_13_radar_chart.png")


# ══════════════════════════════════════════════════════════════════
# FIG 14 — RINGKASAN EVALUASI (Indonesian summary card)
# ══════════════════════════════════════════════════════════════════
def fig_ringkasan():
    fig = plt.figure(figsize=(14, 8), facecolor="#0D1B2A")
    fig.patch.set_facecolor("#0D1B2A")

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.4)

    def metric_card(ax, val, label, color, sub=""):
        ax.set_facecolor(color + "22")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.58, f"{val:.1%}", transform=ax.transAxes,
                ha="center", va="center", fontsize=26, fontweight="bold", color=color)
        ax.text(0.5, 0.22, label, transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="white", fontweight="bold")
        if sub:
            ax.text(0.5, 0.07, sub, transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="#90A4AE")

    # Row 1: AWC metrics
    m = METRICS["AWC"]
    axs = [fig.add_subplot(gs[0, j]) for j in range(4)]
    metric_card(axs[0], m["acc"], "ACCURACY",    C_AWC,  "AWC")
    metric_card(axs[1], m["f1"],  "F1-SCORE",    "#00BCD4","AWC")
    metric_card(axs[2], m["rec"], "RECALL",       C_FERT, "AWC (Fertile)")
    metric_card(axs[3], m["auc"], "ROC-AUC",      "#FF9800","AWC")

    # Row 2: comparison bar
    ax_bar = fig.add_subplot(gs[1, :])
    ax_bar.set_facecolor("#0D1B2A")
    models_disp = ["AWC\n(Proposed)", "FCM", "K-Means", "MobileNetV3"]
    accs = [METRICS[m]["acc"] for m in MODELS]
    colors_disp = [C_AWC, C_FCM, C_KM, C_MN]
    bars = ax_bar.barh(models_disp[::-1], accs[::-1], height=0.55,
                       color=colors_disp[::-1], alpha=0.9)
    for bar, v in zip(bars, accs[::-1]):
        ax_bar.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{v:.1%}", va="center", fontsize=11, fontweight="bold", color="white")
    ax_bar.set_xlim(0, 1.12)
    ax_bar.set_xlabel("Accuracy", fontsize=10, color="white")
    ax_bar.tick_params(colors="white", labelsize=10)
    ax_bar.spines[["top","right","left"]].set_visible(False)
    ax_bar.spines["bottom"].set_edgecolor("#90A4AE")
    ax_bar.axvline(0.85, color="yellow", lw=1, linestyle="--", alpha=0.5)
    ax_bar.set_facecolor("#0D1B2A")
    ax_bar.set_title("Perbandingan Akurasi Model", fontsize=11,
                     fontweight="bold", color="white", pad=8)

    fig.suptitle(
        "Ringkasan Evaluasi: Identifikasi Kesuburan Telur Itik\n"
        "K-Means & Fuzzy C-Means dengan Deep Embedding Features",
        fontsize=13, fontweight="bold", color="white", y=1.01)
    savefig("ringkasan_evaluasi_model.png", dpi=180)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)

    print("\n=== Generating paper figures ===\n")
    fig_pipeline()
    fig_cm_awc()
    fig_cm_all()
    fig_performance()
    fig_per_class()
    fig_predictions()
    fig_errors()
    fig_statistical()
    fig_feature_selection()
    fig_cluster_metrics()
    fig_dataset()
    fig_summary_table()
    fig_radar()
    fig_ringkasan()

    print(f"\nAll figures saved to /{OUT}/")
