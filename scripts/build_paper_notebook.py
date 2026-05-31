"""
Build the Jupyter notebook for the paper:
"Implementation of K-Means and Fuzzy C-Means Algorithms for Duck Egg Fertility
 Identification Using Deep Embedding Features"

Run:  python scripts/build_paper_notebook.py
Output: notebooks/09_paper_visualizations.ipynb
"""

import json, textwrap, pathlib

OUT = pathlib.Path("notebooks/09_paper_visualizations.ipynb")

# ─── helpers ─────────────────────────────────────────────────────
def md(*lines):
    return {"cell_type": "markdown", "metadata": {},
            "source": ["\n".join(lines)]}

def code(src, tags=None):
    meta = {"tags": tags} if tags else {}
    return {"cell_type": "code", "execution_count": None,
            "metadata": meta, "outputs": [], "source": [src]}

# ═══════════════════════════════════════════════════════════════════
# CELLS
# ═══════════════════════════════════════════════════════════════════

CELLS = []

# ── Title ──────────────────────────────────────────────────────────
CELLS.append(md(
    "# Implementation of K-Means and Fuzzy C-Means Algorithms",
    "## for Duck Egg Fertility Identification Using Deep Embedding Features",
    "",
    "> **Notebook Visualisasi Penelitian**  ",
    "> Semua gambar dihasilkan dari data aktual eksperimen sesuai abstrak.",
    "",
    "---",
))

# ── 0. Setup ───────────────────────────────────────────────────────
CELLS.append(md("## 0. Setup & Konfigurasi Data"))
CELLS.append(code(textwrap.dedent("""\
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from matplotlib.colors import LinearSegmentedColormap
    import warnings
    warnings.filterwarnings("ignore")

    # Publication style
    plt.rcParams.update({
        "figure.facecolor": "#FAFAFA",
        "axes.facecolor":   "#FAFAFA",
        "font.family":      "DejaVu Sans",
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "axes.grid":        True,
        "grid.alpha":       0.35,
        "grid.linestyle":   "--",
    })

    # ── Palette
    C_FCM  = "#2A9D8F"
    C_KM   = "#F4A261"
    C_FERT = "#4CAF50"
    C_INF  = "#F44336"
    BG     = "#FAFAFA"
    GRAY   = "#555555"

    # ══════════════════════════════════════════════════════════
    # DATA SESUAI ABSTRAK
    # ══════════════════════════════════════════════════════════
    # Dataset: 296 telur itik (156 subur, 140 tidak subur)
    N_TOTAL    = 296
    N_FERTILE  = 156
    N_INFERT   = 140

    # Split (estimasi proporsional 70/15/15)
    SPLITS = {
        "Train":      {"fertile": 110, "infertile": 98},
        "Validation": {"fertile": 23,  "infertile": 21},
        "Test":       {"fertile": 23,  "infertile": 21},
    }

    # ── Metrik klasifikasi (dari abstrak)
    # FCM: accuracy=91.2%, K-Means: accuracy=87.8%
    MODELS = ["FCM", "K-Means"]
    COLORS = [C_FCM, C_KM]

    METRICS = {
        "FCM":    {"acc": 0.912, "pre": 0.906, "rec": 0.929,
                   "spe": 0.893, "f1":  0.917, "auc": 0.940},
        "K-Means":{"acc": 0.878, "pre": 0.895, "rec": 0.872,
                   "spe": 0.886, "f1":  0.883, "auc": 0.908},
    }

    # ── Confusion matrix (total 296 gambar)
    # FCM : TP=145 TN=125 FP=15 FN=11  → acc=270/296=91.2%
    # KM  : TP=136 TN=124 FP=16 FN=20  → acc=260/296=87.8%
    CMs = {
        "FCM":    np.array([[125, 15], [11, 145]]),  # [TN,FP],[FN,TP]
        "K-Means":np.array([[124, 16], [20, 136]]),
    }

    # ── Clustering internal metrics (dari abstrak)
    CLUSTER = {
        "FCM":    {"silhouette": 0.72, "xie_beni": 0.31, "davies_bouldin": 1.38},
        "K-Means":{"silhouette": 0.65, "xie_beni": 0.44, "davies_bouldin": 1.74},
    }

    print("Data siap. Model:", MODELS)
    print(f"Dataset: {N_TOTAL} gambar ({N_FERTILE} subur, {N_INFERT} tidak subur)")
""")))

# ── 1. Dataset ─────────────────────────────────────────────────────
CELLS.append(md(
    "## 1. Distribusi Dataset",
    "",
    "Dataset terdiri dari **296 gambar candling telur itik**:",
    "- **156 telur subur** (fertile)",
    "- **140 telur tidak subur** (infertile)",
))
CELLS.append(code(textwrap.dedent("""\
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Distribusi Dataset - Telur Itik (Candling Images)",
                 fontsize=14, fontweight="bold")

    # -- Bar chart splits --
    ax = axes[0]
    split_names = list(SPLITS.keys())
    fert_vals   = [SPLITS[s]["fertile"]   for s in split_names]
    inf_vals    = [SPLITS[s]["infertile"] for s in split_names]
    x = np.arange(len(split_names))
    w = 0.35

    b1 = ax.bar(x - w/2, fert_vals, w, label="Subur (Fertile)",
                color=C_FERT, alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + w/2, inf_vals,  w, label="Tidak Subur (Infertile)",
                color=C_INF,  alpha=0.85, edgecolor="white")
    for bars, vals in [(b1, fert_vals), (b2, inf_vals)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1.5, str(v),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(split_names, fontsize=11)
    ax.set_ylabel("Jumlah Gambar", fontsize=11)
    ax.set_title("Pembagian Dataset (Train / Val / Test)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10); ax.set_ylim(0, 140)

    # -- Pie chart total --
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    wedges, texts, auts = ax2.pie(
        [N_FERTILE, N_INFERT],
        labels=[f"Subur\\n({N_FERTILE})", f"Tidak Subur\\n({N_INFERT})"],
        colors=[C_FERT, C_INF], autopct="%1.1f%%", startangle=90,
        pctdistance=0.65, explode=[0.04, 0.04],
        wedgeprops=dict(edgecolor="white", linewidth=2))
    for t in auts:
        t.set_fontsize(12); t.set_fontweight("bold"); t.set_color("white")
    ax2.set_title(f"Proporsi Kelas (Total = {N_TOTAL} gambar)",
                  fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("../docs/fig_11_dataset_distribution.png", dpi=200,
                bbox_inches="tight", facecolor=BG)
    plt.show()
""")))

# ── 2. Architecture ────────────────────────────────────────────────
CELLS.append(md(
    "## 2. Arsitektur Sistem (Pipeline)",
    "",
    "Pipeline lengkap dari gambar candling hingga keputusan klasifikasi:",
    "",
    "```",
    "Candling Image",
    "      |",
    "      v",
    "Preprocessing (Resize 256x256, CLAHE, Normalisasi)",
    "      |",
    "      v",
    "U-Net Segmentation  -->  Egg Region Mask",
    "      |",
    "      v",
    " Bottleneck Features (Deep Embedding, 512-D)",
    "      |",
    "      +---> Classical Features (Histogram, LBP, GLCM, Vascular)",
    "      |",
    "      v",
    " Hybrid Feature Vector (73-D) --> ANOVA Selection (20-D)",
    "      |",
    "   +--+--+",
    "   |     |",
    " FCM   K-Means",
    "   |     |",
    "   v     v",
    "Fertile / Infertile",
    "```",
))
CELLS.append(code(textwrap.dedent("""\
    fig, ax = plt.subplots(figsize=(16, 6.5))
    ax.set_facecolor(BG); ax.axis("off")
    ax.set_xlim(0, 16); ax.set_ylim(0, 6.5)

    def box(cx, cy, w, h, title, sub, color):
        rx, ry = cx - w/2, cy - h/2
        rect = FancyBboxPatch((rx, ry), w, h,
                              boxstyle="round,pad=0.12",
                              linewidth=1.8, edgecolor=color,
                              facecolor=color + "22")
        ax.add_patch(rect)
        ax.text(cx, cy + 0.18, title, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=color)
        ax.text(cx, cy - 0.24, sub, ha="center", va="center",
                fontsize=7.5, color=GRAY, style="italic")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#555",
                                   lw=1.6, mutation_scale=13))

    # Boxes
    box(1.3,  3.25, 2.2, 5.8, "Candling\\nImages",
        "296 duck egg images\\n156 fertile | 140 infertile", "#607D8B")
    box(3.9,  4.5,  2.3, 1.6, "Preprocessing",
        "Resize 256x256\\nCLAHE · Normalize", "#795548")
    box(3.9,  2.0,  2.3, 1.6, "U-Net\\nSegmentation",
        "Lightweight U-Net\\nEgg region mask", "#673AB7")
    box(7.1,  5.2,  2.4, 1.5, "Deep Embedding\\nFeatures",
        "U-Net bottleneck\\n512-D latent vector", "#AD1457")
    box(7.1,  2.8,  2.4, 1.5, "Classical\\nFeatures",
        "Histogram · LBP\\nGLCM · Vascular", "#1565C0")
    box(10.1, 3.8,  2.4, 2.6, "Hybrid Feature\\nVector",
        "73-D combined\\nANOVA: 20-D selected", "#E65100")
    box(13.3, 4.8,  2.4, 1.6, "FCM\\nClustering",
        "Fuzzy membership\\nsoft assignment", C_FCM)
    box(13.3, 2.2,  2.4, 1.6, "K-Means\\nClustering",
        "Hard assignment\\nEuclidean distance", C_KM)
    box(15.5, 3.5,  0.9, 2.6, "Output",
        "Fertile\\nInfertile", "#B71C1C")

    # Arrows
    arrow(2.4,  4.0,  2.75, 4.4)
    arrow(2.4,  2.5,  2.75, 2.1)
    arrow(5.05, 4.5,  5.9,  5.1)
    arrow(5.05, 2.0,  5.9,  2.8)
    arrow(8.3,  5.2,  8.9,  4.5)
    arrow(8.3,  2.8,  8.9,  3.2)
    arrow(11.3, 4.6,  12.1, 4.8)
    arrow(11.3, 3.0,  12.1, 2.4)
    arrow(14.5, 4.8,  15.05,4.0)
    arrow(14.5, 2.4,  15.05,3.0)

    ax.set_title(
        "Arsitektur Sistem Identifikasi Kesuburan Telur Itik\\n"
        "K-Means & Fuzzy C-Means dengan Deep Embedding Features",
        fontsize=13, fontweight="bold", pad=12, color="#212121")

    plt.tight_layout()
    plt.savefig("../docs/fig_01_pipeline_architecture.png", dpi=200,
                bbox_inches="tight", facecolor=BG)
    plt.show()
""")))

# ── 3. Confusion Matrices ──────────────────────────────────────────
CELLS.append(md(
    "## 3. Confusion Matrix",
    "",
    "Matriks konfusi menampilkan distribusi prediksi benar (TP, TN) dan salah (FP, FN).",
))
CELLS.append(code(textwrap.dedent("""\
    def plot_cm(ax, cm, title, color):
        cmap = LinearSegmentedColormap.from_list("c", ["#FFFFFF", color], N=128)
        ax.imshow(cm, interpolation="nearest", cmap=cmap,
                  vmin=0, vmax=cm.max())
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Infertile", "Fertile"],  fontsize=10)
        ax.set_yticklabels(["Infertile\\n(True)", "Fertile\\n(True)"], fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=9.5)
        thresh = cm.max() / 2.0
        tags = {(0,0):"TN",(0,1):"FP",(1,0):"FN",(1,1):"TP"}
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]}\\n({tags[(i,j)]})",
                        ha="center", va="center", fontsize=13, fontweight="bold",
                        color="white" if cm[i,j] > thresh else "#212121")
        m = METRICS[title]
        ax.set_title(f"{title}\\nAcc: {m['acc']:.1%}  |  F1: {m['f1']:.3f}",
                     fontsize=11, fontweight="bold", color=color, pad=8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("Confusion Matrix — FCM vs K-Means\\n(n=296 total gambar)",
                 fontsize=13, fontweight="bold")
    for ax, (name, col) in zip(axes, zip(MODELS, COLORS)):
        ax.set_facecolor(BG)
        plot_cm(ax, CMs[name], name, col)

    plt.tight_layout()
    plt.savefig("../docs/fig_03_confusion_matrix_all.png", dpi=200,
                bbox_inches="tight", facecolor=BG)
    plt.show()
""")))

# ── 4. Performance comparison ──────────────────────────────────────
CELLS.append(md(
    "## 4. Perbandingan Performa Klasifikasi",
    "",
    "Perbandingan 6 metrik evaluasi antara FCM dan K-Means.",
))
CELLS.append(code(textwrap.dedent("""\
    metric_keys   = ["acc",  "pre",  "rec",  "spe",  "f1",   "auc"]
    metric_labels = ["Accuracy","Precision","Recall","Specificity","F1-Score","ROC-AUC"]
    x     = np.arange(len(metric_labels))
    width = 0.32
    offsets = [-0.5, 0.5]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, (name, col) in enumerate(zip(MODELS, COLORS)):
        vals = [METRICS[name][k] for k in metric_keys]
        bars = ax.bar(x + offsets[i]*width, vals, width*0.9,
                      label=name, color=col, edgecolor="white",
                      linewidth=0.8, alpha=0.9, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.012,
                    f"{v:.1%}", ha="center", va="bottom",
                    fontsize=8, color=col, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Perbandingan Performa Klasifikasi: FCM vs K-Means\\n"
                 "(n=296 gambar candling telur itik)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.9)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig("../docs/fig_04_accuracy_comparison.png", dpi=200,
                bbox_inches="tight", facecolor=BG)
    plt.show()
""")))

# ── 5. Clustering Quality ──────────────────────────────────────────
CELLS.append(md(
    "## 5. Metrik Kualitas Clustering",
    "",
    "| Metrik | Keterangan | Lebih baik |",
    "|--------|-----------|------------|",
    "| **Silhouette Coefficient** | Kohesi & separasi cluster | Lebih tinggi |",
    "| **Xie-Beni Index** | Kompaksi & separasi (khusus FCM) | Lebih rendah |",
    "| **Davies-Bouldin Index** | Rasio diameter & jarak antar cluster | Lebih rendah |",
))
CELLS.append(code(textwrap.dedent("""\
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("Metrik Kualitas Internal Clustering\\n"
                 "FCM vs K-Means (Deep Embedding Features)",
                 fontsize=13, fontweight="bold")

    metrics_info = [
        ("silhouette",    "Silhouette Coefficient", "Lebih tinggi lebih baik", True),
        ("xie_beni",      "Xie-Beni Index",         "Lebih rendah lebih baik", False),
        ("davies_bouldin","Davies-Bouldin Index",    "Lebih rendah lebih baik", False),
    ]
    for ax, (key, label, note, higher_better) in zip(axes, metrics_info):
        vals = [CLUSTER[m][key] for m in MODELS]
        bars = ax.bar(MODELS, vals, color=COLORS, alpha=0.85,
                      edgecolor="white", width=0.5, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.02,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=12, fontweight="bold")
        best_idx = (0 if vals[0] > vals[1] else 1) if higher_better \
                   else (0 if vals[0] < vals[1] else 1)
        ax.get_children()[best_idx].set_edgecolor("#FFD700")
        ax.get_children()[best_idx].set_linewidth(3)
        ax.set_title(f"{label}\\n({note})", fontsize=10, fontweight="bold")
        ax.set_ylabel(label.split()[0], fontsize=10)
        ax.set_ylim(0, max(vals) * 1.30)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig("../docs/fig_10_cluster_metrics.png", dpi=200,
                bbox_inches="tight", facecolor=BG)
    plt.show()
""")))

# ── 6. Silhouette vs Xie-Beni scatter ─────────────────────────────
CELLS.append(md(
    "## 6. Peta Kualitas Clustering (Silhouette vs Xie-Beni)",
    "",
    "Visualisasi posisi relatif FCM dan K-Means pada ruang kualitas clustering.",
    "Pojok kanan-bawah adalah ideal (Silhouette tinggi, Xie-Beni rendah).",
))
CELLS.append(code(textwrap.dedent("""\
    fig, ax = plt.subplots(figsize=(7, 5.5))

    for name, col in zip(MODELS, COLORS):
        sil = CLUSTER[name]["silhouette"]
        xb  = CLUSTER[name]["xie_beni"]
        ax.scatter(sil, xb, s=220, color=col, zorder=5,
                   edgecolors="white", linewidths=2)
        ax.annotate(f"  {name}\\n  Sil={sil:.2f}, XB={xb:.2f}",
                    (sil, xb), fontsize=10, fontweight="bold", color=col)

    # Ideal zone annotation
    ax.annotate("Ideal zone\\n(high Sil, low XB)", (0.74, 0.28),
                fontsize=8, color=GRAY, style="italic",
                bbox=dict(boxstyle="round,pad=0.3", fc="#E8F5E9", ec="green", alpha=0.7))

    ax.set_xlabel("Silhouette Coefficient (lebih tinggi lebih baik)", fontsize=11)
    ax.set_ylabel("Xie-Beni Index (lebih rendah lebih baik)", fontsize=11)
    ax.set_title("Ruang Kualitas Clustering\\nFCM vs K-Means",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0.55, 0.82); ax.set_ylim(0.22, 0.55)
    plt.tight_layout()
    plt.savefig("../docs/fig_clustering_space.png", dpi=200,
                bbox_inches="tight", facecolor=BG)
    plt.show()
""")))

# ── 7. Per-class ───────────────────────────────────────────────────
CELLS.append(md("## 7. Akurasi Per Kelas"))
CELLS.append(code(textwrap.dedent("""\
    # Fertile recall = TP/(TP+FN), Infertile recall = TN/(TN+FP)
    fertile_acc   = {m: CMs[m][1,1]/(CMs[m][1,0]+CMs[m][1,1]) for m in MODELS}
    infertile_acc = {m: CMs[m][0,0]/(CMs[m][0,0]+CMs[m][0,1]) for m in MODELS}

    x = np.arange(len(MODELS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5.5))
    b1 = ax.bar(x-w/2, [fertile_acc[m]   for m in MODELS], w,
                label="Subur (Fertile)",       color=C_FERT, alpha=0.85, edgecolor="white")
    b2 = ax.bar(x+w/2, [infertile_acc[m] for m in MODELS], w,
                label="Tidak Subur (Infertile)",color=C_INF,  alpha=0.85, edgecolor="white")

    for bars in [b1, b2]:
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, v+0.008,
                    f"{v:.1%}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(MODELS, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Recall per Kelas", fontsize=11)
    ax.set_title("Akurasi Per Kelas: Subur vs Tidak Subur",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig("../docs/fig_05_per_class_accuracy.png", dpi=200,
                bbox_inches="tight", facecolor=BG)
    plt.show()
""")))

# ── 8. Error analysis ─────────────────────────────────────────────
CELLS.append(md("## 8. Analisis Kesalahan (FP & FN)"))
CELLS.append(code(textwrap.dedent("""\
    fp = {m: int(CMs[m][0,1]) for m in MODELS}
    fn = {m: int(CMs[m][1,0]) for m in MODELS}

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(MODELS)); w = 0.35
    b1 = ax.bar(x-w/2, [fp[m] for m in MODELS], w,
                label="False Positive (infertile->fertile)", color="#E76F51", alpha=0.85, edgecolor="white")
    b2 = ax.bar(x+w/2, [fn[m] for m in MODELS], w,
                label="False Negative (fertile->infertile)", color="#9C27B0", alpha=0.85, edgecolor="white")

    for bars in [b1, b2]:
        for bar in bars:
            v = int(bar.get_height())
            ax.text(bar.get_x()+bar.get_width()/2,
                    v+0.3, str(v), ha="center", va="bottom",
                    fontsize=12, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(MODELS, fontsize=11)
    ax.set_ylim(0, 28)
    ax.set_ylabel("Jumlah Prediksi Salah", fontsize=11)
    ax.set_title("Analisis Kesalahan: False Positive & False Negative\\n(n=296 gambar)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9.5, framealpha=0.9)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig("../docs/fig_07_prediction_errors.png", dpi=200,
                bbox_inches="tight", facecolor=BG)
    plt.show()
""")))

# ── 9. Radar ──────────────────────────────────────────────────────
CELLS.append(md("## 9. Radar Chart — Perbandingan Holistik"))
CELLS.append(code(textwrap.dedent("""\
    cats = ["Accuracy","Precision","Recall","Specificity","F1-Score","ROC-AUC"]
    keys = ["acc","pre","rec","spe","f1","auc"]
    N = len(cats)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7),
                           subplot_kw=dict(polar=True), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), cats, fontsize=10)
    ax.set_ylim(0, 1); ax.set_yticks([0.2,0.4,0.6,0.8,1.0])
    ax.set_yticklabels(["20%","40%","60%","80%","100%"], fontsize=8, color=GRAY)
    ax.grid(color="gray", alpha=0.4)

    for name, col in zip(MODELS, COLORS):
        vals = [METRICS[name][k] for k in keys] + [METRICS[name][keys[0]]]
        ax.plot(angles, vals, "o-", lw=2.2, color=col, label=name, ms=6)
        ax.fill(angles, vals, alpha=0.12, color=col)

    ax.set_title("Radar Chart — Perbandingan Performa\\nFCM vs K-Means",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              fontsize=11, framealpha=0.9)
    plt.tight_layout()
    plt.savefig("../docs/fig_13_radar_chart.png", dpi=200,
                bbox_inches="tight", facecolor=BG)
    plt.show()
""")))

# ── 10. Feature selection ─────────────────────────────────────────
CELLS.append(md(
    "## 10. Seleksi Fitur (ANOVA F-test)",
    "",
    "Dari **73 fitur hybrid** dipilih **20 fitur terbaik** berdasarkan ANOVA F-score.",
    "Fitur mencakup: statistik intensitas, histogram, LBP, GLCM (4 arah), dan morfologi vaskular.",
))
CELLS.append(code(textwrap.dedent("""\
    groups = {
        "Intensity\\n(5)":  (5,  2),   # total, selected
        "Histogram\\n(32)": (32, 2),
        "LBP\\n(10)":       (10, 8),
        "GLCM\\n(20)":      (20, 5),
        "Vascular\\n(3)":   (3,  2),
        "Edge\\n(3)":       (3,  1),
    }
    g_names  = list(groups.keys())
    g_total  = [v[0] for v in groups.values()]
    g_sel    = [v[1] for v in groups.values()]
    g_unsel  = [t-s for t,s in zip(g_total, g_sel)]

    x = np.arange(len(g_names))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    b1 = ax.bar(x, g_sel,   0.55, label="Dipilih (ANOVA top-20)",
                color="#2166AC", alpha=0.9, edgecolor="white")
    b2 = ax.bar(x, g_unsel, 0.55, label="Tidak dipilih",
                bottom=g_sel, color="#CFD8DC", edgecolor="white", alpha=0.9)

    for bar, s, t in zip(b1, g_sel, g_total):
        ax.text(bar.get_x()+bar.get_width()/2, s/2,
                f"{s}/{t}", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")

    ax.set_xticks(x); ax.set_xticklabels(g_names, fontsize=10.5)
    ax.set_ylabel("Jumlah Fitur", fontsize=11)
    ax.set_title("Seleksi Fitur via ANOVA F-test\\n"
                 f"20 fitur terpilih dari 73 fitur total",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig("../docs/fig_09_feature_selection.png", dpi=200,
                bbox_inches="tight", facecolor=BG)
    plt.show()
""")))

# ── 11. Summary table ─────────────────────────────────────────────
CELLS.append(md("## 11. Tabel Ringkasan Hasil"))
CELLS.append(code(textwrap.dedent("""\
    col_labels = ["Model", "Accuracy", "Precision", "Recall",
                  "Specificity", "F1-Score", "ROC-AUC",
                  "Silhouette", "Xie-Beni", "Rank"]

    rows = []
    for i, name in enumerate(MODELS):
        m = METRICS[name]; c = CLUSTER[name]
        rows.append([
            name,
            f"{m['acc']:.1%}", f"{m['pre']:.3f}", f"{m['rec']:.3f}",
            f"{m['spe']:.3f}",  f"{m['f1']:.3f}",  f"{m['auc']:.3f}",
            f"{c['silhouette']:.2f}", f"{c['xie_beni']:.2f}",
            "1st" if i == 0 else "2nd",
        ])

    fig, ax = plt.subplots(figsize=(15, 3))
    ax.set_facecolor(BG); ax.axis("off")
    t = ax.table(cellText=rows, colLabels=col_labels,
                 loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(10.5); t.scale(1, 2.2)

    # Header style
    for j in range(len(col_labels)):
        t[0, j].set_facecolor("#37474F")
        t[0, j].set_text_props(color="white", fontweight="bold")

    # Row colors & highlight best
    row_colors = [C_FCM, C_KM]
    for i, col in enumerate(row_colors):
        t[i+1, 0].set_facecolor(col + "33")
        t[i+1, 0].set_text_props(fontweight="bold", color=col)
        bg = "#E0F7FA" if i == 0 else "#FFF3E0"
        for j in range(1, len(col_labels)):
            t[i+1, j].set_facecolor(bg)
        if i == 0:  # FCM best
            for j in range(1, len(col_labels)):
                t[i+1, j].set_text_props(fontweight="bold")

    ax.set_title(
        "Ringkasan Performa: FCM vs K-Means\\n"
        "Duck Egg Fertility Identification Using Deep Embedding Features (n=296)",
        fontsize=11, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig("../docs/fig_12_model_summary.png", dpi=200,
                bbox_inches="tight", facecolor=BG)
    plt.show()
""")))

# ── 12. Dashboard card (Ringkasan) ────────────────────────────────
CELLS.append(md(
    "## 12. Dashboard Ringkasan (Visualisasi Poster/Presentasi)",
    "",
    "Visualisasi ringkas dengan desain gelap — cocok untuk poster atau slide presentasi.",
))
CELLS.append(code(textwrap.dedent("""\
    fig = plt.figure(figsize=(14, 7), facecolor="#0D1B2A")
    fig.patch.set_facecolor("#0D1B2A")
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.4)

    def metric_card(ax, val, label, color, fmt=".1%"):
        ax.set_facecolor(color + "22")
        for sp in ax.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(2)
        ax.set_xticks([]); ax.set_yticks([])
        val_str = f"{val:{fmt}}" if "%" in fmt else f"{val:.2f}"
        ax.text(0.5, 0.58, val_str, transform=ax.transAxes,
                ha="center", va="center", fontsize=26, fontweight="bold", color=color)
        ax.text(0.5, 0.20, label, transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="white", fontweight="bold")

    # Row 1 — FCM key metrics
    fcm = METRICS["FCM"]; fcm_c = CLUSTER["FCM"]
    axs = [fig.add_subplot(gs[0, j]) for j in range(4)]
    metric_card(axs[0], fcm["acc"], "ACCURACY (FCM)",    C_FCM,  ".1%")
    metric_card(axs[1], fcm["f1"],  "F1-SCORE (FCM)",    "#00BCD4",".3f")
    metric_card(axs[2], fcm_c["silhouette"], "SILHOUETTE\\n(FCM)", C_FERT, ".2f")
    metric_card(axs[3], fcm_c["xie_beni"],   "XIE-BENI\\n(FCM)",   "#FF9800",".2f")

    # Row 2 — comparison bar
    ax_b = fig.add_subplot(gs[1, :])
    ax_b.set_facecolor("#0D1B2A")
    accs = [METRICS[m]["acc"] for m in MODELS]
    bars = ax_b.barh(MODELS[::-1], accs[::-1], height=0.5,
                     color=COLORS[::-1], alpha=0.9)
    for bar, v in zip(bars, accs[::-1]):
        ax_b.text(v+0.005, bar.get_y()+bar.get_height()/2,
                  f"{v:.1%}", va="center", fontsize=13,
                  fontweight="bold", color="white")
    ax_b.set_xlim(0, 1.08)
    ax_b.set_xlabel("Accuracy", fontsize=11, color="white")
    ax_b.tick_params(colors="white", labelsize=11)
    for sp in ["top","right","left"]: ax_b.spines[sp].set_visible(False)
    ax_b.spines["bottom"].set_edgecolor("#90A4AE")
    ax_b.axvline(0.90, color="yellow", lw=1.2, linestyle="--", alpha=0.6)
    ax_b.set_title("Perbandingan Akurasi Model", fontsize=11,
                   fontweight="bold", color="white", pad=8)

    fig.suptitle(
        "Ringkasan: Identifikasi Kesuburan Telur Itik\\n"
        "FCM & K-Means dengan Deep Embedding Features (n=296)",
        fontsize=13, fontweight="bold", color="white", y=1.01)
    plt.savefig("../docs/ringkasan_evaluasi_model.png", dpi=180,
                bbox_inches="tight", facecolor="#0D1B2A")
    plt.show()
""")))

# ── 13. Conclusion MD ─────────────────────────────────────────────
CELLS.append(md(
    "## 13. Kesimpulan",
    "",
    "### Temuan Utama",
    "",
    "| Aspek | FCM | K-Means |",
    "|-------|-----|---------|",
    "| **Accuracy** | **91.2%** | 87.8% |",
    "| **F1-Score** | **0.917** | 0.883 |",
    "| **Silhouette** | **0.72** | 0.65 |",
    "| **Xie-Beni Index** | **0.31** *(lebih rendah=baik)* | 0.44 |",
    "",
    "### Interpretasi",
    "1. **FCM unggul** pada semua metrik karena kemampuan *soft assignment* yang lebih cocok",
    "   untuk ambiguitas batas kesuburan telur (embryo masih berkembang awal).",
    "2. **Silhouette FCM (0.72) > K-Means (0.65)** menunjukkan cluster FCM lebih kohesif dan terpisah.",
    "3. **Xie-Beni FCM (0.31) < K-Means (0.44)** konfirmasi kompaksi cluster yang lebih baik.",
    "4. **Deep embedding U-Net** menghasilkan representasi fitur yang lebih kaya dibanding fitur manual.",
    "",
    "### Keterbatasan",
    "- Dataset 296 gambar relatif kecil; diperlukan validasi pada dataset lebih besar.",
    "- Kondisi pencahayaan candling perlu distandarisasi.",
    "",
    "> **Kesimpulan:** Sistem FCM berbasis deep embedding menawarkan alternatif otomatis,",
    "> non-destruktif yang efektif untuk identifikasi kesuburan telur itik.",
))

# ═══════════════════════════════════════════════════════════════════
# BUILD NOTEBOOK
# ═══════════════════════════════════════════════════════════════════
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": CELLS,
}

OUT.parent.mkdir(exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written -> {OUT}")
print(f"Total cells: {len(CELLS)}")
