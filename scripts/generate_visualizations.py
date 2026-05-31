"""
Generate separate visualization figures for the duck egg fertility detection project.
Saves each figure as an individual PNG in docs/.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})

def save(fig, name):
    path = DOCS / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved -> {path.name}")

def load_predictions():
    return pd.read_csv(ROOT / "results/evaluation/full_evaluation_predictions.csv")

def compute_cm(df, model):
    classes = ["fertile", "infertile"]
    cm = np.zeros((2, 2), dtype=int)
    for _, row in df.iterrows():
        ti = classes.index(row["true_label"])
        pi = classes.index(row[model])
        cm[ti][pi] += 1
    return cm

def per_class_acc(cm):
    f = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
    i = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0
    return f, i

def fig_pipeline():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 5); ax.axis("off")
    fig.patch.set_facecolor("#0d0d0d"); ax.set_facecolor("#0d0d0d")
    blocks = [
        (1.0,  "Input\nTelur Bebek",  "224x224 px"),
        (3.2,  "Preprocessing\nCLAHE", "Contrast\nEnhancement"),
        (5.4,  "Segmentasi\nU-Net",    "Mask\nVaskular"),
        (7.6,  "Ekstraksi\nFitur",     "73 Fitur\n(GLCM, LBP,\nVaskular)"),
        (9.8,  "Seleksi\nFitur",       "20 Fitur\nTerpilih"),
        (12.0, "Klasifikasi\nAWC",     "Fertil /\nInfertil"),
    ]
    box_w, box_h, box_y = 1.7, 1.6, 1.7
    for x, title, sub in blocks:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - box_w/2, box_y), box_w, box_h, boxstyle="round,pad=0.08",
            facecolor="#1e1e1e", edgecolor="white", linewidth=1.5))
        ax.text(x, box_y+box_h*0.65, title, ha="center", va="center",
                fontsize=9, color="white", fontweight="bold", linespacing=1.3)
        ax.text(x, box_y+box_h*0.22, sub, ha="center", va="center",
                fontsize=7.5, color="#aaaaaa", linespacing=1.3)
    for i in range(len(blocks)-1):
        x0 = blocks[i][0]+box_w/2; x1 = blocks[i+1][0]-box_w/2
        ax.annotate("", xy=(x1, box_y+box_h/2), xytext=(x0, box_y+box_h/2),
                    arrowprops=dict(arrowstyle="-|>", color="white", lw=1.5))
    last_x = blocks[-1][0]
    ax.text(last_x, box_y-0.25, "Fertil",   ha="center", color="#4CAF50", fontsize=9, fontweight="bold")
    ax.text(last_x, box_y-0.65, "Infertil", ha="center", color="#F44336", fontsize=9, fontweight="bold")
    fig.suptitle("Arsitektur Pipeline - Deteksi Kesuburan Telur Bebek",
                 color="white", fontsize=14, fontweight="bold", y=0.97)
    save(fig, "fig_01_pipeline_architecture.png")

def fig_cm_awc(df):
    cm = compute_cm(df, "AWC"); labels = ["Fertil", "Infertil"]
    acc = np.trace(cm)/cm.sum()
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Label Aktual")
    ax.set_title(f"Confusion Matrix - AWC\nAkurasi: {acc:.2%}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j]>cm.max()/2 else "black",
                    fontsize=18, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); save(fig, "fig_02_confusion_matrix_awc.png")

def fig_cm_all(df):
    models = ["AWC", "KMeans", "FCM"]; labels = ["Fertil", "Infertil"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Confusion Matrix - Semua Model (Test + Val, n=68)", fontsize=14, fontweight="bold")
    for ax, model in zip(axes, models):
        cm = compute_cm(df, model); acc = np.trace(cm)/cm.sum()
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        ax.set_xlabel("Prediksi"); ax.set_ylabel("Label Aktual")
        ax.set_title(f"{model}\nAkurasi: {acc:.2%}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                        color="white" if cm[i,j]>cm.max()/2 else "black",
                        fontsize=16, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); save(fig, "fig_03_confusion_matrix_all.png")

def fig_accuracy_comparison():
    with open(ROOT/"results/statistical_tests/eval_summary.json") as f:
        summary = json.load(f)
    models = list(summary["model_accuracy"].keys())
    accs = [summary["model_accuracy"][m] for m in models]
    colors = ["#2196F3", "#FF9800", "#9C27B0"]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(models, [a*100 for a in accs], color=colors, width=0.5, edgecolor="white")
    ax.set_ylim(0, 100); ax.set_ylabel("Akurasi (%)")
    ax.set_title("Perbandingan Akurasi Model\n(Test + Val Split, n=68)")
    ax.axhline(80, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="80%")
    ax.legend(fontsize=10)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                f"{acc*100:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")
    fig.tight_layout(); save(fig, "fig_04_accuracy_comparison.png")

def fig_per_class_accuracy(df):
    models = ["AWC", "KMeans", "FCM"]
    f_accs, i_accs = [], []
    for m in models:
        cm = compute_cm(df, m); f, i = per_class_acc(cm)
        f_accs.append(f*100); i_accs.append(i*100)
    x = np.arange(len(models)); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x-w/2, f_accs, w, label="Fertil", color="#4CAF50", edgecolor="white")
    b2 = ax.bar(x+w/2, i_accs, w, label="Infertil", color="#F44336", edgecolor="white")
    for bar in list(b1)+list(b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=12)
    ax.set_ylim(0, 115); ax.set_ylabel("Akurasi per Kelas (%)")
    ax.set_title("Akurasi per Kelas - Semua Model\n(Test + Val Split, n=68)")
    ax.axhline(80, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.legend(fontsize=11); fig.tight_layout()
    save(fig, "fig_05_per_class_accuracy.png")

def _load_image(img_name, true_label, split):
    base = ROOT/"data"/split/true_label/img_name
    if base.exists():
        try:
            import cv2
            img = cv2.imread(str(base))
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            pass
    return None

def fig_correct_predictions(df, model="AWC", n=8):
    correct = df[df["true_label"]==df[model]].copy()
    fert = correct[correct["true_label"]=="fertile"].head(n//2)
    inf  = correct[correct["true_label"]=="infertile"].head(n//2)
    sample = pd.concat([fert, inf]).reset_index(drop=True)
    cols = 4; rows = (len(sample)+cols-1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.2, rows*3.5))
    axes = np.array(axes).flatten()
    fig.suptitle(f"Contoh Prediksi Benar - {model}", fontsize=14, fontweight="bold")
    last_i = 0
    for i, (_, row) in enumerate(sample.iterrows()):
        img = _load_image(row["image"], row["true_label"], row["split"])
        ax = axes[i]
        if img is not None: ax.imshow(img)
        else:
            ax.set_facecolor("#cccccc")
            ax.text(0.5, 0.5, "No Image", ha="center", va="center", transform=ax.transAxes)
        color   = "#4CAF50" if row["true_label"]=="fertile" else "#F44336"
        caption = "FERTIL"  if row["true_label"]=="fertile" else "INFERTIL"
        ax.set_title(caption, color=color, fontsize=10, fontweight="bold")
        ax.axis("off"); last_i = i
    for j in range(last_i+1, len(axes)): axes[j].axis("off")
    fig.tight_layout(); save(fig, "fig_06_prediction_correct.png")

def fig_error_predictions(df, model="AWC"):
    errors = df[df["true_label"]!=df[model]].copy().reset_index(drop=True)
    if len(errors)==0: print("  no errors to show"); return
    n = min(len(errors), 8); sample = errors.head(n)
    cols = 4; rows = (n+cols-1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.2, rows*3.8))
    axes = np.array(axes).flatten()
    fig.suptitle(f"Contoh Kesalahan Klasifikasi - {model}", fontsize=14, fontweight="bold")
    last_i = 0
    for i, (_, row) in enumerate(sample.iterrows()):
        img = _load_image(row["image"], row["true_label"], row["split"])
        ax = axes[i]
        if img is not None: ax.imshow(img)
        else:
            ax.set_facecolor("#ffcccc")
            ax.text(0.5, 0.5, "No Image", ha="center", va="center", transform=ax.transAxes)
        ts = "Fertil" if row["true_label"]=="fertile" else "Infertil"
        ps = "Fertil" if row[model]=="fertile" else "Infertil"
        ax.set_title(f"Aktual: {ts}\nPrediksi: {ps}", color="red", fontsize=9, fontweight="bold")
        ax.axis("off"); last_i = i
    for j in range(last_i+1, len(axes)): axes[j].axis("off")
    fig.tight_layout(); save(fig, "fig_07_prediction_errors.png")

def fig_statistical_tests():
    with open(ROOT/"results/statistical_tests/report_full.json") as f:
        report = json.load(f)
    pairs = [{"Pair": f"{e['model_a']} vs {e['model_b']}", "p": e["p_value"]}
             for e in report["mcnemar"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Uji Statistik Perbandingan Model", fontsize=14, fontweight="bold")
    ax = axes[0]
    pair_labels = [p["Pair"] for p in pairs]; pvals = [p["p"] for p in pairs]
    bar_colors = ["#F44336" if p<0.05 else "#9E9E9E" for p in pvals]
    bars = ax.barh(pair_labels, pvals, color=bar_colors, edgecolor="white")
    ax.axvline(0.05, color="red", linestyle="--", linewidth=1.5, label="alpha=0.05")
    ax.set_xlabel("p-value (McNemar)"); ax.set_title("p-value antar Model"); ax.legend(fontsize=10)
    for bar, pval in zip(bars, pvals):
        ax.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2,
                f"{pval:.4f}", va="center", fontsize=10)
    ax.set_xlim(0, max(pvals)*1.4)
    friedman = report["friedman"][0]
    models_r = list(friedman["average_ranks"].keys())
    ranks = [friedman["average_ranks"][m] for m in models_r]
    bar_cols = ["#2196F3", "#FF9800", "#9C27B0"]
    bars2 = axes[1].bar(models_r, ranks, color=bar_cols, width=0.5, edgecolor="white")
    for bar, rank in zip(bars2, ranks):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                     f"{rank:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Rata-rata Rank (Friedman)")
    interp = friedman["interpretation"].split("(")[0].strip()
    axes[1].set_title(f"Rank Friedman\n(p={friedman['p_value']:.4f}, {interp})")
    axes[1].set_ylim(0, max(ranks)*1.3)
    fig.tight_layout(); save(fig, "fig_08_statistical_tests.png")

def fig_feature_selection():
    with open(ROOT/"results/awc_evaluation/metrics.json") as f: metrics = json.load(f)
    with open(ROOT/"results/statistical_tests/eval_summary.json") as f: summary = json.load(f)
    all_features = summary["feature_names"]
    selected_idx = set(metrics["cluster_info"]["feature_indices"])
    groups = {
        "Statistik\nDasar":    list(range(0, 5)),
        "Histogram\n(32 bin)": list(range(5, 37)),
        "LBP\n(10 bin)":       list(range(37, 47)),
        "GLCM\n(4 arah)":      list(range(47, 67)),
        "Vaskular":             list(range(67, 70)),
        "Edge":                 list(range(70, 73)),
    }
    gnames = list(groups.keys())
    total_g = [len(v) for v in groups.values()]
    sel_g   = [sum(1 for i in v if i in selected_idx) for v in groups.values()]
    x = np.arange(len(gnames)); w = 0.4
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x-w/2, total_g, w, label="Total Fitur", color="#90CAF9", edgecolor="white")
    b2 = ax.bar(x+w/2, sel_g, w, label="Fitur Terpilih (AWC)", color="#1565C0", edgecolor="white")
    for bar in b2:
        if bar.get_height()>0:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                    str(int(bar.get_height())), ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="#1565C0")
    ax.set_xticks(x); ax.set_xticklabels(gnames, fontsize=10)
    ax.set_ylabel("Jumlah Fitur")
    ax.set_title(f"Seleksi Fitur AWC - {len(selected_idx)} dari {len(all_features)} Fitur Dipilih")
    ax.legend(fontsize=11); fig.tight_layout()
    save(fig, "fig_09_feature_selection.png")

def fig_cluster_metrics():
    with open(ROOT/"results/awc_evaluation/metrics.json") as f: metrics = json.load(f)
    ev = metrics["evaluation"]; ci = metrics["cluster_info"]
    metric_labels = ["Silhouette\n(Training)", "Silhouette\n(Eval)", "Adj. Rand\nIndex", "Norm. Mutual\nInfo"]
    values = [ci["silhouette_score"], ev["silhouette"], ev["adjusted_rand"], ev["normalized_mutual_info"]]
    colors = ["#42A5F5", "#26C6DA", "#66BB6A", "#AB47BC"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(metric_labels, values, color=colors, width=0.55, edgecolor="white")
    ax.set_ylim(0, 1.0); ax.set_ylabel("Nilai"); ax.set_title("Metrik Kualitas Klaster - AWC")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="0.5"); ax.legend(fontsize=10)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.015,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    fig.text(0.5, 0.01,
             f"Davies-Bouldin: {ev['davies_bouldin']:.3f}  |  Calinski-Harabasz: {ev['calinski_harabasz']:.3f}  |  Inertia: {ci['inertia']:.1f}",
             ha="center", fontsize=9, color="gray")
    fig.tight_layout(rect=[0, 0.06, 1, 1]); save(fig, "fig_10_cluster_metrics.png")

def fig_dataset_distribution():
    def count_jpg(subpath):
        p = ROOT/"data"/subpath
        return len(list(p.glob("*.jpg"))) if p.exists() else 0
    splits_data = {
        "Train (Fertil)":   count_jpg("train/fertile"),
        "Train (Infertil)": count_jpg("train/infertile"),
        "Val (Fertil)":     count_jpg("val/fertile"),
        "Val (Infertil)":   count_jpg("val/infertile"),
        "Test (Fertil)":    count_jpg("test/fertile"),
        "Test (Infertil)":  count_jpg("test/infertile"),
    }
    labels = list(splits_data.keys()); counts = list(splits_data.values())
    colors = ["#4CAF50", "#F44336"]*3
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Distribusi Dataset Telur Bebek", fontsize=14, fontweight="bold")
    ax = axes[0]
    bars = ax.bar(labels, counts, color=colors, edgecolor="white")
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                str(c), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Jumlah Gambar"); ax.set_title("Jumlah per Split dan Kelas")
    ax.legend(handles=[mpatches.Patch(color="#4CAF50", label="Fertil"),
                       mpatches.Patch(color="#F44336", label="Infertil")], fontsize=11)
    tf = splits_data["Train (Fertil)"]+splits_data["Val (Fertil)"]+splits_data["Test (Fertil)"]
    ti = splits_data["Train (Infertil)"]+splits_data["Val (Infertil)"]+splits_data["Test (Infertil)"]
    axes[1].pie([tf, ti], labels=["Fertil","Infertil"], colors=["#4CAF50","#F44336"],
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"edgecolor":"white","linewidth":1.5}, textprops={"fontsize":12})
    axes[1].set_title(f"Distribusi Keseluruhan\n(Total: {tf+ti} gambar)")
    fig.tight_layout(); save(fig, "fig_11_dataset_distribution.png")

def fig_model_summary(df):
    models = ["AWC", "KMeans", "FCM"]; colors = ["#2196F3", "#FF9800", "#9C27B0"]
    metrics_data = {}
    for m in models:
        cm = compute_cm(df, m)
        tp, fn = cm[0,0], cm[0,1]; fp, tn = cm[1,0], cm[1,1]
        acc = (tp+tn)/cm.sum()
        pre = tp/(tp+fp) if (tp+fp)>0 else 0
        rec = tp/(tp+fn) if (tp+fn)>0 else 0
        f1  = 2*pre*rec/(pre+rec) if (pre+rec)>0 else 0
        metrics_data[m] = {"Akurasi":acc, "Presisi":pre, "Recall":rec, "F1-Score":f1}
    metric_names = ["Akurasi", "Presisi", "Recall", "F1-Score"]
    x = np.arange(len(metric_names)); w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (model, col) in enumerate(zip(models, colors)):
        vals = [metrics_data[model][mn]*100 for mn in metric_names]
        bars = ax.bar(x+(i-1)*w, vals, w, label=model, color=col, edgecolor="white", linewidth=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color=col)
    ax.set_xticks(x); ax.set_xticklabels(metric_names, fontsize=12)
    ax.set_ylim(0, 108); ax.set_ylabel("Nilai (%)")
    ax.set_title("Ringkasan Metrik Evaluasi - Semua Model\n(Test + Val Split, n=68)")
    ax.axhline(80, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.legend(fontsize=11); fig.tight_layout()
    save(fig, "fig_12_model_summary.png")

def main():
    print("Generating visualizations ...")
    df = load_predictions()
    fig_pipeline()
    fig_cm_awc(df)
    fig_cm_all(df)
    fig_accuracy_comparison()
    fig_per_class_accuracy(df)
    fig_correct_predictions(df)
    fig_error_predictions(df)
    fig_statistical_tests()
    fig_feature_selection()
    fig_cluster_metrics()
    fig_dataset_distribution()
    fig_model_summary(df)
    print(f"\nDone. All 12 figures saved to docs/")

if __name__ == "__main__":
    main()