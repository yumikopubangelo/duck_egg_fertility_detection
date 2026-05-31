from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parent.parent

PALETTE = {
    "AWC": "#163A70",
    "KMeans": "#E28743",
    "FCM": "#2E9C6A",
    "MobileNetV3": "#A23EC6",
}


def load_results(input_path: Path) -> dict:
    with input_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def draw_metric_cards(fig: plt.Figure, best_model: str, best_metrics: dict, meta: dict) -> None:
    cards = [
        ("Model terbaik", best_model, PALETTE.get(best_model, "#163A70")),
        ("Accuracy", f"{best_metrics['accuracy'] * 100:.2f}%", "#1E88E5"),
        ("F1-score", f"{best_metrics['f1'] * 100:.2f}%", "#2E9C6A"),
        ("ROC-AUC", f"{best_metrics['roc_auc']:.4f}", "#F57C00"),
        ("Data uji", f"{meta['n_test']} sampel", "#6D4C41"),
    ]

    left_positions = [0.05, 0.235, 0.42, 0.605, 0.79]
    for (title, value, color), left in zip(cards, left_positions):
        fig.text(
            left,
            0.905,
            f"{title}\n{value}",
            ha="left",
            va="top",
            fontsize=13,
            color="white",
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": color,
                "edgecolor": "none",
                "alpha": 0.96,
            },
        )


def plot_metric_comparison(ax: plt.Axes, results: dict) -> None:
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    model_names = list(results.keys())
    x = np.arange(len(metric_keys))
    width = 0.18
    offsets = np.linspace(-(width * 1.5), width * 1.5, len(model_names))

    for offset, model_name in zip(offsets, model_names):
        values = [results[model_name]["metrics"][key] for key in metric_keys]
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=model_name,
            color=PALETTE.get(model_name, "#546E7A"),
            alpha=0.9,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title("Perbandingan Metrik Utama", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Skor")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper left")


def plot_f1_ranking(ax: plt.Axes, results: dict) -> None:
    ranked = sorted(
        ((name, payload["metrics"]["f1"]) for name, payload in results.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    names = [item[0] for item in ranked]
    values = [item[1] for item in ranked]
    colors = [PALETTE.get(name, "#90A4AE") for name in names]

    ax.barh(names, values, color=colors, alpha=0.92)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_title("Peringkat Model berdasarkan F1-score", fontsize=14, fontweight="bold")
    ax.set_xlabel("F1-score")
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for idx, value in enumerate(values):
        ax.text(value + 0.015, idx, f"{value:.3f}", va="center", fontsize=10, fontweight="bold")


def plot_best_confusion_matrix(ax: plt.Axes, model_name: str, payload: dict) -> None:
    y_true = np.array(payload["y_true"], dtype=int)
    y_pred = np.array(payload["y_pred"], dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_totals = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals != 0)

    image = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
    labels = ["Infertil", "Fertil"]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label asli")

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            value = cm[row, col]
            ratio = cm_norm[row, col]
            text_color = "white" if ratio > 0.5 else "#10233F"
            ax.text(
                col,
                row,
                f"{value}\n{ratio * 100:.1f}%",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=text_color,
            )


def plot_summary_text(ax: plt.Axes, meta: dict, best_model: str, best_metrics: dict, wilcoxon: dict) -> None:
    ax.axis("off")

    highlight_lines = [
        "Ringkasan eksperimen",
        f"- Run: {meta['timestamp']}",
        f"- Fitur total: {meta['n_features']}",
        f"- Fitur AWC terpilih: {meta.get('awc_selected_features', meta['n_features'])}",
        f"- Dataset uji seimbang: {meta['n_fertile']} fertil vs {meta['n_infertile']} infertil",
        "",
        "Temuan utama",
        f"- {best_model} mencapai accuracy {best_metrics['accuracy'] * 100:.2f}%",
        f"- F1-score tertinggi: {best_metrics['f1'] * 100:.2f}%",
        f"- ROC-AUC terbaik: {best_metrics['roc_auc']:.4f}",
        "",
        "Uji Wilcoxon",
    ]

    for pair_name, result in wilcoxon.items():
        p_value = result.get("p_value")
        p_text = "N/A" if p_value is None else f"{p_value:.4f}"
        status = "signifikan" if result.get("significant") else "tidak signifikan"
        better = result.get("better_model", "-")
        highlight_lines.append(f"- {pair_name}: p={p_text}, {status}, unggul {better}")

    ax.text(
        0.0,
        1.0,
        "\n".join(highlight_lines),
        va="top",
        ha="left",
        fontsize=11,
        linespacing=1.5,
        bbox={
            "boxstyle": "round,pad=0.7",
            "facecolor": "#F5F7FA",
            "edgecolor": "#D0D7DE",
        },
    )


def build_visualization(input_path: Path, output_prefix: Path) -> list[Path]:
    payload = load_results(input_path)
    results = payload["results"]
    meta = payload["meta"]
    wilcoxon = payload.get("wilcoxon", {})
    best_model = max(results, key=lambda name: results[name]["metrics"]["f1"])
    best_metrics = results[best_model]["metrics"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    grid = fig.add_gridspec(2, 2, left=0.05, right=0.97, bottom=0.08, top=0.78, hspace=0.28, wspace=0.18)

    ax_metrics = fig.add_subplot(grid[0, 0])
    ax_cm = fig.add_subplot(grid[0, 1])
    ax_ranking = fig.add_subplot(grid[1, 0])
    ax_summary = fig.add_subplot(grid[1, 1])

    fig.suptitle(
        "Ringkasan Evaluasi Model Fertilitas Telur Bebek",
        fontsize=24,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.05,
        0.835,
        "Sumber: results/evaluation/full_vascular_glcm_mobilenet/full_evaluation_20260427_004032.json",
        fontsize=10,
        color="#586069",
    )

    draw_metric_cards(fig, best_model, best_metrics, meta)
    plot_metric_comparison(ax_metrics, results)
    plot_best_confusion_matrix(ax_cm, best_model, results[best_model])
    plot_f1_ranking(ax_ranking, results)
    plot_summary_text(ax_summary, meta, best_model, best_metrics, wilcoxon)

    outputs = []
    for suffix in (".png", ".svg"):
        output_path = output_prefix.with_suffix(suffix)
        fig.savefig(output_path, dpi=220 if suffix == ".png" else None, bbox_inches="tight")
        outputs.append(output_path)

    plt.close(fig)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate documentation-ready evaluation visualization.")
    parser.add_argument(
        "--input",
        default=str(ROOT / "results" / "evaluation" / "full_vascular_glcm_mobilenet" / "full_evaluation_20260427_004032.json"),
        help="Path to full evaluation JSON.",
    )
    parser.add_argument(
        "--output-prefix",
        default=str(ROOT / "docs" / "ringkasan_evaluasi_model"),
        help="Output path without extension.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = build_visualization(input_path, output_prefix)
    for output in outputs:
        print(f"[saved] {output}")


if __name__ == "__main__":
    main()
