"""Generate standardized K-Means vs FCM figures from one n=24 test split.

This script is intended for the paper figures where both charts must use
the same held-out test set and English labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, wilcoxon


ROOT = Path(__file__).resolve().parent.parent
BG = "#FAFAFA"
C_KM = "#F4A261"
C_FCM = "#2A9D8F"
C_WILCOXON = "#2166AC"
ALPHA = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate English K-Means vs FCM figures from the standardized 24-image test split."
    )
    parser.add_argument(
        "--input",
        default=str(
            ROOT
            / "results"
            / "evaluation"
            / "full_vascular_glcm_mobilenet"
            / "full_evaluation_20260427_004032.json"
        ),
        help="Path to the evaluation JSON that contains KMeans and FCM predictions.",
    )
    parser.add_argument(
        "--correct-output",
        default=str(ROOT / "docs" / "fig_06_prediction_correct.png"),
        help="Output path for the correct-vs-incorrect figure.",
    )
    parser.add_argument(
        "--stats-output",
        default=str(ROOT / "docs" / "fig_08_statistical_tests.png"),
        help="Output path for the statistical test figure.",
    )
    return parser.parse_args()


def load_payload(input_path: Path) -> dict:
    with input_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_arrays(payload: dict, model_key: str) -> tuple[np.ndarray, np.ndarray]:
    result = payload["results"][model_key]
    return np.asarray(result["y_true"], dtype=int), np.asarray(result["y_pred"], dtype=int)


def run_mcnemar_exact(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray
) -> tuple[float, int, int, int]:
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b_count = int((correct_a & ~correct_b).sum())
    c_count = int((~correct_a & correct_b).sum())
    discordant = b_count + c_count
    if discordant == 0:
        return 1.0, b_count, c_count, discordant
    p_value = float(binomtest(min(b_count, c_count), discordant, p=0.5, alternative="two-sided").pvalue)
    return p_value, b_count, c_count, discordant


def run_wilcoxon_correctness(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray
) -> tuple[float, float, int]:
    diff = (pred_a == y_true).astype(float) - (pred_b == y_true).astype(float)
    non_zero = diff[diff != 0]
    if len(non_zero) == 0:
        return 0.0, 1.0, 0
    stat, p_value = wilcoxon(non_zero, alternative="two-sided", zero_method="wilcox")
    return float(stat), float(p_value), int(len(non_zero))


def build_correct_incorrect_figure(
    y_true: np.ndarray, km_pred: np.ndarray, fcm_pred: np.ndarray, output_path: Path
) -> None:
    n_test = len(y_true)
    summaries = [
        ("K-Means", int((km_pred == y_true).sum()), C_KM),
        ("FCM", int((fcm_pred == y_true).sum()), C_FCM),
    ]

    fig, ax = plt.subplots(figsize=(7.0, 5.5), facecolor=BG)
    ax.set_facecolor(BG)

    x = np.arange(len(summaries))
    width = 0.5
    correct_values = [item[1] for item in summaries]
    wrong_values = [n_test - item[1] for item in summaries]
    colors = [item[2] for item in summaries]

    bars_correct = ax.bar(
        x,
        correct_values,
        width,
        color=colors,
        alpha=0.9,
        edgecolor="white",
        linewidth=1.2,
    )
    bars_wrong = ax.bar(
        x,
        wrong_values,
        width,
        bottom=correct_values,
        color=colors,
        alpha=0.32,
        edgecolor=colors,
        linewidth=1.2,
        hatch="////",
    )

    for bar, value in zip(bars_correct, correct_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value / 2,
            str(value),
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            color="white",
        )

    for bar, value, base, color in zip(bars_wrong, wrong_values, correct_values, colors):
        if value > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                base + value / 2,
                str(value),
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=color,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([item[0] for item in summaries], fontsize=12)
    ax.set_ylabel(f"Number of Predictions (n={n_test})", fontsize=11)
    ax.set_ylim(0, n_test + 6)
    ax.set_title(
        "Correct vs Incorrect Predictions per Model\n"
        f"(Standardized test split, n={n_test}; 12 fertile, 12 infertile)",
        fontsize=12,
        fontweight="bold",
    )
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            mpatches.Patch(color="#888888", alpha=0.9, label="Correct"),
            mpatches.Patch(color="#888888", alpha=0.35, hatch="////", label="Incorrect"),
        ],
        fontsize=10,
        framealpha=0.95,
        loc="upper right",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def build_statistical_figure(
    y_true: np.ndarray, km_pred: np.ndarray, fcm_pred: np.ndarray, output_path: Path
) -> None:
    n_test = len(y_true)
    mcnemar_p, b_count, c_count, discordant = run_mcnemar_exact(y_true, km_pred, fcm_pred)
    wilcoxon_stat, wilcoxon_p, wilcoxon_n = run_wilcoxon_correctness(y_true, km_pred, fcm_pred)

    fig, ax = plt.subplots(figsize=(6.8, 4.9), facecolor=BG)
    ax.set_facecolor(BG)

    x = np.arange(1)
    width = 0.32

    bars_wilcoxon = ax.bar(
        x - width / 2,
        [wilcoxon_p],
        width,
        color=C_WILCOXON,
        alpha=0.85,
        edgecolor="white",
        label="Wilcoxon Signed-Rank",
        zorder=3,
    )
    bars_mcnemar = ax.bar(
        x + width / 2,
        [mcnemar_p],
        width,
        color=C_KM,
        alpha=0.85,
        edgecolor="white",
        label="McNemar (exact)",
        zorder=3,
    )

    ax.axhline(
        ALPHA,
        color="#D32F2F",
        linestyle="--",
        linewidth=1.3,
        label=f"alpha = {ALPHA:.2f} threshold",
        zorder=4,
    )

    for bars, p_value in ((bars_wilcoxon, wilcoxon_p), (bars_mcnemar, mcnemar_p)):
        bar = bars[0]
        marker = "(n.s.)" if p_value >= ALPHA else "(*)"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{p_value:.4f}\n{marker}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#555555",
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["K-Means\nvs\nFCM"], fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("p-value", fontsize=11)
    ax.set_title(
        "Statistical Tests Pairwise Model Comparison\n"
        f"(Standardized test split, n={n_test} images; n.s. = not significant)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9, framealpha=0.95, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)

    fig.text(
        0.5,
        0.01,
        (
            f"Discordant pairs: b={b_count}, c={c_count}, total={discordant}; "
            f"Wilcoxon non-zero pairs={wilcoxon_n}; W={wilcoxon_stat:.1f}"
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#666666",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    payload = load_payload(Path(args.input))
    y_true, km_pred = get_arrays(payload, "KMeans")
    _, fcm_pred = get_arrays(payload, "FCM")

    correct_output = Path(args.correct_output)
    stats_output = Path(args.stats_output)

    build_correct_incorrect_figure(y_true, km_pred, fcm_pred, correct_output)
    build_statistical_figure(y_true, km_pred, fcm_pred, stats_output)

    print(f"[saved] {correct_output}")
    print(f"[saved] {stats_output}")
    print(
        "[summary] "
        f"K-Means={int((km_pred == y_true).sum())}/{len(y_true)} correct, "
        f"FCM={int((fcm_pred == y_true).sum())}/{len(y_true)} correct"
    )


if __name__ == "__main__":
    main()


