"""
Statistical Significance Testing for Model Comparison.

Implements the tests required for Scopus Q1 publication:

  1. Wilcoxon signed-rank test  -- non-parametric paired comparison of
     per-sample metrics (IoU, Dice, Accuracy) between models.
     Use when normality cannot be assumed (small dataset).

  2. McNemar's test  -- compare binary correct/incorrect predictions
     between two classifiers on the same test set.

  3. Friedman test  -- non-parametric k-way ANOVA for three or more
     models simultaneously (AWC vs K-Means vs FCM).

  4. Effect size (Cohen's d / rank-biserial r)  -- quantify how *large*
     the difference is, not just whether it is significant.

All results include:
  - test statistic and p-value
  - interpretation (significant / not significant)
  - effect size
  - confidence interval (bootstrap, 95%)

References
----------
- Wilcoxon (1945) Individual comparisons by ranking methods.
- McNemar (1947) Note on the sampling error of the difference.
- Cohen (1988) Statistical Power Analysis for the Behavioral Sciences.
- Demsar (2006) Statistical Comparisons of Classifiers over Multiple Data Sets.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    func,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    rng_seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap 95 % CI of func(a) - func(b)."""
    rng = np.random.default_rng(rng_seed)
    diffs = np.empty(n_bootstrap)
    n = len(a)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs[i] = func(a[idx]) - func(b[idx])
    lo = np.percentile(diffs, 100 * alpha / 2)
    hi = np.percentile(diffs, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def _rank_biserial(stat_w: float, n: int) -> float:
    """Rank-biserial correlation from Wilcoxon W statistic."""
    return 1 - (4 * stat_w) / (n * (n + 1))


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired samples."""
    diff = a - b
    return float(diff.mean() / (diff.std(ddof=1) + 1e-10))


def _interpret_p(p: float, alpha: float = 0.05) -> str:
    if p < 0.001:
        return "highly significant (p < 0.001)"
    elif p < 0.01:
        return f"significant (p = {p:.4f} < 0.01)"
    elif p < alpha:
        return f"significant (p = {p:.4f} < {alpha})"
    else:
        return f"not significant (p = {p:.4f} >= {alpha})"


def _effect_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# 1. Wilcoxon signed-rank test
# ---------------------------------------------------------------------------

class WilcoxonTest:
    """Paired non-parametric test for two models on per-sample metrics.

    Parameters
    ----------
    alpha : float
        Significance level (default 0.05).
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def test(
        self,
        scores_a: Sequence[float],
        scores_b: Sequence[float],
        metric_name: str = "metric",
        model_a_name: str = "Model A",
        model_b_name: str = "Model B",
    ) -> Dict:
        """Run the test and return a result dictionary.

        Parameters
        ----------
        scores_a, scores_b :
            Per-sample metric values (same length, same order).
        """
        a = np.asarray(scores_a, dtype=float)
        b = np.asarray(scores_b, dtype=float)

        if len(a) != len(b):
            raise ValueError("scores_a and scores_b must have the same length.")
        if len(a) < 6:
            warnings.warn(
                f"Wilcoxon test has low power with only {len(a)} samples. "
                "Interpret results with caution.",
                UserWarning,
                stacklevel=2,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")

        r = _rank_biserial(stat, len(a))
        d = _cohens_d(a, b)
        ci_lo, ci_hi = _bootstrap_ci(a, b, np.mean)

        return {
            "test": "Wilcoxon signed-rank",
            "metric": metric_name,
            "model_a": model_a_name,
            "model_b": model_b_name,
            "n_samples": int(len(a)),
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "mean_diff": float(np.mean(a) - np.mean(b)),
            "ci_95_diff": [ci_lo, ci_hi],
            "statistic_W": float(stat),
            "p_value": float(p),
            "interpretation": _interpret_p(p, self.alpha),
            "significant": bool(p < self.alpha),
            "effect_size_r": float(r),
            "effect_size_d": float(d),
            "effect_magnitude": _effect_label(d),
        }

    def compare_all_metrics(
        self,
        metrics_a: Dict[str, Sequence[float]],
        metrics_b: Dict[str, Sequence[float]],
        model_a_name: str = "Model A",
        model_b_name: str = "Model B",
    ) -> List[Dict]:
        """Run Wilcoxon for every metric key present in both dicts."""
        results = []
        for key in metrics_a:
            if key in metrics_b and metrics_a[key] and metrics_b[key]:
                try:
                    r = self.test(
                        metrics_a[key],
                        metrics_b[key],
                        metric_name=key,
                        model_a_name=model_a_name,
                        model_b_name=model_b_name,
                    )
                    results.append(r)
                except Exception as exc:
                    results.append({"metric": key, "error": str(exc)})
        return results


# ---------------------------------------------------------------------------
# 2. McNemar's test
# ---------------------------------------------------------------------------

class McNemarTest:
    """McNemar's test for comparing two binary classifiers.

    Compares the number of samples that one model gets right
    while the other gets wrong (and vice versa).

    Parameters
    ----------
    alpha : float
        Significance level (default 0.05).
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def test(
        self,
        y_true: Sequence[int],
        y_pred_a: Sequence[int],
        y_pred_b: Sequence[int],
        model_a_name: str = "Model A",
        model_b_name: str = "Model B",
    ) -> Dict:
        """Run McNemar's test.

        Parameters
        ----------
        y_true   : ground-truth binary labels (0 / 1).
        y_pred_a : predictions from model A.
        y_pred_b : predictions from model B.
        """
        yt = np.asarray(y_true, dtype=int)
        ya = np.asarray(y_pred_a, dtype=int)
        yb = np.asarray(y_pred_b, dtype=int)

        correct_a = (ya == yt)
        correct_b = (yb == yt)

        # Contingency table
        # b: A wrong, B right  |  c: A right, B wrong
        b = int(((~correct_a) & correct_b).sum())
        c = int((correct_a & (~correct_b)).sum())

        # Use exact binomial test when b+c < 25, chi-square otherwise
        n_discordant = b + c
        if n_discordant == 0:
            return {
                "test": "McNemar",
                "model_a": model_a_name,
                "model_b": model_b_name,
                "b": b, "c": c,
                "n_discordant": 0,
                "statistic": None,
                "p_value": 1.0,
                "interpretation": "not significant (no discordant pairs)",
                "significant": False,
                "accuracy_a": float(correct_a.mean()),
                "accuracy_b": float(correct_b.mean()),
            }

        if n_discordant < 25:
            # Exact two-sided binomial test (mid-p correction)
            p = 2 * stats.binom.cdf(min(b, c), n_discordant, 0.5)
            p = min(p, 1.0)
            stat = float(min(b, c))
            test_variant = "McNemar (exact binomial)"
        else:
            # Chi-square with continuity correction
            stat = float((abs(b - c) - 1) ** 2 / (b + c))
            p = float(stats.chi2.sf(stat, df=1))
            test_variant = "McNemar (chi-square, continuity correction)"

        return {
            "test": test_variant,
            "model_a": model_a_name,
            "model_b": model_b_name,
            "n_samples": int(len(yt)),
            "contingency_b": b,
            "contingency_c": c,
            "n_discordant": n_discordant,
            "statistic": stat,
            "p_value": float(p),
            "interpretation": _interpret_p(p, self.alpha),
            "significant": bool(p < self.alpha),
            "accuracy_a": float(correct_a.mean()),
            "accuracy_b": float(correct_b.mean()),
            "accuracy_diff": float(correct_a.mean() - correct_b.mean()),
        }


# ---------------------------------------------------------------------------
# 3. Friedman test (3+ models)
# ---------------------------------------------------------------------------

class FriedmanTest:
    """Non-parametric Friedman test for three or more models.

    Analogous to repeated-measures ANOVA but without normality assumption.
    Followed by post-hoc Nemenyi test for pairwise comparisons.

    Parameters
    ----------
    alpha : float
        Significance level (default 0.05).
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def test(
        self,
        model_scores: Dict[str, Sequence[float]],
        metric_name: str = "metric",
        higher_is_better: bool = True,
    ) -> Dict:
        """Run Friedman test across multiple models.

        Parameters
        ----------
        model_scores : {model_name: [per-sample scores]}
            All lists must be the same length.
        higher_is_better :
            True  (default) for accuracy/F1/IoU — higher score = better model.
            False for loss/error metrics — lower score = better model.
        """
        names = list(model_scores.keys())
        arrays = [np.asarray(model_scores[n], dtype=float) for n in names]

        n_samples = len(arrays[0])
        for arr in arrays[1:]:
            if len(arr) != n_samples:
                raise ValueError("All model score lists must have the same length.")

        matrix = np.column_stack(arrays)           # shape (n_samples, n_models)
        stat, p = stats.friedmanchisquare(*[matrix[:, i] for i in range(len(names))])

        # Critical difference (Nemenyi) — approximation
        k = len(names)
        cd = self._nemenyi_cd(k, n_samples)

        # rankdata: rank 1 = lowest value, rank k = highest value
        ranks = stats.rankdata(matrix, axis=1)
        avg_ranks = {n: float(ranks[:, i].mean()) for i, n in enumerate(names)}

        # Best model: highest average rank when higher_is_better, else lowest
        if higher_is_better:
            best_model = max(avg_ranks, key=avg_ranks.get)
        else:
            best_model = min(avg_ranks, key=avg_ranks.get)

        return {
            "test": "Friedman",
            "metric": metric_name,
            "higher_is_better": higher_is_better,
            "models": names,
            "n_samples": n_samples,
            "statistic_chi2": float(stat),
            "p_value": float(p),
            "interpretation": _interpret_p(p, self.alpha),
            "significant": bool(p < self.alpha),
            "average_ranks": avg_ranks,
            "best_model": best_model,
            "nemenyi_critical_difference_0.05": float(cd),
            "note": (
                "Pairwise significance: |rank_i - rank_j| > CD implies p < 0.05 "
                "between those two models. "
                f"Higher rank = {'better' if higher_is_better else 'worse'}."
            ),
        }

    @staticmethod
    def _nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
        """Nemenyi critical difference (two-tailed, alpha=0.05)."""
        # Critical values q_{alpha} for the Studentised range statistic
        q_alpha = {2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728,
                   6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
        q = q_alpha.get(k, 3.164)
        return q * np.sqrt(k * (k + 1) / (6 * n))


# ---------------------------------------------------------------------------
# 4. Composite report runner
# ---------------------------------------------------------------------------

class StatisticalTestSuite:
    """Run all statistical tests and compile a single structured report.

    Typical usage
    -------------
    suite = StatisticalTestSuite(alpha=0.05)

    # Wilcoxon: AWC vs K-Means on IoU and Dice
    suite.add_wilcoxon(
        metrics_a={"iou": [0.9, 0.7, 0.8], "dice": [0.88, 0.72, 0.79]},
        metrics_b={"iou": [0.6, 0.5, 0.55], "dice": [0.58, 0.52, 0.54]},
        model_a_name="AWC", model_b_name="K-Means",
    )

    # McNemar: AWC vs FCM on binary predictions
    suite.add_mcnemar(
        y_true=[1, 0, 1, 1, 0],
        y_pred_a=[1, 0, 1, 0, 0],
        y_pred_b=[1, 1, 0, 1, 0],
        model_a_name="AWC", model_b_name="FCM",
    )

    # Friedman: all three methods
    suite.add_friedman(
        model_scores={"AWC": [...], "K-Means": [...], "FCM": [...]},
        metric_name="accuracy",
    )

    report = suite.run()
    suite.save("results/statistical_tests/report.json")
    suite.print_summary()
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self._wilcoxon_tasks: List[dict] = []
        self._mcnemar_tasks: List[dict] = []
        self._friedman_tasks: List[dict] = []
        self._results: Optional[Dict] = None

    # -- Task registration --

    def add_wilcoxon(
        self,
        metrics_a: Dict[str, Sequence[float]],
        metrics_b: Dict[str, Sequence[float]],
        model_a_name: str = "Model A",
        model_b_name: str = "Model B",
    ) -> "StatisticalTestSuite":
        self._wilcoxon_tasks.append(
            dict(metrics_a=metrics_a, metrics_b=metrics_b,
                 model_a_name=model_a_name, model_b_name=model_b_name)
        )
        return self

    def add_mcnemar(
        self,
        y_true: Sequence[int],
        y_pred_a: Sequence[int],
        y_pred_b: Sequence[int],
        model_a_name: str = "Model A",
        model_b_name: str = "Model B",
    ) -> "StatisticalTestSuite":
        self._mcnemar_tasks.append(
            dict(y_true=y_true, y_pred_a=y_pred_a, y_pred_b=y_pred_b,
                 model_a_name=model_a_name, model_b_name=model_b_name)
        )
        return self

    def add_friedman(
        self,
        model_scores: Dict[str, Sequence[float]],
        metric_name: str = "metric",
    ) -> "StatisticalTestSuite":
        self._friedman_tasks.append(
            dict(model_scores=model_scores, metric_name=metric_name)
        )
        return self

    # -- Execution --

    def run(self) -> Dict:
        wt = WilcoxonTest(self.alpha)
        mt = McNemarTest(self.alpha)
        ft = FriedmanTest(self.alpha)

        wilcoxon_results = []
        for task in self._wilcoxon_tasks:
            wilcoxon_results.extend(
                wt.compare_all_metrics(
                    task["metrics_a"], task["metrics_b"],
                    task["model_a_name"], task["model_b_name"],
                )
            )

        mcnemar_results = [
            mt.test(**task) for task in self._mcnemar_tasks
        ]

        friedman_results = [
            ft.test(**task) for task in self._friedman_tasks
        ]

        self._results = {
            "alpha": self.alpha,
            "wilcoxon": wilcoxon_results,
            "mcnemar": mcnemar_results,
            "friedman": friedman_results,
        }
        return self._results

    # -- Output --

    def save(self, output_path: str) -> None:
        if self._results is None:
            self.run()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self._results, f, indent=2)

    def print_summary(self) -> None:
        if self._results is None:
            self.run()

        sep = "=" * 65
        print(f"\n{sep}")
        print("  STATISTICAL SIGNIFICANCE TESTING — SUMMARY")
        print(sep)

        print(f"\n{'─'*65}")
        print("  WILCOXON SIGNED-RANK TESTS")
        print(f"{'─'*65}")
        for r in self._results["wilcoxon"]:
            if "error" in r:
                print(f"  {r.get('metric','?')} → ERROR: {r['error']}")
                continue
            sig = "✓ SIG" if r["significant"] else "✗ n.s."
            print(
                f"  [{sig}] {r['model_a']} vs {r['model_b']} | "
                f"{r['metric']:10s} | "
                f"Δmean={r['mean_diff']:+.4f} | "
                f"p={r['p_value']:.4f} | "
                f"effect={r['effect_magnitude']}"
            )

        print(f"\n{'─'*65}")
        print("  McNEMAR TESTS")
        print(f"{'─'*65}")
        for r in self._results["mcnemar"]:
            sig = "✓ SIG" if r["significant"] else "✗ n.s."
            print(
                f"  [{sig}] {r['model_a']} vs {r['model_b']} | "
                f"acc_a={r['accuracy_a']:.3f} acc_b={r['accuracy_b']:.3f} | "
                f"p={r['p_value']:.4f}"
            )

        print(f"\n{'─'*65}")
        print("  FRIEDMAN TESTS")
        print(f"{'─'*65}")
        for r in self._results["friedman"]:
            sig = "✓ SIG" if r["significant"] else "✗ n.s."
            ranks = "  ".join(f"{m}={v:.2f}" for m, v in r["average_ranks"].items())
            print(
                f"  [{sig}] {r['metric']} | chi2={r['statistic_chi2']:.3f} "
                f"p={r['p_value']:.4f} | Ranks: {ranks}"
            )
            print(f"         Best: {r['best_model']}  |  CD(0.05)={r['nemenyi_critical_difference_0.05']:.3f}")

        print(f"\n{sep}\n")

    def save_latex_table(self, output_path: str) -> None:
        """Export a LaTeX table of Wilcoxon results for paper inclusion."""
        if self._results is None:
            self.run()

        rows = []
        for r in self._results["wilcoxon"]:
            if "error" in r:
                continue
            sig_star = "**" if r["p_value"] < 0.01 else ("*" if r["significant"] else "")
            rows.append(
                f"  {r['model_a']} vs {r['model_b']} & "
                f"{r['metric']} & "
                f"{r['mean_a']:.4f} & "
                f"{r['mean_b']:.4f} & "
                f"{r['mean_diff']:+.4f} & "
                f"{r['p_value']:.4f}{sig_star} & "
                f"{r['effect_magnitude']} \\\\"
            )

        table = (
            "\\begin{table}[ht]\n"
            "\\centering\n"
            "\\caption{Wilcoxon Signed-Rank Test Results}\n"
            "\\label{tab:wilcoxon}\n"
            "\\begin{tabular}{llccccc}\n"
            "\\hline\n"
            "Comparison & Metric & Mean A & Mean B & $\\Delta$ & p-value & Effect \\\\\n"
            "\\hline\n"
            + "\n".join(rows) + "\n"
            "\\hline\n"
            "\\multicolumn{7}{l}{$^{*}p<0.05$; $^{**}p<0.01$} \\\\\n"
            "\\end{tabular}\n"
            "\\end{table}"
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(table)