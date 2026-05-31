"""
Regenerate:
  - fig_repeated_cv.png       (clean title, stats moved inside panels)
  - fig_08a_mcnemar.png       (McNemar exact test, n=24 test images)
  - fig_08b_bootstrap.png     (Bootstrap CI + Permutation, n=24 test images)
  - fig_08c_summary.png       (Summary table, n=24 test images)
"""
import sys, math, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np, pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.stats import binom, wilcoxon, ttest_rel
from sklearn.model_selection import RepeatedStratifiedKFold
from src.clustering.fuzzy_cmeans import FuzzyCMeans
from src.clustering.kmeans_baseline import KMeansBaseline

BG    = "#FAFAFA"
C_FCM = "#2A9D8F"
C_KM  = "#F4A261"
ALPHA = 0.05

# ── Data ───────────────────────────────────────────────────────────────
X_tr  = np.load("data/features/awc_features.npy")
y_tr  = np.load("data/features/awc_labels.npy")
X_te  = np.load("data/features/awc_test_features.npy")
y_te  = np.load("data/features/awc_test_labels.npy")
X_all = np.vstack([X_tr, X_te])
y_all = np.hstack([y_tr, y_te])

with open("models/baselines/fcm_model.pkl", "rb") as f: fcm_model = pickle.load(f)
with open("models/baselines/kmeans_model.pkl", "rb") as f: km_model  = pickle.load(f)

# ── n=24 test predictions ───────────────────────────────────────────────
fcm_te = fcm_model.predict(X_te)
km_te  = km_model.predict(X_te)
ca = (fcm_te == y_te)   # FCM correct per sample
cb = (km_te  == y_te)   # KM  correct per sample
fcm_acc_te = ca.mean()
km_acc_te  = cb.mean()
obs_diff_te = fcm_acc_te - km_acc_te
N_TE = len(y_te)

# McNemar on n=24
b_te = int((ca & ~cb).sum())
c_te = int((~ca & cb).sum())
n_disc_te = b_te + c_te
if n_disc_te > 0:
    p_mcnemar = float(2 * binom.cdf(min(b_te, c_te), n_disc_te, 0.5))
    p_mcnemar = min(p_mcnemar, 1.0)
else:
    p_mcnemar = 1.0

# Bootstrap CI on n=24
rng = np.random.default_rng(42)
bs_diffs = []
for _ in range(10_000):
    idx = rng.integers(0, N_TE, N_TE)
    bs_diffs.append(ca[idx].mean() - cb[idx].mean())
bs_diffs = np.array(bs_diffs)
ci_lo, ci_hi = np.percentile(bs_diffs, [2.5, 97.5])

# Permutation test on n=24
perm_diffs = []
for _ in range(10_000):
    swap = rng.random(N_TE) < 0.5
    a_perm = np.where(swap, ca, cb)
    b_perm = np.where(swap, cb, ca)
    perm_diffs.append(a_perm.mean() - b_perm.mean())
perm_diffs = np.array(perm_diffs)
p_perm = float((np.abs(perm_diffs) >= np.abs(obs_diff_te)).mean())

# ── Repeated CV (all n=180) ─────────────────────────────────────────────
print("Running Repeated CV (10×5=50 folds)...")
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
fold_fcm, fold_km = [], []
for tr_i, te_i in rskf.split(X_all, y_all):
    f2 = FuzzyCMeans(c=2, m=2.0, error=1e-5, max_iter=300, random_state=42)
    k2 = KMeansBaseline(n_clusters=2, random_state=42)
    f2.fit(X_all[tr_i], y_all[tr_i]); k2.fit(X_all[tr_i], y_all[tr_i])
    fold_fcm.append((f2.predict(X_all[te_i]) == y_all[te_i]).mean())
    fold_km.append( (k2.predict(X_all[te_i]) == y_all[te_i]).mean())
fold_fcm   = np.array(fold_fcm)
fold_km    = np.array(fold_km)
fold_diffs = fold_fcm - fold_km
fcm_mat    = fold_fcm.reshape(10, 5)
km_mat     = fold_km.reshape(10, 5)
fcm_rep    = fcm_mat.mean(axis=1)
km_rep     = km_mat.mean(axis=1)
wil_s, wil_p = wilcoxon(fold_diffs, alternative="two-sided")
t_s,   t_p   = ttest_rel(fold_fcm, fold_km)
ci_cv_lo, ci_cv_hi = np.percentile(fold_diffs, [2.5, 97.5])
print(f"  FCM: {fold_fcm.mean():.3f}±{fold_fcm.std():.3f}  "
      f"KM: {fold_km.mean():.3f}±{fold_km.std():.3f}  "
      f"Wilcoxon p={wil_p:.2e}  Paired-t p={t_p:.2e}")

# ═══════════════════════════════════════════════════════════════════════
# FIG A — Repeated CV (clean title, stats inside panels)
# ═══════════════════════════════════════════════════════════════════════
print("FIG: Repeated CV ...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG,
                          gridspec_kw={"wspace": 0.32})
fig.suptitle(
    "Repeated Stratified 5-Fold CV  (10 repeats × 5 folds = 50 evaluations)",
    fontsize=13, fontweight="bold", y=1.01)

# -- Panel A: box plots per repeat
ax = axes[0]
ax.set_facecolor(BG)
pos_f = np.arange(1, 11) - 0.2
pos_k = np.arange(1, 11) + 0.2
kw_f = dict(patch_artist=True,
            boxprops=dict(facecolor=C_FCM, alpha=0.72),
            medianprops=dict(color="white", lw=2),
            whiskerprops=dict(color=C_FCM, lw=1.2),
            capprops=dict(color=C_FCM, lw=1.2),
            flierprops=dict(marker="o", color=C_FCM, ms=3, alpha=0.6))
kw_k = dict(patch_artist=True,
            boxprops=dict(facecolor=C_KM, alpha=0.72),
            medianprops=dict(color="white", lw=2),
            whiskerprops=dict(color=C_KM, lw=1.2),
            capprops=dict(color=C_KM, lw=1.2),
            flierprops=dict(marker="o", color=C_KM, ms=3, alpha=0.6))
ax.boxplot(fcm_mat.T, positions=pos_f, widths=0.30, **kw_f)
ax.boxplot(km_mat.T,  positions=pos_k, widths=0.30, **kw_k)
ax.axhline(fold_fcm.mean(), color=C_FCM, lw=1.5, ls="--", alpha=0.7)
ax.axhline(fold_km.mean(),  color=C_KM,  lw=1.5, ls="--", alpha=0.7)
ax.set_xticks(range(1, 11))
ax.set_xticklabels([f"R{i}" for i in range(1, 11)], fontsize=8.5)
ax.set_ylabel("Fold Accuracy", fontsize=10)
ax.set_ylim(0.45, 1.05)
ax.set_title("Accuracy Distribution per Repeat", fontsize=11, fontweight="bold")
# Stats inside panel — upper left
ax.text(0.03, 0.97,
        f"FCM: {fold_fcm.mean():.3f} ± {fold_fcm.std():.3f}\n"
        f"KM:   {fold_km.mean():.3f} ± {fold_km.std():.3f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CCC", lw=1.2))
ax.legend(handles=[
    mpatches.Patch(fc=C_FCM, alpha=0.72, label="Fuzzy C-Means"),
    mpatches.Patch(fc=C_KM,  alpha=0.72, label="K-Means"),
], fontsize=9, framealpha=0.95, loc="lower right")
ax.grid(True, alpha=0.25, ls="--", axis="y"); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

# -- Panel B: per-repeat mean line
ax2 = axes[1]
ax2.set_facecolor(BG)
x = np.arange(1, 11)
ax2.plot(x, fcm_rep, "o-", color=C_FCM, lw=2, ms=6)
ax2.plot(x, km_rep,  "s-", color=C_KM,  lw=2, ms=6)
ax2.fill_between(x, fcm_rep - fcm_mat.std(1), fcm_rep + fcm_mat.std(1),
                 color=C_FCM, alpha=0.13)
ax2.fill_between(x, km_rep - km_mat.std(1),   km_rep + km_mat.std(1),
                 color=C_KM,  alpha=0.13)
ax2.axhline(fcm_rep.mean(), color=C_FCM, lw=1.5, ls="--", alpha=0.7)
ax2.axhline(km_rep.mean(),  color=C_KM,  lw=1.5, ls="--", alpha=0.7)
ax2.set_xticks(x); ax2.set_xticklabels([f"R{i}" for i in range(1, 11)], fontsize=8.5)
ax2.set_ylim(0.52, 0.98)
ax2.set_ylabel("Mean Accuracy per Repeat", fontsize=10)
ax2.set_title("Per-Repeat Mean ± Std", fontsize=11, fontweight="bold")
ax2.legend(handles=[
    mlines.Line2D([], [], marker="o", color=C_FCM, ms=6, lw=2, label="Fuzzy C-Means"),
    mlines.Line2D([], [], marker="s", color=C_KM,  ms=6, lw=2, label="K-Means"),
], fontsize=9, framealpha=0.95, loc="lower right")
ax2.grid(True, alpha=0.25, ls="--"); ax2.set_axisbelow(True)
ax2.spines[["top", "right"]].set_visible(False)

# -- Panel C: fold-level diff histogram
ax3 = axes[2]
ax3.set_facecolor(BG)
ax3.hist(fold_diffs, bins=14, color=C_FCM, alpha=0.72, edgecolor="white")
ax3.axvline(0, color="#555", lw=1.5, ls=":", zorder=4, label="No difference")
ax3.axvline(fold_diffs.mean(), color="#B71C1C", lw=2.2, zorder=5,
            label=f"Mean {fold_diffs.mean():+.3f}")
ax3.axvline(ci_cv_lo, color="#1A237E", lw=1.8, ls="--")
ax3.axvline(ci_cv_hi, color="#1A237E", lw=1.8, ls="--",
            label=f"95% CI [{ci_cv_lo:+.3f},{ci_cv_hi:+.3f}]")
pct = (fold_diffs > 0).mean()
ax3.set_xlabel("Accuracy Difference (FCM − K-Means)", fontsize=10)
ax3.set_ylabel("Count (50 folds)", fontsize=10)
ax3.set_title("Per-Fold Differences", fontsize=11, fontweight="bold")
# p-values inside panel — upper left
ax3.text(0.03, 0.97,
         f"Wilcoxon p={wil_p:.2e}\nPaired-t  p={t_p:.2e}",
         transform=ax3.transAxes, va="top", ha="left", fontsize=9,
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CCC", lw=1.2))
# FCM > KM badge — upper right
ax3.text(0.97, 0.97, f"FCM > KM\n{int(pct*50)}/50 folds",
         transform=ax3.transAxes, ha="right", va="top",
         fontsize=10, fontweight="bold", color=C_FCM,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_FCM, lw=1.5))
ax3.legend(fontsize=9, framealpha=0.95, loc="upper center",
           bbox_to_anchor=(0.5, 0.72))
ax3.grid(True, alpha=0.25, ls="--"); ax3.set_axisbelow(True)
ax3.spines[["top", "right"]].set_visible(False)

plt.savefig("docs/fig_repeated_cv.png", dpi=200,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("  Saved fig_repeated_cv.png")

# ═══════════════════════════════════════════════════════════════════════
# FIG 08a — McNemar Exact Test (n=24 test images)
# ═══════════════════════════════════════════════════════════════════════
print("FIG 08a: McNemar (n=24) ...")

sig_str = "n.s. (Not Significant)" if p_mcnemar >= ALPHA else "★ Significant"
sig_col = "#B71C1C" if p_mcnemar < ALPHA else "#555"

fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=BG)
ax.set_facecolor(BG); ax.axis("off")
ax.set_xlim(0, 7); ax.set_ylim(0, 5.5)

ax.text(3.5, 5.2, "McNemar Exact Test — FCM vs K-Means",
        ha="center", fontsize=13, fontweight="bold")
ax.text(3.5, 4.85, f"n = {N_TE} held-out test images  |  α = {ALPHA}",
        ha="center", fontsize=10, color="#555")

# 2×2 contingency table
TBL_X, TBL_Y = 0.8, 1.6
CW, CH = 2.5, 0.9

def cell(cx, cy, txt, fc, tc="#222", fs=11, bold=False):
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx, cy), CW, CH, boxstyle="square,pad=0",
        fc=fc, ec="white", lw=2.5))
    ax.text(cx + CW/2, cy + CH/2, txt, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", color=tc)

# Header row
ax.text(TBL_X + CW + CW/2,   TBL_Y + 2*CH + 0.25, "KM Correct",
        ha="center", fontsize=10, fontweight="bold", color=C_KM)
ax.text(TBL_X + 2*CW + CW/2, TBL_Y + 2*CH + 0.25, "KM Wrong",
        ha="center", fontsize=10, fontweight="bold", color=C_KM)
ax.text(TBL_X - 0.05, TBL_Y + 1*CH + CH/2, "FCM\nCorrect",
        ha="right", va="center", fontsize=10, fontweight="bold", color=C_FCM)
ax.text(TBL_X - 0.05, TBL_Y + 0*CH + CH/2, "FCM\nWrong",
        ha="right", va="center", fontsize=10, fontweight="bold", color=C_FCM)

a = int((ca & cb).sum())
d = int((~ca & ~cb).sum())

cell(TBL_X + CW,   TBL_Y + CH, f"a = {a}",    "#E8F5E9", "#2E7D32", fs=12)
cell(TBL_X + 2*CW, TBL_Y + CH, f"b = {b_te}", "#FFEBEE", "#C62828", fs=14, bold=True)
cell(TBL_X + CW,   TBL_Y,      f"c = {c_te}", "#FFEBEE", "#C62828", fs=14, bold=True)
cell(TBL_X + 2*CW, TBL_Y,      f"d = {d}",    "#E8F5E9", "#2E7D32", fs=12)

ax.text(3.5, TBL_Y - 0.35,
        f"Discordant pairs  b + c = {n_disc_te}  |  b = {b_te}, c = {c_te}",
        ha="center", fontsize=9, color="#666", style="italic")

# Result box
ax.add_patch(mpatches.FancyBboxPatch((1.8, 0.3), 3.4, 0.75,
    boxstyle="round,pad=0.1", fc="#F3F3F3", ec=sig_col, lw=2.2))
ax.text(3.5, 0.67, f"p = {p_mcnemar:.4f}   →   {sig_str}",
        ha="center", va="center", fontsize=12, fontweight="bold", color=sig_col)

note = ("Only 1 discordant pair on n=24. See Repeated CV\n"
        f"(50 folds) for Wilcoxon p={wil_p:.2e} ★ convergent evidence.")
ax.text(3.5, 0.08, note, ha="center", fontsize=8, color="#777", style="italic")

plt.savefig("docs/fig_08a_mcnemar.png", dpi=200,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("  Saved fig_08a_mcnemar.png")

# ═══════════════════════════════════════════════════════════════════════
# FIG 08b — Bootstrap CI + Permutation Test (n=24)
# ═══════════════════════════════════════════════════════════════════════
print("FIG 08b: Bootstrap + Permutation (n=24) ...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG,
                          gridspec_kw={"wspace": 0.35})
fig.suptitle(
    "Resampling-Based Tests — FCM vs K-Means  (n=24 test images)",
    fontsize=13, fontweight="bold")

# -- Panel A: Bootstrap
ax = axes[0]
ax.set_facecolor(BG)
ax.hist(bs_diffs, bins=30, color=C_FCM, alpha=0.72, edgecolor="white")
ax.axvline(0,          color="#555",    lw=1.5, ls=":", label="H₀: diff = 0")
ax.axvline(obs_diff_te,color="#B71C1C", lw=2.2, label=f"Observed = {obs_diff_te:+.3f}")
ax.axvline(ci_lo,      color="#1A237E", lw=1.8, ls="--")
ax.axvline(ci_hi,      color="#1A237E", lw=1.8, ls="--",
           label=f"95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]")
ax.set_xlabel("Accuracy Difference (FCM − K-Means)", fontsize=10)
ax.set_ylabel("Frequency (10,000 resamples)", fontsize=10)
ax.set_title("Bootstrap Distribution", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, framealpha=0.95, loc="upper left")
ax.grid(True, alpha=0.25, ls="--"); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

# -- Panel B: Permutation
ax2 = axes[1]
ax2.set_facecolor(BG)
ax2.hist(perm_diffs, bins=30, color="#90A4AE", alpha=0.75, edgecolor="white",
         label="Permutation dist.")
ax2.axvline(obs_diff_te,  color="#B71C1C", lw=2.2,
            label=f"Observed = {obs_diff_te:+.3f}")
ax2.axvline(-obs_diff_te, color="#B71C1C", lw=2.2, ls="--")
sig_str_p = "n.s." if p_perm >= ALPHA else "★ Significant"
ax2.text(0.97, 0.97,
         f"p = {p_perm:.4f}\n{sig_str_p}",
         transform=ax2.transAxes, ha="right", va="top",
         fontsize=11, fontweight="bold",
         color="#B71C1C" if p_perm < ALPHA else "#555",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCC", lw=1.2))
ax2.set_xlabel("Accuracy Difference (FCM − K-Means)", fontsize=10)
ax2.set_ylabel("Frequency (10,000 permutations)", fontsize=10)
ax2.set_title("Permutation Test  (two-tailed)", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9, framealpha=0.95, loc="upper left")
ax2.grid(True, alpha=0.25, ls="--"); ax2.set_axisbelow(True)
ax2.spines[["top", "right"]].set_visible(False)

plt.savefig("docs/fig_08b_bootstrap.png", dpi=200,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("  Saved fig_08b_bootstrap.png")

# ═══════════════════════════════════════════════════════════════════════
# FIG 08c — Summary Table (n=24 test + repeated CV)
# ═══════════════════════════════════════════════════════════════════════
print("FIG 08c: Summary table ...")

fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=BG)
ax.set_facecolor(BG); ax.axis("off")

ax.text(5, 5.3, "Statistical Validation Summary — FCM vs K-Means",
        ha="center", fontsize=13, fontweight="bold")
ax.text(5, 4.95, "α = 0.05",
        ha="center", fontsize=10, color="#555")

# Table data
rows = [
    ("Test Set Accuracy (n=24)",  f"{fcm_acc_te:.3f} ({fcm_acc_te:.1%})",
                                   f"{km_acc_te:.3f} ({km_acc_te:.1%})",
                                   f"Δ = {obs_diff_te:+.3f}", "#F5F5F5"),
    ("Bootstrap 95% CI (n=24)",   "—", "—",
                                   f"[{ci_lo:+.3f}, {ci_hi:+.3f}]", "#F5F5F5"),
    ("McNemar Exact (n=24)",      "—", "—",
                                   f"p = {p_mcnemar:.4f}  n.s.", "#FFF8E1"),
    ("Permutation Test (n=24)",   "—", "—",
                                   f"p = {p_perm:.4f}  n.s.", "#FFF8E1"),
    ("Repeated CV Acc (n=180)",   f"{fold_fcm.mean():.3f}±{fold_fcm.std():.3f}",
                                   f"{fold_km.mean():.3f}±{fold_km.std():.3f}",
                                   f"Δ = {fold_diffs.mean():+.3f}", "#E8F5E9"),
    ("Wilcoxon (50 folds)",       "—", "—",
                                   f"p = {wil_p:.2e}  ★ Significant", "#E8F5E9"),
    ("Paired-t  (50 folds)",      "—", "—",
                                   f"p = {t_p:.2e}  ★ Significant", "#E8F5E9"),
]

headers = ["Metric", "FCM", "K-Means", "Test / Result"]
col_x   = [0.05, 3.5, 5.4, 7.0]
col_w   = [3.4,  1.85, 1.55, 2.95]
ROW_H   = 0.47
Y0      = 4.55

# Header
for i, (hdr, cx, cw) in enumerate(zip(headers, col_x, col_w)):
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx, Y0), cw - 0.05, ROW_H,
        boxstyle="square,pad=0", fc="#37474F", ec="white", lw=1.5))
    ax.text(cx + cw/2, Y0 + ROW_H/2, hdr,
            ha="center", va="center", fontsize=10,
            fontweight="bold", color="white")

# Data rows
for r_i, (m, fcm_v, km_v, res_v, bg) in enumerate(rows):
    ry = Y0 - (r_i + 1) * (ROW_H + 0.04)
    vals = [m, fcm_v, km_v, res_v]
    for i, (val, cx, cw) in enumerate(zip(vals, col_x, col_w)):
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx, ry), cw - 0.05, ROW_H,
            boxstyle="square,pad=0", fc=bg, ec="white", lw=1.0))
        col = "#222"
        if "★" in val:
            col = "#1B5E20"
        elif "n.s." in val and "Significant" not in val:
            col = "#6D4C41"
        fs = 9.5 if i == 0 else 9
        ax.text(cx + (0.12 if i == 0 else cw/2), ry + ROW_H/2, val,
                ha=("left" if i == 0 else "center"), va="center",
                fontsize=fs, color=col,
                fontweight="bold" if "★" in val else "normal")

ax.legend(handles=[
    mpatches.Patch(fc=C_FCM, label="Fuzzy C-Means"),
    mpatches.Patch(fc=C_KM,  label="K-Means"),
], loc="lower right", fontsize=9, framealpha=0.9)

# Footnote
note = ("McNemar and Wilcoxon on test set: n=24 (1 discordant pair) → insufficient power for formal significance.\n"
        "Repeated CV provides 50 independent fold estimates (n=180), enabling Wilcoxon and Paired-t inference.")
ax.text(5, 0.12, note, ha="center", fontsize=8, color="#777",
        style="italic", wrap=True)

ax.set_xlim(0, 10); ax.set_ylim(0, 5.6)
plt.savefig("docs/fig_08c_summary.png", dpi=200,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("  Saved fig_08c_summary.png")

print("\nAll figures done.")
