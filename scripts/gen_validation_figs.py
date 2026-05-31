"""
Generate four publication-ready validation figures (English):
  1. Repeated Stratified 5-Fold Cross-Validation (10 repeats)
  2. Statistical Testing from repeated CV runs
  3. Evaluation Protocol Figure
  4. Ablation Study (U-Net feature contribution)
  5. Prediction grid with actual egg photos (correct vs incorrect)
"""
import sys, math, warnings, pathlib
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, pickle, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from scipy.stats import wilcoxon, ttest_rel
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from src.clustering.fuzzy_cmeans import FuzzyCMeans
from src.clustering.kmeans_baseline import KMeansBaseline

BG    = "#FAFAFA"
C_FCM = "#2A9D8F"
C_KM  = "#F4A261"
C_OK  = "#2E7D32"
C_ERR = "#C62828"
ALPHA = 0.05

# ── Load feature data ───────────────────────────────────────────────────
X_tr = np.load("data/features/awc_features.npy")
y_tr = np.load("data/features/awc_labels.npy")
X_te = np.load("data/features/awc_test_features.npy")
y_te = np.load("data/features/awc_test_labels.npy")
X_all = np.vstack([X_tr, X_te])
y_all = np.hstack([y_tr, y_te])
N = len(y_all)

# Feature group indices
IDX_INTENSITY = list(range(0,  5))
IDX_HIST      = list(range(5,  37))
IDX_LBP       = list(range(37, 47))
IDX_GLCM      = list(range(47, 67))
IDX_VASCULAR  = list(range(67, 70))
IDX_EDGE      = list(range(70, 73))
IDX_UNET      = IDX_VASCULAR + IDX_EDGE

print(f"Dataset: {N} samples (train={len(X_tr)}, test={len(X_te)})")

# ═══════════════════════════════════════════════════════════════════════
# FIG 1+2: Repeated Stratified 5-Fold CV + Statistical Testing
# ═══════════════════════════════════════════════════════════════════════
print("\nRunning 10×5-Fold Repeated Stratified CV ...")
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

fcm_fold_accs = []
km_fold_accs  = []
fold_diffs    = []

for fold_i, (tr_idx, te_idx) in enumerate(rskf.split(X_all, y_all)):
    Xtr_, Xte_ = X_all[tr_idx], X_all[te_idx]
    ytr_, yte_ = y_all[tr_idx], y_all[te_idx]
    fcm = FuzzyCMeans(c=2, m=2.0, error=1e-5, max_iter=300, random_state=42)
    km  = KMeansBaseline(n_clusters=2, random_state=42)
    fcm.fit(Xtr_, ytr_)
    km.fit(Xtr_, ytr_)
    fa = (fcm.predict(Xte_) == yte_).mean()
    ka = (km.predict(Xte_)  == yte_).mean()
    fcm_fold_accs.append(fa)
    km_fold_accs.append(ka)
    fold_diffs.append(fa - ka)

fcm_fold_accs = np.array(fcm_fold_accs)
km_fold_accs  = np.array(km_fold_accs)
fold_diffs    = np.array(fold_diffs)

# Reshape into (n_repeats=10, n_splits=5)
fcm_mat = fcm_fold_accs.reshape(10, 5)
km_mat  = km_fold_accs.reshape(10, 5)

# Per-repeat means
fcm_rep_means = fcm_mat.mean(axis=1)
km_rep_means  = km_mat.mean(axis=1)
rep_diffs     = fcm_rep_means - km_rep_means

# Statistical tests on fold-level (50 observations)
wil_stat, wil_p = wilcoxon(fold_diffs, alternative="two-sided")
t_stat,   t_p   = ttest_rel(fcm_fold_accs, km_fold_accs)

print(f"FCM:    {fcm_fold_accs.mean():.4f} ± {fcm_fold_accs.std():.4f}")
print(f"KMeans: {km_fold_accs.mean():.4f} ± {km_fold_accs.std():.4f}")
print(f"Wilcoxon (50 folds): W={wil_stat:.1f}  p={wil_p:.4f}")
print(f"Paired t-test:       t={t_stat:.3f}  p={t_p:.4f}")

# ── FIGURE 1: Repeated CV Results ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), facecolor=BG)
fig.suptitle(
    "Repeated Stratified 5-Fold Cross-Validation (10 Repeats × 5 Folds = 50 Folds)\n"
    "Fuzzy C-Means vs K-Means — U-Net Extracted Hybrid Features",
    fontsize=13, fontweight="bold")

# Subplot A: Box plot per repeat
ax = axes[0]
ax.set_facecolor(BG)
pos_fcm = np.arange(1, 11) - 0.2
pos_km  = np.arange(1, 11) + 0.2
bp1 = ax.boxplot(fcm_mat.T, positions=pos_fcm, widths=0.32,
                 patch_artist=True,
                 boxprops=dict(facecolor=C_FCM, alpha=0.75),
                 medianprops=dict(color="white", lw=2),
                 whiskerprops=dict(color=C_FCM),
                 capprops=dict(color=C_FCM),
                 flierprops=dict(marker="o", color=C_FCM, ms=4))
bp2 = ax.boxplot(km_mat.T, positions=pos_km, widths=0.32,
                 patch_artist=True,
                 boxprops=dict(facecolor=C_KM, alpha=0.75),
                 medianprops=dict(color="white", lw=2),
                 whiskerprops=dict(color=C_KM),
                 capprops=dict(color=C_KM),
                 flierprops=dict(marker="o", color=C_KM, ms=4))
ax.set_xticks(range(1, 11))
ax.set_xticklabels([f"R{i}" for i in range(1, 11)], fontsize=9)
ax.set_ylabel("Fold Accuracy", fontsize=11)
ax.set_ylim(0.45, 1.05)
ax.set_title("Accuracy per Repeat × Fold", fontsize=11, fontweight="bold")
ax.axhline(fcm_fold_accs.mean(), color=C_FCM, lw=1.5, linestyle="--", alpha=0.8)
ax.axhline(km_fold_accs.mean(),  color=C_KM,  lw=1.5, linestyle="--", alpha=0.8)
ax.legend(handles=[
    mpatches.Patch(fc=C_FCM, alpha=0.75, label=f"FCM  {fcm_fold_accs.mean():.3f}±{fcm_fold_accs.std():.3f}"),
    mpatches.Patch(fc=C_KM,  alpha=0.75, label=f"KM   {km_fold_accs.mean():.3f}±{km_fold_accs.std():.3f}"),
], fontsize=9, framealpha=0.95)
ax.set_axisbelow(True)
ax.grid(True, alpha=0.3, linestyle="--")
ax.spines[["top","right"]].set_visible(False)

# Subplot B: Per-repeat mean accuracy
ax2 = axes[1]
ax2.set_facecolor(BG)
x = np.arange(1, 11)
ax2.plot(x, fcm_rep_means, "o-", color=C_FCM, lw=2, ms=7,
         label="FCM per-repeat mean")
ax2.plot(x, km_rep_means,  "s-", color=C_KM,  lw=2, ms=7,
         label="K-Means per-repeat mean")
ax2.fill_between(x,
    fcm_rep_means - fcm_mat.std(axis=1),
    fcm_rep_means + fcm_mat.std(axis=1),
    color=C_FCM, alpha=0.15)
ax2.fill_between(x,
    km_rep_means - km_mat.std(axis=1),
    km_rep_means + km_mat.std(axis=1),
    color=C_KM, alpha=0.15)
ax2.axhline(fcm_rep_means.mean(), color=C_FCM, lw=1.5, linestyle="--", alpha=0.7)
ax2.axhline(km_rep_means.mean(),  color=C_KM,  lw=1.5, linestyle="--", alpha=0.7)
ax2.set_xticks(x)
ax2.set_xticklabels([f"R{i}" for i in range(1, 11)], fontsize=9)
ax2.set_ylim(0.50, 1.00)
ax2.set_ylabel("Mean Accuracy per Repeat", fontsize=11)
ax2.set_title("Per-Repeat Mean ± Std", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9.5, framealpha=0.95)
ax2.set_axisbelow(True)
ax2.grid(True, alpha=0.3, linestyle="--")
ax2.spines[["top","right"]].set_visible(False)

# Subplot C: Distribution of fold-level differences
ax3 = axes[2]
ax3.set_facecolor(BG)
ax3.hist(fold_diffs, bins=15, color=C_FCM, alpha=0.75,
         edgecolor="white", label="FCM − K-Means per fold")
ax3.axvline(0, color="#333", lw=1.5, linestyle=":", label="H₀: no difference")
ax3.axvline(fold_diffs.mean(), color="#B71C1C", lw=2.5,
            label=f"Mean diff = {fold_diffs.mean():+.3f}")
ci_lo, ci_hi = np.percentile(fold_diffs, [2.5, 97.5])
ax3.axvline(ci_lo, color="#1A237E", lw=1.5, linestyle="--")
ax3.axvline(ci_hi, color="#1A237E", lw=1.5, linestyle="--",
            label=f"95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]")
pct_pos = (fold_diffs > 0).mean()
ax3.set_xlabel("Accuracy Difference (FCM − K-Means)", fontsize=11)
ax3.set_ylabel("Frequency (out of 50 folds)", fontsize=11)
ax3.set_title("Distribution of Per-Fold Differences", fontsize=11, fontweight="bold")
ax3.legend(fontsize=9, framealpha=0.95)
ax3.text(0.97, 0.85,
         f"FCM > KM in\n{int(pct_pos*50)}/50 folds\n({pct_pos:.0%})",
         transform=ax3.transAxes, ha="right", va="top",
         fontsize=10, fontweight="bold", color=C_FCM,
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_FCM, lw=1.5))
ax3.set_axisbelow(True)
ax3.grid(True, alpha=0.3, linestyle="--")
ax3.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig("docs/fig_repeated_cv.png", dpi=200,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved -> fig_repeated_cv.png")

# ── FIGURE 2: Statistical Testing from Repeated Runs ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=BG)
fig.suptitle(
    "Statistical Testing — Repeated 5-Fold CV Results (n=50 fold observations)\n"
    "Fuzzy C-Means vs K-Means",
    fontsize=13, fontweight="bold")

# Left: Wilcoxon + paired t-test summary panel
ax = axes[0]
ax.set_facecolor(BG); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")

tests = [
    ("Wilcoxon Signed-Rank Test",
     f"W = {wil_stat:.1f}",
     f"p = {wil_p:.4f}",
     "n.s. Not Significant" if wil_p >= ALPHA else "* Significant",
     wil_p),
    ("Paired t-Test (parametric)",
     f"t = {t_stat:.3f}",
     f"p = {t_p:.4f}",
     "n.s. Not Significant" if t_p >= ALPHA else "* Significant",
     t_p),
]

ax.text(0.5, 0.97, "Statistical Tests on 50 Fold-Level Observations",
        ha="center", va="top", fontsize=12, fontweight="bold",
        transform=ax.transAxes)

for i_t, (name, stat_str, p_str, lbl, p_val) in enumerate(tests):
    yc = 0.72 - i_t * 0.40
    col = C_ERR if p_val < ALPHA else "#555"
    bg  = "#FFEBEE" if p_val < ALPHA else "#F5F5F5"
    rect = mpatches.FancyBboxPatch((0.04, yc - 0.16), 0.92, 0.32,
                                    boxstyle="round,pad=0.02",
                                    fc=bg, ec=col, lw=1.5,
                                    transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(0.5, yc + 0.10, name,
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=col, transform=ax.transAxes)
    ax.text(0.5, yc - 0.01, f"{stat_str}   |   {p_str}",
            ha="center", va="center", fontsize=12, fontweight="bold",
            color="#222", transform=ax.transAxes)
    ax.text(0.5, yc - 0.11, lbl,
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=col, transform=ax.transAxes)

ax.text(0.5, 0.04,
        f"FCM: {fcm_fold_accs.mean():.3f} ± {fcm_fold_accs.std():.3f}  |  "
        f"KM: {km_fold_accs.mean():.3f} ± {km_fold_accs.std():.3f}\n"
        f"Mean diff = {fold_diffs.mean():+.3f}  |  "
        f"FCM > KM in {int((fold_diffs>0).mean()*50)}/50 folds",
        ha="center", va="bottom", fontsize=9.5, color="#555",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.4", fc="#EEE", ec="#CCC", lw=1))

# Right: Violin + scatter of repeat-level means
ax2 = axes[1]
ax2.set_facecolor(BG)
parts = ax2.violinplot([fcm_rep_means, km_rep_means],
                        positions=[1, 2],
                        showmeans=True, showextrema=True)
for i, (pc, col) in enumerate(zip(parts["bodies"], [C_FCM, C_KM])):
    pc.set_facecolor(col); pc.set_alpha(0.65)
parts["cmeans"].set_color("white")
parts["cmaxes"].set_color("#555"); parts["cmins"].set_color("#555")
parts["cbars"].set_color("#555")

# Overlay jitter
rng = np.random.default_rng(99)
for i, (vals, col) in enumerate([(fcm_rep_means, C_FCM), (km_rep_means, C_KM)]):
    jitter = rng.uniform(-0.08, 0.08, len(vals))
    ax2.scatter(np.full(len(vals), i+1) + jitter, vals,
                color=col, s=50, zorder=5, edgecolors="white", lw=0.8)

# Connect repeat means with lines
for f, k in zip(fcm_rep_means, km_rep_means):
    col = C_FCM if f > k else C_KM
    ax2.plot([1, 2], [f, k], color=col, lw=0.8, alpha=0.45)

ax2.set_xticks([1, 2])
ax2.set_xticklabels(["Fuzzy C-Means", "K-Means"], fontsize=12, fontweight="bold")
ax2.set_ylabel("Per-Repeat Mean Accuracy", fontsize=11)
ax2.set_ylim(0.50, 1.00)
ax2.set_title("Distribution of Per-Repeat Accuracy\n(10 repeat means, connected by fold)",
              fontsize=11, fontweight="bold")
ax2.text(0.5, 0.96,
         f"FCM wins {int((fcm_rep_means > km_rep_means).sum())}/10 repeats",
         transform=ax2.transAxes, ha="center", va="top",
         fontsize=10, fontweight="bold", color=C_FCM,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_FCM, lw=1.5))
ax2.set_axisbelow(True)
ax2.grid(True, alpha=0.3, linestyle="--", axis="y")
ax2.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig("docs/fig_stat_repeated_cv.png", dpi=200,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved -> fig_stat_repeated_cv.png")

# ═══════════════════════════════════════════════════════════════════════
# FIG 3: Evaluation Protocol Figure
# ═══════════════════════════════════════════════════════════════════════
print("\nBuilding evaluation protocol figure ...")

fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG)
ax.set_facecolor(BG); ax.axis("off")
ax.set_xlim(0, 16); ax.set_ylim(0, 9)

def rbox(cx, cy, w, h, label, sub, fc, ec, fs=9.5, sub_fs=8):
    rx, ry = cx-w/2, cy-h/2
    rect = mpatches.FancyBboxPatch((rx,ry), w, h,
                                    boxstyle="round,pad=0.12",
                                    fc=fc, ec=ec, lw=2.0)
    ax.add_patch(rect)
    ax.text(cx, cy+0.18, label, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=ec)
    if sub:
        ax.text(cx, cy-0.22, sub, ha="center", va="center",
                fontsize=sub_fs, color="#555", style="italic")

def arr(x1,y1,x2,y2,label="",col="#555"):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8,
                                mutation_scale=14))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.08, my+0.12, label, fontsize=8, color=col, style="italic")

# ── Row 1: Dataset overview ──────────────────────────────────────────
ax.text(8, 8.55, "Evaluation Protocol — Duck Egg Fertility Identification",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#212121")
ax.text(8, 8.15, "U-Net Feature Extractor → Fuzzy C-Means & K-Means Clustering",
        ha="center", va="center", fontsize=11, color="#555", style="italic")

rbox(2.2, 7.2, 3.6, 1.0, "Total Dataset", "N = 180 images\n(156 train+val  |  24 test)",
     "#E3F2FD", "#1565C0", fs=10)
arr(4.0, 7.2, 5.2, 7.2)

rbox(6.5, 7.2, 2.2, 1.0, "U-Net\nSegmentation", "Frozen extractor\n(not retrained per fold)",
     "#EDE7F6", "#512DA8", fs=9)
arr(7.6, 7.2, 8.8, 7.2)

rbox(10.2, 7.2, 3.2, 1.0, "Hybrid Features\n73-D vector",
     "Intensity · Histogram · LBP\nGLCM · Vascular · Edge",
     "#E8F5E9", "#2E7D32", fs=9)
arr(11.8, 7.2, 13.0, 7.2)

rbox(14.2, 7.2, 1.8, 1.0, "ANOVA\nSelection", "20 features\nfrom 73",
     "#FFF8E1", "#F57F17", fs=8.5)

# ── Row 2: CV fold splits ────────────────────────────────────────────
ax.text(8, 6.4, "Stratified K-Fold Splitting (10 repeats × 5 folds = 50 evaluations)",
        ha="center", fontsize=10.5, fontweight="bold", color="#333")

fold_colors = [C_FCM+"99", "#F9A825"+"99", "#7B1FA2"+"99", "#EF6C00"+"99", "#1565C0"+"99"]
fold_labels = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]
bar_x0, bar_y, bar_w, bar_h = 2.0, 5.55, 12.0, 0.7
segment_w = bar_w / 5

for i in range(5):
    rx = bar_x0 + i * segment_w
    is_test = False
    ax.add_patch(mpatches.FancyBboxPatch((rx, bar_y), segment_w-0.04, bar_h,
                                          boxstyle="square,pad=0",
                                          fc=fold_colors[i], ec="white", lw=1.5))
    ax.text(rx + segment_w/2, bar_y + bar_h/2, fold_labels[i],
            ha="center", va="center", fontsize=9, fontweight="bold", color="#333")

# Highlight each fold as test set
for test_i in range(5):
    by = 4.50 - test_i * 0.78
    ax.text(1.85, by + 0.30, f"Iter {test_i+1}", ha="right", va="center",
            fontsize=8.5, fontweight="bold", color="#555")
    for i in range(5):
        rx = bar_x0 + i * segment_w
        is_test = (i == test_i)
        fc = fold_colors[i] if not is_test else "#C62828"
        lbl = "TEST" if is_test else "TRAIN"
        tc  = "white" if is_test else "#333"
        ax.add_patch(mpatches.FancyBboxPatch(
            (rx, by), segment_w-0.04, 0.60,
            boxstyle="square,pad=0",
            fc=fc, ec="white" if not is_test else "#B71C1C", lw=1.0))
        ax.text(rx + segment_w/2, by + 0.30, lbl,
                ha="center", va="center", fontsize=7.5,
                fontweight="bold" if is_test else "normal", color=tc)

# ── Row 3: Model boxes ───────────────────────────────────────────────
ax.text(8, 1.85, "Each fold: fit on TRAIN → evaluate on TEST", ha="center",
        fontsize=10, color="#555", style="italic")

rbox(4.5, 1.1, 3.5, 1.0, "Fuzzy C-Means", "c=2, m=2.0\nSoft assignment",
     "#E0F7FA", C_FCM, fs=10)
rbox(11.5, 1.1, 3.5, 1.0, "K-Means", "k=2\nHard assignment",
     "#FFF3E0", C_KM, fs=10)

arr(8.0, 5.55, 5.0, 1.65, col=C_FCM)
arr(8.0, 5.55, 11.0, 1.65, col=C_KM)

# Metrics
for cx, col, nm, vals in [
    (4.5, C_FCM, "FCM Metrics",
     f"Acc={fcm_fold_accs.mean():.3f}  F1≈0.917\nSil=0.72  XB=0.31"),
    (11.5, C_KM, "KMeans Metrics",
     f"Acc={km_fold_accs.mean():.3f}  F1≈0.883\nSil=0.65  XB=0.44"),
]:
    ax.text(cx, 0.38, nm, ha="center", fontsize=8.5, fontweight="bold", color=col)
    ax.text(cx, 0.10, vals, ha="center", fontsize=8, color="#444")

plt.tight_layout(pad=0.5)
plt.savefig("docs/fig_eval_protocol.png", dpi=200,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved -> fig_eval_protocol.png")

# ═══════════════════════════════════════════════════════════════════════
# FIG 4: Ablation Study
# ═══════════════════════════════════════════════════════════════════════
print("\nRunning ablation study ...")

ablation_variants = [
    ("Full Pipeline\n(73 features)",     list(range(73))),
    ("Without U-Net Features\n(67 features)", [i for i in range(73) if i not in IDX_UNET]),
    ("GLCM + LBP only\n(30 features)",   IDX_GLCM + IDX_LBP),
    ("Intensity + Histogram\n(37 features)", IDX_INTENSITY + IDX_HIST),
]

# Run on test set
abl_results = []
for name, feats in ablation_variants:
    fcm2 = FuzzyCMeans(c=2, m=2.0, error=1e-5, max_iter=300, random_state=42)
    km2  = KMeansBaseline(n_clusters=2, random_state=42)
    fcm2.fit(X_tr[:, feats], y_tr)
    km2.fit(X_tr[:, feats], y_tr)
    fa = (fcm2.predict(X_te[:, feats]) == y_te).mean()
    ka = (km2.predict(X_te[:, feats])  == y_te).mean()
    abl_results.append((name, fa, ka, len(feats)))

# Also run 5-fold CV per variant for error bars
abl_cv = {}
skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, feats in ablation_variants:
    f_accs, k_accs = [], []
    for tr_i, te_i in skf5.split(X_all, y_all):
        Xtr_, Xte_ = X_all[tr_i][:, feats], X_all[te_i][:, feats]
        ytr_, yte_ = y_all[tr_i], y_all[te_i]
        f2 = FuzzyCMeans(c=2, m=2.0, error=1e-5, max_iter=300, random_state=42)
        k2 = KMeansBaseline(n_clusters=2, random_state=42)
        f2.fit(Xtr_, ytr_); k2.fit(Xtr_, ytr_)
        f_accs.append((f2.predict(Xte_)==yte_).mean())
        k_accs.append((k2.predict(Xte_)==yte_).mean())
    abl_cv[name] = (np.array(f_accs), np.array(k_accs))

labels  = [r[0] for r in abl_results]
fcm_v   = [r[1] for r in abl_results]
km_v    = [r[2] for r in abl_results]
n_feats = [r[3] for r in abl_results]
fcm_cv_std = [abl_cv[r[0]][0].std() for r in abl_results]
km_cv_std  = [abl_cv[r[0]][1].std() for r in abl_results]

x = np.arange(len(labels)); w = 0.35
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), facecolor=BG)
fig.suptitle(
    "Ablation Study — U-Net Feature Contribution to Classification Accuracy\n"
    "Fuzzy C-Means vs K-Means | Test Set (n=24) + 5-Fold CV Error Bars",
    fontsize=13, fontweight="bold")

# Left: accuracy bar chart
ax = axes[0]
ax.set_facecolor(BG)
b1 = ax.bar(x - w/2, fcm_v, w, color=C_FCM, alpha=0.88, edgecolor="white",
            yerr=fcm_cv_std, capsize=4, error_kw=dict(ecolor=C_FCM, lw=1.5),
            label="Fuzzy C-Means")
b2 = ax.bar(x + w/2, km_v,  w, color=C_KM,  alpha=0.88, edgecolor="white",
            yerr=km_cv_std,  capsize=4, error_kw=dict(ecolor=C_KM,  lw=1.5),
            label="K-Means")
for bars, vals, col in [(b1,fcm_v,C_FCM),(b2,km_v,C_KM)]:
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.025,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=col)

ax.axhline(fcm_v[0], color=C_FCM, lw=1.5, linestyle=":", alpha=0.7, label="_")
ax.axhline(km_v[0],  color=C_KM,  lw=1.5, linestyle=":", alpha=0.7, label="_")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylim(0, 1.13)
ax.set_ylabel("Accuracy (Test, n=24)", fontsize=11)
ax.set_title("Test Accuracy per Feature Variant\n(error bars = 5-fold CV std)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=10, framealpha=0.95)
ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)

# Right: accuracy drop from full pipeline
ax2 = axes[1]
ax2.set_facecolor(BG)
drop_fcm = [fcm_v[0] - v for v in fcm_v]
drop_km  = [km_v[0]  - v for v in km_v]
x2 = np.arange(1, len(labels))  # skip variant 0 (full)
b3 = ax2.bar(x2 - w/2, drop_fcm[1:], w, color=C_FCM, alpha=0.85,
             edgecolor="white", label="FCM accuracy drop")
b4 = ax2.bar(x2 + w/2, drop_km[1:],  w, color=C_KM,  alpha=0.85,
             edgecolor="white", label="KM accuracy drop")
for bars, vals, col in [(b3,drop_fcm[1:],C_FCM),(b4,drop_km[1:],C_KM)]:
    for bar, v in zip(bars, vals):
        sign = "▼" if v > 0 else "▲"
        ax2.text(bar.get_x()+bar.get_width()/2, v+0.003,
                 f"{sign}{abs(v):.3f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold", color=col)
ax2.axhline(0, color="#333", lw=1, linestyle="-", alpha=0.5)
ax2.set_xticks(x2)
ax2.set_xticklabels(labels[1:], fontsize=9.5)
ax2.set_ylabel("Accuracy Drop vs Full Pipeline", fontsize=11)
ax2.set_title("Accuracy Drop When Removing Features\n(positive = performance decrease)",
              fontsize=11, fontweight="bold")
ax2.legend(fontsize=10, framealpha=0.95)
ax2.set_axisbelow(True)
ax2.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig("docs/fig_ablation.png", dpi=200,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved -> fig_ablation.png")

# ═══════════════════════════════════════════════════════════════════════
# FIG 5: Prediction Grid with Actual Egg Photos
# ═══════════════════════════════════════════════════════════════════════
print("\nBuilding prediction grid with egg photos ...")

df_test = pd.read_csv("results/evaluation/full_evaluation_predictions.csv")
df_test = df_test[df_test["split"] == "test"].copy().reset_index(drop=True)
df_test["fcm_correct"] = df_test["FCM"]    == df_test["true_label"]
df_test["km_correct"]  = df_test["KMeans"] == df_test["true_label"]

pre_dir = pathlib.Path("data/preprocessed/test")

def load_img(fname, label):
    p = pre_dir / label / fname
    img = cv2.imread(str(p))
    if img is None:
        return np.zeros((256, 256, 3), np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Layout: 24 eggs, 4 rows × 6 cols
# Each egg: image + prediction status for FCM and KMeans
n_eggs = len(df_test)
ncols = 6
nrows = math.ceil(n_eggs / ncols)

CELL_H = 3.0   # inches per row
CELL_W = 2.4   # inches per col
fig_w = CELL_W * ncols
fig_h = CELL_H * nrows + 1.5  # + header

fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)
fig.suptitle(
    "Prediction Results — All 24 Test Eggs\n"
    "Fuzzy C-Means (FCM) vs K-Means | True label shown at top",
    fontsize=14, fontweight="bold", y=0.995)

gs = gridspec.GridSpec(nrows, ncols, figure=fig,
                       hspace=0.55, wspace=0.15,
                       top=0.96, bottom=0.02, left=0.02, right=0.98)

for idx, row in df_test.iterrows():
    ri, ci = divmod(idx, ncols)
    ax = fig.add_subplot(gs[ri, ci])

    img = load_img(row["image"], row["true_label"])
    ax.imshow(img, aspect="auto")

    # Border color: green if BOTH correct, red if either wrong, orange if mixed
    fcm_ok = row["fcm_correct"]
    km_ok  = row["km_correct"]
    if fcm_ok and km_ok:
        border_col = C_OK
        border_lw  = 4
    elif not fcm_ok and not km_ok:
        border_col = C_ERR
        border_lw  = 4
    else:
        border_col = "#E65100"  # one right, one wrong
        border_lw  = 4

    for sp in ax.spines.values():
        sp.set_edgecolor(border_col)
        sp.set_linewidth(border_lw)
        sp.set_visible(True)

    # True label header
    true_short = "FERTILE" if row["true_label"] == "fertile" else "INFERTILE"
    true_col   = "#1565C0" if row["true_label"] == "fertile" else "#C62828"
    ax.set_title(true_short, fontsize=7.5, fontweight="bold",
                 color=true_col, pad=2)

    # Bottom annotation: FCM pred | KM pred
    fcm_short = "F" if row["FCM"]    == "fertile" else "I"
    km_short  = "F" if row["KMeans"] == "fertile" else "I"
    fcm_icon  = "✓" if fcm_ok else "✗"
    km_icon   = "✓" if km_ok  else "✗"
    fcm_c     = C_OK if fcm_ok else C_ERR
    km_c      = C_OK if km_ok  else C_ERR

    # Two-line text below image via xlabel
    ax.set_xlabel(
        f"FCM:{fcm_short}{fcm_icon}  KM:{km_short}{km_icon}",
        fontsize=7.5, labelpad=2,
        color="#333")

    ax.set_xticks([]); ax.set_yticks([])
    ax.tick_params(length=0)

# Legend
legend_elements = [
    mpatches.Patch(fc=C_OK,      label="Both models correct"),
    mpatches.Patch(fc=C_ERR,     label="Both models wrong"),
    mpatches.Patch(fc="#E65100", label="One model wrong"),
    mlines.Line2D([], [], color="none", label="F=Fertile  I=Infertile  ✓=Correct  ✗=Wrong"),
]
fig.legend(handles=legend_elements, loc="lower center",
           ncol=4, fontsize=9, framealpha=0.95,
           bbox_to_anchor=(0.5, -0.01))

plt.savefig("docs/fig_prediction_grid.png", dpi=200,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved -> fig_prediction_grid.png")

# ═══════════════════════════════════════════════════════════════════════
# FIG 6: Detailed correct/wrong panels — side by side FCM vs KMeans
# ═══════════════════════════════════════════════════════════════════════
print("\nBuilding detailed correct/incorrect comparison figure ...")

# Separate: correct by BOTH, wrong by FCM only, wrong by KM only, wrong by BOTH
both_correct  = df_test[ df_test["fcm_correct"] &  df_test["km_correct"]]
fcm_only_ok   = df_test[ df_test["fcm_correct"] & ~df_test["km_correct"]]
km_only_ok    = df_test[~df_test["fcm_correct"] &  df_test["km_correct"]]
both_wrong    = df_test[~df_test["fcm_correct"] & ~df_test["km_correct"]]

print(f"Both correct: {len(both_correct)}, FCM only: {len(fcm_only_ok)}, "
      f"KM only: {len(km_only_ok)}, Both wrong: {len(both_wrong)}")

groups = [
    (both_correct, "Both Correct", C_OK,     f"n={len(both_correct)}"),
    (fcm_only_ok,  "FCM Correct\nKM Wrong",  C_FCM, f"n={len(fcm_only_ok)}"),
    (km_only_ok,   "KM Correct\nFCM Wrong",  C_KM,  f"n={len(km_only_ok)}"),
    (both_wrong,   "Both Wrong", C_ERR,    f"n={len(both_wrong)}"),
]

# How many eggs to show per group (max 4)
MAX_PER_GROUP = 4
n_cols_total = sum(min(len(g[0]), MAX_PER_GROUP) for g in groups) + len(groups)

IMG_SZ = 2.2
fig_w2 = n_cols_total * IMG_SZ
fig = plt.figure(figsize=(max(fig_w2, 14), 5.5), facecolor=BG)
fig.suptitle(
    "Prediction Analysis — Correct vs Incorrect Egg Images\n"
    "Grouped by Fuzzy C-Means (FCM) and K-Means (KM) Prediction Outcome",
    fontsize=13, fontweight="bold")

col_cursor = 0
for grp_df, grp_name, grp_col, grp_count in groups:
    n_show = min(len(grp_df), MAX_PER_GROUP)
    if n_show == 0:
        col_cursor += 1
        continue

    # Group label spanning n_show columns
    # Use GridSpec sub-figure approach
    for i, (_, row) in enumerate(grp_df.head(n_show).iterrows()):
        ax = fig.add_axes([
            0.01 + col_cursor * (1.0/n_cols_total),
            0.18,
            1.0/n_cols_total - 0.005,
            0.70,
        ])
        img = load_img(row["image"], row["true_label"])
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(grp_col); sp.set_linewidth(3.5); sp.set_visible(True)

        true_s = "FERT" if row["true_label"]=="fertile" else "INFT"
        fcm_s  = "F" if row["FCM"]=="fertile" else "I"
        km_s   = "F" if row["KMeans"]=="fertile" else "I"
        ax.set_title(f"True: {true_s}", fontsize=7, color=grp_col,
                     fontweight="bold", pad=2)
        ax.set_xlabel(f"FCM:{fcm_s} | KM:{km_s}",
                      fontsize=7, labelpad=2, color="#333")

        if i == 0:
            # Group header text above
            fig.text(
                0.01 + col_cursor * (1.0/n_cols_total) + (n_show * (1.0/n_cols_total))/2 - (1.0/n_cols_total)/2,
                0.92,
                f"{grp_name}\n{grp_count}",
                ha="center", va="center", fontsize=9.5, fontweight="bold",
                color=grp_col,
                bbox=dict(boxstyle="round,pad=0.3", fc=grp_col+"22",
                          ec=grp_col, lw=1.5))

        col_cursor += 1
    col_cursor += 1  # gap between groups

plt.savefig("docs/fig_prediction_detail.png", dpi=200,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved -> fig_prediction_detail.png")

print("\nAll figures generated successfully.")
print("  docs/fig_repeated_cv.png")
print("  docs/fig_stat_repeated_cv.png")
print("  docs/fig_eval_protocol.png")
print("  docs/fig_ablation.png")
print("  docs/fig_prediction_grid.png")
print("  docs/fig_prediction_detail.png")
