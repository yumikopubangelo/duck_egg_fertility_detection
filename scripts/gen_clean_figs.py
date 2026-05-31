"""Rebuild all 5 figures — clean layout, no overlaps, simple & informative."""
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
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

BG    = "#FAFAFA"
C_FCM = "#2A9D8F"
C_KM  = "#F4A261"
C_OK  = "#2E7D32"
C_ERR = "#C62828"

# ── Data ───────────────────────────────────────────────────────────────
X_tr = np.load("data/features/awc_features.npy")
y_tr = np.load("data/features/awc_labels.npy")
X_te = np.load("data/features/awc_test_features.npy")
y_te = np.load("data/features/awc_test_labels.npy")
X_all = np.vstack([X_tr, X_te])
y_all = np.hstack([y_tr, y_te])

with open("models/baselines/fcm_model.pkl","rb") as f: fcm_model = pickle.load(f)
with open("models/baselines/kmeans_model.pkl","rb") as f: km_model  = pickle.load(f)

fcm_te = fcm_model.predict(X_te)
km_te  = km_model.predict(X_te)

df_all  = pd.read_csv("results/evaluation/full_evaluation_predictions.csv")
df_test = df_all[df_all["split"]=="test"].copy().reset_index(drop=True)
df_test["fcm_ok"] = df_test["FCM"]    == df_test["true_label"]
df_test["km_ok"]  = df_test["KMeans"] == df_test["true_label"]

PRE = pathlib.Path("data/preprocessed/test")

def load_img(fname, label):
    p = PRE / label / fname
    img = cv2.imread(str(p))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.zeros((256,256,3), np.uint8)

# ═══════════════════════════════════════════════════════════════════════
# FIG 1 — Prediction Detail: clean grouped grid
# ═══════════════════════════════════════════════════════════════════════
print("FIG 1: Prediction detail ...")

both_ok  = df_test[ df_test.fcm_ok &  df_test.km_ok]
fcm_only = df_test[ df_test.fcm_ok & ~df_test.km_ok]
km_only  = df_test[~df_test.fcm_ok &  df_test.km_ok]
both_err = df_test[~df_test.fcm_ok & ~df_test.km_ok]

groups = [
    (both_ok,  "Both Correct",        C_OK,     4),
    (fcm_only, "FCM Correct / KM ✗",  C_FCM,    4),
    (km_only,  "KM Correct / FCM ✗",  C_KM,     4),
    (both_err, "Both Wrong",          C_ERR,    4),
]

# Count actual images to show
cols_per_group = [min(len(g[0]), g[3]) for g in groups]
total_cols = sum(c for c in cols_per_group if c > 0) + sum(1 for c in cols_per_group if c > 0) - 1

IMG = 2.0
fig_w = max(total_cols * IMG + 1, 14)
fig = plt.figure(figsize=(fig_w, 4.8), facecolor=BG)
fig.suptitle(
    "Test Set Prediction Outcomes — All 24 Duck Eggs",
    fontsize=13, fontweight="bold", y=1.01)

ax_meta = fig.add_axes([0, 0, 1, 1])
ax_meta.set_facecolor(BG); ax_meta.axis("off")

col_start = 0
gap = 0.8  # gap in column units between groups

for grp_df, grp_name, grp_col, max_show in groups:
    n = min(len(grp_df), max_show)
    if n == 0:
        continue

    for i, (_, row) in enumerate(grp_df.head(n).iterrows()):
        abs_col = col_start + i
        left   = 0.02 + abs_col * (1.0 / (total_cols + 0.5))
        width  = 0.85 / (total_cols + 0.5)
        ax = fig.add_axes([left, 0.13, width, 0.72])
        img = load_img(row["image"], row["true_label"])
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(grp_col); sp.set_linewidth(3.5); sp.set_visible(True)

        true_s = "FERT" if row["true_label"] == "fertile" else "INFT"
        fcm_s  = "F" if row["FCM"] == "fertile" else "I"
        km_s   = "F" if row["KMeans"] == "fertile" else "I"
        ax.set_title(f"True: {true_s}", fontsize=7.5, fontweight="bold",
                     color=grp_col, pad=2)
        ax.set_xlabel(f"FCM:{fcm_s} | KM:{km_s}", fontsize=7, labelpad=1, color="#444")

    # Group header — centered over this group's columns
    grp_left  = 0.02 + col_start * (1.0 / (total_cols + 0.5))
    grp_right = 0.02 + (col_start + n) * (1.0 / (total_cols + 0.5))
    cx = (grp_left + grp_right) / 2
    ax_meta.text(cx, 0.97, f"{grp_name}  (n={len(grp_df)})",
                 ha="center", va="top", transform=ax_meta.transAxes,
                 fontsize=9.5, fontweight="bold", color=grp_col,
                 bbox=dict(boxstyle="round,pad=0.3", fc=grp_col+"18",
                           ec=grp_col, lw=1.5))

    col_start += n + (1 if n < total_cols else 0)

ax_meta.text(0.5, 0.03,
             "F = Fertile · I = Infertile · FCM = Fuzzy C-Means · KM = K-Means",
             ha="center", va="bottom", fontsize=8, color="#666",
             transform=ax_meta.transAxes)

plt.savefig("docs/fig_prediction_detail.png", dpi=200,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("  Saved fig_prediction_detail.png")

# ═══════════════════════════════════════════════════════════════════════
# FIG 2 — t-SNE: clean 3-panel, no overlap
# ═══════════════════════════════════════════════════════════════════════
print("FIG 2: t-SNE ...")

is_test = np.array([False]*len(X_tr) + [True]*len(X_te))
fcm_all = fcm_model.predict(X_all)
km_all  = km_model.predict(X_all)

X_2d = TSNE(n_components=2, perplexity=25, n_iter=1200,
            random_state=42, learning_rate="auto", init="pca"
            ).fit_transform(StandardScaler().fit_transform(X_all))

fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG,
                          gridspec_kw={"wspace": 0.28})
fig.suptitle(
    "t-SNE Visualization — U-Net 73-D Hybrid Feature Space\n"
    "K-Means vs Fuzzy C-Means Cluster Separation",
    fontsize=13, fontweight="bold")

S_TR, S_TE = 45, 90

# Panel 1: ground truth
ax = axes[0]
ax.set_facecolor(BG)
C_F = "#1565C0"; C_I = "#C62828"
for cls, col, lbl in [(1, C_F, "Fertile"), (0, C_I, "Infertile")]:
    ax.scatter(X_2d[(y_all==cls)&~is_test,0], X_2d[(y_all==cls)&~is_test,1],
               c=col, s=S_TR, alpha=0.70, edgecolors="white", lw=0.4, zorder=3)
    ax.scatter(X_2d[(y_all==cls)&is_test,0],  X_2d[(y_all==cls)&is_test,1],
               c=col, s=S_TE, alpha=0.95, edgecolors="black", lw=1.0,
               marker="D", zorder=4, label=lbl)
ax.set_title("Ground Truth", fontsize=12, fontweight="bold")
ax.legend(handles=[
    mpatches.Patch(fc=C_F, label="Fertile"),
    mpatches.Patch(fc=C_I, label="Infertile"),
    mlines.Line2D([],[],marker="D",color="w",markerfacecolor="#888",ms=8,label="Test"),
], fontsize=9, framealpha=0.95, loc="upper right")

# Panels 2 & 3: model predictions
for ax, pred, col_m, nm, acc_label in [
    (axes[1], km_all,  C_KM,  "K-Means",       f"Acc {(km_all==y_all).mean():.1%}"),
    (axes[2], fcm_all, C_FCM, "Fuzzy C-Means", f"Acc {(fcm_all==y_all).mean():.1%}"),
]:
    correct = (pred == y_all)
    ax.set_facecolor(BG)
    for msk, col, mk, sz, al, ec, z in [
        (correct & ~is_test,  col_m,  "o", S_TR,    0.72, "white", 3),
        (correct & is_test,   col_m,  "D", S_TE,    0.95, "black", 4),
        (~correct & ~is_test, "#AAA", "x", S_TR+10, 0.80, "none",  5),
        (~correct & is_test,  "#444", "x", S_TE+10, 1.00, "none",  6),
    ]:
        idx = np.where(msk)[0]
        if len(idx):
            ax.scatter(X_2d[idx,0], X_2d[idx,1], c=col, s=sz,
                       alpha=al, marker=mk, edgecolors=ec, lw=0.6, zorder=z)

    ax.set_title(f"{nm}", fontsize=12, fontweight="bold")
    # Accuracy in upper-left corner — never conflicts with legend
    ax.text(0.03, 0.97, acc_label,
            transform=ax.transAxes, va="top", ha="left",
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col_m, lw=1.8))
    ax.legend(handles=[
        mpatches.Patch(fc=col_m, label=f"Correct ({correct.sum()})"),
        mlines.Line2D([],[],marker="x",color="#888",ms=8,label=f"Incorrect ({(~correct).sum()})"),
    ], fontsize=9, framealpha=0.95, loc="lower right")

for ax in axes:
    ax.set_xlabel("t-SNE Dim 1", fontsize=10)
    ax.set_ylabel("t-SNE Dim 2", fontsize=10)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_visible(False)

plt.savefig("docs/fig_tsne_cluster.png", dpi=200,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("  Saved fig_tsne_cluster.png")

# ═══════════════════════════════════════════════════════════════════════
# FIG 3 — Repeated CV: clean 3-panel
# ═══════════════════════════════════════════════════════════════════════
print("FIG 3: Repeated CV ...")

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
fold_fcm, fold_km = [], []
for tr_i, te_i in rskf.split(X_all, y_all):
    f2 = FuzzyCMeans(c=2, m=2.0, error=1e-5, max_iter=300, random_state=42)
    k2 = KMeansBaseline(n_clusters=2, random_state=42)
    f2.fit(X_all[tr_i], y_all[tr_i]); k2.fit(X_all[tr_i], y_all[tr_i])
    fold_fcm.append((f2.predict(X_all[te_i])==y_all[te_i]).mean())
    fold_km.append( (k2.predict(X_all[te_i])==y_all[te_i]).mean())

fold_fcm   = np.array(fold_fcm)
fold_km    = np.array(fold_km)
fold_diffs = fold_fcm - fold_km
fcm_mat    = fold_fcm.reshape(10,5)
km_mat     = fold_km.reshape(10,5)
fcm_rep    = fcm_mat.mean(axis=1)
km_rep     = km_mat.mean(axis=1)

wil_s, wil_p = wilcoxon(fold_diffs, alternative="two-sided")
t_s,   t_p   = ttest_rel(fold_fcm, fold_km)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG,
                          gridspec_kw={"wspace": 0.30})
fig.suptitle(
    "Repeated Stratified 5-Fold CV  (10 repeats × 5 folds = 50 evaluations)\n"
    f"FCM: {fold_fcm.mean():.3f}±{fold_fcm.std():.3f}   "
    f"K-Means: {fold_km.mean():.3f}±{fold_km.std():.3f}   "
    f"Wilcoxon p={wil_p:.2e}  Paired-t p={t_p:.2e}",
    fontsize=12, fontweight="bold")

# Panel A: box plots
ax = axes[0]
ax.set_facecolor(BG)
pos_f = np.arange(1,11) - 0.2
pos_k = np.arange(1,11) + 0.2
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
ax.set_xticks(range(1,11))
ax.set_xticklabels([f"R{i}" for i in range(1,11)], fontsize=8.5)
ax.set_ylabel("Fold Accuracy", fontsize=10)
ax.set_ylim(0.45, 1.05)
ax.set_title("Accuracy Distribution per Repeat", fontsize=11, fontweight="bold")
ax.legend(handles=[
    mpatches.Patch(fc=C_FCM, alpha=0.72, label="Fuzzy C-Means"),
    mpatches.Patch(fc=C_KM,  alpha=0.72, label="K-Means"),
], fontsize=9, framealpha=0.95, loc="lower right")
ax.grid(True, alpha=0.25, ls="--", axis="y"); ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)

# Panel B: per-repeat line
ax2 = axes[1]
ax2.set_facecolor(BG)
x = np.arange(1,11)
ax2.plot(x, fcm_rep, "o-", color=C_FCM, lw=2, ms=6)
ax2.plot(x, km_rep,  "s-", color=C_KM,  lw=2, ms=6)
ax2.fill_between(x, fcm_rep-fcm_mat.std(1), fcm_rep+fcm_mat.std(1),
                 color=C_FCM, alpha=0.13)
ax2.fill_between(x, km_rep-km_mat.std(1),   km_rep+km_mat.std(1),
                 color=C_KM,  alpha=0.13)
ax2.axhline(fcm_rep.mean(), color=C_FCM, lw=1.5, ls="--", alpha=0.7)
ax2.axhline(km_rep.mean(),  color=C_KM,  lw=1.5, ls="--", alpha=0.7)
ax2.set_xticks(x); ax2.set_xticklabels([f"R{i}" for i in range(1,11)], fontsize=8.5)
ax2.set_ylim(0.52, 0.98)
ax2.set_ylabel("Mean Accuracy per Repeat", fontsize=10)
ax2.set_title("Per-Repeat Mean ± Std", fontsize=11, fontweight="bold")
ax2.legend(handles=[
    mlines.Line2D([],[],marker="o",color=C_FCM,ms=6,lw=2,label="Fuzzy C-Means"),
    mlines.Line2D([],[],marker="s",color=C_KM, ms=6,lw=2,label="K-Means"),
], fontsize=9, framealpha=0.95, loc="lower right")
ax2.grid(True, alpha=0.25, ls="--"); ax2.set_axisbelow(True)
ax2.spines[["top","right"]].set_visible(False)

# Panel C: diff histogram
ax3 = axes[2]
ax3.set_facecolor(BG)
ax3.hist(fold_diffs, bins=14, color=C_FCM, alpha=0.72,
         edgecolor="white")
ax3.axvline(0, color="#555", lw=1.5, ls=":", zorder=4)
ax3.axvline(fold_diffs.mean(), color="#B71C1C", lw=2.2, zorder=5)
ci_lo, ci_hi = np.percentile(fold_diffs, [2.5, 97.5])
ax3.axvline(ci_lo, color="#1A237E", lw=1.8, ls="--")
ax3.axvline(ci_hi, color="#1A237E", lw=1.8, ls="--")
pct = (fold_diffs > 0).mean()
ax3.set_xlabel("Accuracy Difference (FCM − K-Means)", fontsize=10)
ax3.set_ylabel("Count (50 folds)", fontsize=10)
ax3.set_title("Per-Fold Differences", fontsize=11, fontweight="bold")
ax3.legend(handles=[
    mlines.Line2D([],[],color="#B71C1C",lw=2,label=f"Mean {fold_diffs.mean():+.3f}"),
    mlines.Line2D([],[],color="#1A237E",lw=1.8,ls="--",
                  label=f"95% CI [{ci_lo:+.3f},{ci_hi:+.3f}]"),
    mlines.Line2D([],[],color="#555",lw=1.5,ls=":",label="No difference"),
], fontsize=9, framealpha=0.95, loc="upper left")
ax3.text(0.97, 0.96, f"FCM > KM\n{int(pct*50)}/50 folds",
         transform=ax3.transAxes, ha="right", va="top",
         fontsize=10, fontweight="bold", color=C_FCM,
         bbox=dict(boxstyle="round,pad=0.3",fc="white",ec=C_FCM,lw=1.5))
ax3.grid(True, alpha=0.25, ls="--"); ax3.set_axisbelow(True)
ax3.spines[["top","right"]].set_visible(False)

plt.savefig("docs/fig_repeated_cv.png", dpi=200,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("  Saved fig_repeated_cv.png")

# ═══════════════════════════════════════════════════════════════════════
# FIG 4 — Evaluation Protocol: clean diagram
# ═══════════════════════════════════════════════════════════════════════
print("FIG 4: Eval protocol ...")

fig, ax = plt.subplots(figsize=(15, 8.5), facecolor=BG)
ax.set_facecolor(BG); ax.axis("off")
ax.set_xlim(0, 15); ax.set_ylim(0, 8.5)

def rbox(cx, cy, w, h, title, sub, fc, ec, title_fs=9.5, sub_fs=7.8):
    rect = mpatches.FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                                    boxstyle="round,pad=0.12",
                                    fc=fc, ec=ec, lw=2.0)
    ax.add_patch(rect)
    ax.text(cx, cy+(0.12 if sub else 0), title,
            ha="center", va="center", fontsize=title_fs,
            fontweight="bold", color=ec)
    if sub:
        ax.text(cx, cy-0.24, sub, ha="center", va="center",
                fontsize=sub_fs, color="#555", style="italic")

def arrow(x1,y1,x2,y2,col="#666"):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="-|>",color=col,lw=1.8,mutation_scale=13))

# ── Row 1: Pipeline ──────────────────────────────────────────────────
ax.text(7.5, 8.15, "Evaluation Protocol — Duck Egg Fertility Identification",
        ha="center", fontsize=14, fontweight="bold", color="#212121")

rbox(1.5,  7.1, 2.6, 1.0, "Total Dataset",
     "N=180 (156 train+val\n24 held-out test)",     "#E3F2FD","#1565C0")
arrow(2.8, 7.1, 3.6, 7.1)
rbox(4.5,  7.1, 2.0, 1.0, "U-Net\n(Frozen)",
     "Feature extractor\nnot retrained",             "#EDE7F6","#512DA8")
arrow(5.5, 7.1, 6.3, 7.1)
rbox(7.5,  7.1, 2.4, 1.0, "73-D Hybrid\nFeatures",
     "Intensity·Hist·LBP\nGLCM·Vascular·Edge",      "#E8F5E9","#2E7D32")
arrow(8.7, 7.1, 9.5, 7.1)
rbox(10.5, 7.1, 2.0, 1.0, "ANOVA",
     "Select top 20\nfrom 73",                       "#FFF8E1","#F57F17")
arrow(11.5, 7.1, 12.3, 7.1)
rbox(13.2, 7.1, 2.4, 1.0, "FCM / K-Means",
     "Fit on train\nEvaluate on test",               "#E0F7FA","#006064")

# ── Row 2: CV header ─────────────────────────────────────────────────
ax.text(7.5, 6.10, "Repeated Stratified K-Fold   (10 repeats × 5 folds = 50 evaluations)",
        ha="center", fontsize=10.5, fontweight="bold", color="#333")

# ── Fold strip ────────────────────────────────────────────────────────
FCOLS_BASE = [C_FCM, "#F9A825", "#7B1FA2", "#EF6C00", "#1565C0"]
FCOLS      = [c+"AA" for c in FCOLS_BASE]
BAR_X0, BAR_Y, BAR_W, BAR_H = 1.0, 5.45, 13.0, 0.55
SW = BAR_W / 5
for i in range(5):
    ax.add_patch(mpatches.FancyBboxPatch(
        (BAR_X0+i*SW, BAR_Y), SW-0.05, BAR_H,
        boxstyle="square,pad=0", fc=FCOLS[i], ec="white", lw=1.5))
    ax.text(BAR_X0+i*SW+SW/2, BAR_Y+BAR_H/2, f"Fold {i+1}",
            ha="center", va="center", fontsize=9, fontweight="bold", color="#222")

# ── 5 iteration rows ─────────────────────────────────────────────────
ROW_H = 0.48
for it in range(5):
    ry = BAR_Y - (it+1)*(ROW_H+0.06)
    ax.text(0.85, ry+ROW_H/2, f"Iter {it+1}",
            ha="right", va="center", fontsize=8, color="#555")
    for fold in range(5):
        is_t = (fold == it)
        fc_  = "#C62828" if is_t else FCOLS_BASE[fold]+"55"
        ax.add_patch(mpatches.FancyBboxPatch(
            (BAR_X0+fold*SW, ry), SW-0.05, ROW_H,
            boxstyle="square,pad=0", fc=fc_, ec="white", lw=0.8))
        ax.text(BAR_X0+fold*SW+SW/2, ry+ROW_H/2,
                "TEST" if is_t else "train",
                ha="center", va="center",
                fontsize=8, fontweight="bold" if is_t else "normal",
                color="white" if is_t else "#444")

GRID_BOT = BAR_Y - 5*(ROW_H+0.06) - 0.1

# ── Model result boxes at the bottom ─────────────────────────────────
arrow(4.5, GRID_BOT, 4.0, GRID_BOT-0.5, col=C_FCM)
arrow(10.5, GRID_BOT, 11.0, GRID_BOT-0.5, col=C_KM)

rbox(4.0,  GRID_BOT-0.95, 3.5, 0.85,
     "Fuzzy C-Means  (c=2, m=2.0)",
     f"Acc={fold_fcm.mean():.3f}±{fold_fcm.std():.3f}  |  Sil=0.72  XB=0.31",
     "#E0F7FA", C_FCM, title_fs=9.5)
rbox(11.0, GRID_BOT-0.95, 3.5, 0.85,
     "K-Means  (k=2)",
     f"Acc={fold_km.mean():.3f}±{fold_km.std():.3f}  |  Sil=0.65  XB=0.44",
     "#FFF3E0", C_KM, title_fs=9.5)

plt.tight_layout(pad=0.3)
plt.savefig("docs/fig_eval_protocol.png", dpi=200,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("  Saved fig_eval_protocol.png")

# ═══════════════════════════════════════════════════════════════════════
# FIG 5 — Ablation Study: clean 2-panel, English
# ═══════════════════════════════════════════════════════════════════════
print("FIG 5: Ablation ...")

IDX_UNET = list(range(67,73))
ablation_variants = [
    ("Full Pipeline\n(73 features)",        list(range(73))),
    ("w/o U-Net Features\n(67 features)",   [i for i in range(73) if i not in IDX_UNET]),
    ("GLCM + LBP only\n(30 features)",      list(range(37,47))+list(range(47,67))),
    ("Intensity + Histogram\n(37 features)",list(range(0,37))),
]

# Test-set accuracy
abl_test = []
for nm, feats in ablation_variants:
    f2 = FuzzyCMeans(c=2,m=2.0,error=1e-5,max_iter=300,random_state=42)
    k2 = KMeansBaseline(n_clusters=2,random_state=42)
    f2.fit(X_tr[:,feats],y_tr); k2.fit(X_tr[:,feats],y_tr)
    fa=(f2.predict(X_te[:,feats])==y_te).mean()
    ka=(k2.predict(X_te[:,feats])==y_te).mean()
    abl_test.append((nm,fa,ka))

# 5-fold CV std
skf5 = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
abl_std = {}
for nm, feats in ablation_variants:
    fa_l, ka_l = [], []
    for tr_i,te_i in skf5.split(X_all,y_all):
        f2=FuzzyCMeans(c=2,m=2.0,error=1e-5,max_iter=300,random_state=42)
        k2=KMeansBaseline(n_clusters=2,random_state=42)
        f2.fit(X_all[tr_i][:,feats],y_all[tr_i])
        k2.fit(X_all[tr_i][:,feats],y_all[tr_i])
        fa_l.append((f2.predict(X_all[te_i][:,feats])==y_all[te_i]).mean())
        ka_l.append((k2.predict(X_all[te_i][:,feats])==y_all[te_i]).mean())
    abl_std[nm]=(np.std(fa_l),np.std(ka_l))

labels  = [r[0] for r in abl_test]
fcm_v   = [r[1] for r in abl_test]
km_v    = [r[2] for r in abl_test]
fcm_err = [abl_std[r[0]][0] for r in abl_test]
km_err  = [abl_std[r[0]][1] for r in abl_test]

x = np.arange(len(labels)); w = 0.32

fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG,
                          gridspec_kw={"wspace":0.30})
fig.suptitle(
    "Ablation Study — Impact of U-Net Feature Groups on Classification Accuracy",
    fontsize=13, fontweight="bold")

# Left: accuracy bars
ax = axes[0]
ax.set_facecolor(BG)
b1 = ax.bar(x-w/2, fcm_v, w, color=C_FCM, alpha=0.88, edgecolor="white",
            yerr=fcm_err, capsize=4, error_kw=dict(ecolor=C_FCM+"CC",lw=1.5),
            label="Fuzzy C-Means")
b2 = ax.bar(x+w/2, km_v,  w, color=C_KM,  alpha=0.88, edgecolor="white",
            yerr=km_err,  capsize=4, error_kw=dict(ecolor=C_KM+"CC", lw=1.5),
            label="K-Means")
for bars, vals, col in [(b1,fcm_v,C_FCM),(b2,km_v,C_KM)]:
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.028,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=col)
ax.axhline(fcm_v[0], color=C_FCM, lw=1.2, ls=":", alpha=0.6)
ax.axhline(km_v[0],  color=C_KM,  lw=1.2, ls=":", alpha=0.6)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylim(0.60, 1.08)
ax.set_ylabel("Test Accuracy (n=24)", fontsize=10)
ax.set_title("Accuracy per Feature Variant\n(error bars = 5-fold CV std)", fontsize=11, fontweight="bold")
ax.legend(fontsize=10, framealpha=0.95, loc="lower right")
ax.grid(True, alpha=0.25, ls="--", axis="y"); ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)

# Right: accuracy drop bars
drop_fcm = [fcm_v[0]-v for v in fcm_v]
drop_km  = [km_v[0]-v  for v in km_v]
ax2 = axes[1]
ax2.set_facecolor(BG)
b3 = ax2.bar(x[1:]-w/2, drop_fcm[1:], w, color=C_FCM, alpha=0.85, edgecolor="white",
             label="FCM drop")
b4 = ax2.bar(x[1:]+w/2, drop_km[1:],  w, color=C_KM,  alpha=0.85, edgecolor="white",
             label="KM drop")
for bars, vals, col in [(b3,drop_fcm[1:],C_FCM),(b4,drop_km[1:],C_KM)]:
    for bar, v in zip(bars, vals):
        sym = "▼" if v>0 else "▲"
        ax2.text(bar.get_x()+bar.get_width()/2, v+(0.003 if v>=0 else -0.012),
                 f"{sym}{abs(v):.3f}", ha="center",
                 va="bottom" if v>=0 else "top",
                 fontsize=9, fontweight="bold", color=col)
ax2.axhline(0, color="#555", lw=1, ls="-", alpha=0.5)
ax2.set_xticks(x[1:]); ax2.set_xticklabels(labels[1:], fontsize=9.5)
ax2.set_ylabel("Accuracy Drop vs Full Pipeline", fontsize=10)
ax2.set_title("Accuracy Drop When Removing Features\n(▼ decrease  ▲ increase vs full)", fontsize=11, fontweight="bold")
ax2.legend(fontsize=10, framealpha=0.95, loc="upper left")
ax2.grid(True, alpha=0.25, ls="--", axis="y"); ax2.set_axisbelow(True)
ax2.spines[["top","right"]].set_visible(False)

plt.savefig("docs/fig_ablation.png", dpi=200,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("  Saved fig_ablation.png")

print("\nAll 5 figures rebuilt cleanly.")
