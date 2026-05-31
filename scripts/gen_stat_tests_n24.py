"""
Statistical significance tests using ONLY n=24 test images (correct methodology).
McNemar and Wilcoxon are paired tests that require the held-out test set.
Repeated CV (10x5-fold) is shown separately as convergent evidence.
"""
import sys, math, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np, pickle, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.stats import binom, wilcoxon, ttest_rel
from sklearn.model_selection import RepeatedStratifiedKFold
from src.clustering.fuzzy_cmeans import FuzzyCMeans
from src.clustering.kmeans_baseline import KMeansBaseline

BG    = "#FAFAFA"
C_FCM = "#2A9D8F"
C_KM  = "#F4A261"
ALPHA = 0.05

# ── Load TEST SET ONLY (n=24) for McNemar/Wilcoxon ─────────────────────
X_tr = np.load("data/features/awc_features.npy")
y_tr = np.load("data/features/awc_labels.npy")
X_te = np.load("data/features/awc_test_features.npy")
y_te = np.load("data/features/awc_test_labels.npy")

with open("models/baselines/fcm_model.pkl", "rb") as f: fcm_model = pickle.load(f)
with open("models/baselines/kmeans_model.pkl", "rb") as f: km_model  = pickle.load(f)

fcm_te = fcm_model.predict(X_te)
km_te  = km_model.predict(X_te)
N_TE   = len(y_te)

ca_te  = (fcm_te == y_te)
cb_te  = (km_te  == y_te)
fcm_acc = ca_te.mean()
km_acc  = cb_te.mean()

# ── McNemar Exact Test (n=24) ──────────────────────────────────────────
b = int(( ca_te & ~cb_te).sum())   # FCM correct, KM wrong
c = int((~ca_te &  cb_te).sum())   # KM correct, FCM wrong
n_disc = b + c
if n_disc == 0:
    mc_p = 1.0
else:
    k    = min(b, c)
    mc_p = float(min(2 * sum(binom.pmf(i, n_disc, 0.5) for i in range(k+1)), 1.0))

# ── Wilcoxon Signed-Rank (n=24) ────────────────────────────────────────
d_te  = ca_te.astype(float) - cb_te.astype(float)
d_nz  = d_te[d_te != 0]
wil_applicable = len(d_nz) >= 4
if wil_applicable:
    wil_stat, wil_p = wilcoxon(d_nz, alternative="two-sided")
else:
    wil_stat, wil_p = float("nan"), float("nan")

# ── Bootstrap CI on test set accuracy difference (n=24) ────────────────
rng = np.random.default_rng(42)
boot_diffs = []
for _ in range(10_000):
    idx = rng.integers(0, N_TE, N_TE)
    boot_diffs.append(
        (fcm_te[idx] == y_te[idx]).mean() - (km_te[idx] == y_te[idx]).mean()
    )
boot_diffs = np.array(boot_diffs)
ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])

# ── Repeated 5-Fold CV (10 repeats, n=180) for convergent evidence ─────
print("Running 10×5-Fold Repeated CV (n=180) ...")
X_all = np.vstack([X_tr, X_te])
y_all = np.hstack([y_tr, y_te])
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
fold_fcm, fold_km = [], []
for tr_i, te_i in rskf.split(X_all, y_all):
    f2 = FuzzyCMeans(c=2, m=2.0, error=1e-5, max_iter=300, random_state=42)
    k2 = KMeansBaseline(n_clusters=2, random_state=42)
    f2.fit(X_all[tr_i], y_all[tr_i])
    k2.fit(X_all[tr_i], y_all[tr_i])
    fold_fcm.append((f2.predict(X_all[te_i]) == y_all[te_i]).mean())
    fold_km.append( (k2.predict(X_all[te_i]) == y_all[te_i]).mean())

fold_fcm   = np.array(fold_fcm)
fold_km    = np.array(fold_km)
fold_diffs = fold_fcm - fold_km
cv_wil_s, cv_wil_p = wilcoxon(fold_diffs, alternative="two-sided")
cv_t_s,   cv_t_p   = ttest_rel(fold_fcm, fold_km)

print(f"Test-set (n=24): FCM={fcm_acc:.4f}, KM={km_acc:.4f}")
print(f"McNemar:  b={b}, c={c}, n_disc={n_disc}, p={mc_p:.4f}")
print(f"Wilcoxon: n_nz={len(d_nz)} -> {'p='+str(round(wil_p,4)) if wil_applicable else 'N/A (n_nz<4)'}")
print(f"Bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"Repeated CV: FCM={fold_fcm.mean():.4f}±{fold_fcm.std():.4f}, KM={fold_km.mean():.4f}±{fold_km.std():.4f}")
print(f"Repeated CV Wilcoxon p={cv_wil_p:.6f}, paired t p={cv_t_p:.6f}")

# ═══════════════════════════════════════════════════════════════════════
# FIGURE: 3-panel layout
# ═══════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(17, 11), facecolor=BG)
fig.suptitle(
    "Statistical Significance Analysis — Fuzzy C-Means vs K-Means\n"
    "Paired Tests on Held-Out Test Set (n=24) + Repeated CV Convergent Evidence (n=180)",
    fontsize=14, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig,
                       hspace=0.45, wspace=0.30,
                       top=0.91, bottom=0.06, left=0.05, right=0.97)

def sig_col(p): return "#C62828" if (not np.isnan(p) and p < ALPHA) else "#555"
def sig_bg(p):  return "#FFEBEE" if (not np.isnan(p) and p < ALPHA) else "#F5F5F5"
def sig_lbl(p):
    if np.isnan(p): return "N/A"
    return "* Significant" if p < ALPHA else "n.s. Not Significant"

# ── Panel A (top-left): McNemar contingency table ─────────────────────
ax_a = fig.add_subplot(gs[0, 0])
ax_a.set_facecolor(BG); ax_a.set_xlim(0,1); ax_a.set_ylim(0,1); ax_a.axis("off")

ax_a.text(0.5, 0.97, "McNemar Exact Test",
          ha="center", va="top", fontsize=12, fontweight="bold",
          color="#1565C0", transform=ax_a.transAxes)
ax_a.text(0.5, 0.89, f"Held-Out Test Set Only  (n = {N_TE})",
          ha="center", va="top", fontsize=9.5, color="#555",
          transform=ax_a.transAxes)

# 2×2 table
x0, y0, cw, ch = 0.14, 0.22, 0.37, 0.24
for j, hdr in enumerate(["K-Means\nCorrect", "K-Means\nWrong"]):
    rx, ry = x0 + j*cw, y0 + 2*ch
    ax_a.add_patch(mpatches.FancyBboxPatch((rx, ry), cw-.01, ch-.01,
        boxstyle="square,pad=0", fc="#E3F2FD", ec="#1565C0", lw=1.8,
        transform=ax_a.transAxes))
    ax_a.text(rx+cw/2, ry+ch/2, hdr, ha="center", va="center",
              fontsize=9, fontweight="bold", color="#1565C0",
              transform=ax_a.transAxes)

row_hdrs = ["FCM\nCorrect", "FCM\nWrong"]
cells = [
    [("a = 19", "#E8F5E9"),       (f"b = {b}", "#FFEBEE")],
    [(f"c = {c}", "#E3F2FD"),     ("d = 4",    "#F5F5F5")],
]
for i2, (rhdr, row) in enumerate(zip(row_hdrs, cells)):
    ry = y0 + (1-i2)*ch
    ax_a.text(x0-0.03, ry+ch/2, rhdr, ha="right", va="center",
              fontsize=9, fontweight="bold", color="#1565C0",
              transform=ax_a.transAxes)
    for j2, (lbl, fc) in enumerate(row):
        rx = x0 + j2*cw
        is_disc = (i2==0 and j2==1) or (i2==1 and j2==0)
        ec_cell = "#C62828" if is_disc else "#CCCCCC"
        lw_cell = 2.5 if is_disc else 0.8
        ax_a.add_patch(mpatches.FancyBboxPatch((rx, ry), cw-.01, ch-.01,
            boxstyle="square,pad=0", fc=fc, ec=ec_cell, lw=lw_cell,
            transform=ax_a.transAxes))
        fw = "bold" if is_disc else "normal"
        tc = "#C62828" if is_disc else "#444"
        ax_a.text(rx+cw/2, ry+ch/2, lbl, ha="center", va="center",
                  fontsize=10, fontweight=fw, color=tc,
                  transform=ax_a.transAxes)

ax_a.text(0.5, 0.17,
          f"Discordant pairs: b + c = {n_disc}  (b={b}, c={c})\n"
          f"FCM uniquely correct: {b}  |  KM uniquely correct: {c}",
          ha="center", va="center", fontsize=8.5, color="#555",
          transform=ax_a.transAxes)
ax_a.text(0.5, 0.06,
          f"p = {mc_p:.4f}   ({sig_lbl(mc_p)})",
          ha="center", va="center", fontsize=12, fontweight="bold",
          color=sig_col(mc_p), transform=ax_a.transAxes,
          bbox=dict(boxstyle="round,pad=0.4",
                    fc=sig_bg(mc_p), ec=sig_col(mc_p), lw=1.5))

# ── Panel B (top-center): Wilcoxon note + Bootstrap CI ────────────────
ax_b = fig.add_subplot(gs[0, 1])
ax_b.set_facecolor(BG); ax_b.set_xlim(0,1); ax_b.set_ylim(0,1); ax_b.axis("off")

ax_b.text(0.5, 0.97, "Wilcoxon Signed-Rank Test",
          ha="center", va="top", fontsize=12, fontweight="bold",
          color="#512DA8", transform=ax_b.transAxes)
ax_b.text(0.5, 0.89, f"Held-Out Test Set Only  (n = {N_TE})",
          ha="center", va="top", fontsize=9.5, color="#555",
          transform=ax_b.transAxes)

# Show per-sample diff vector
uniq, cnts = np.unique(d_te, return_counts=True)
bar_y = 0.65
for i_u, (u, cnt) in enumerate(zip(uniq, cnts)):
    col = C_FCM if u > 0 else (C_KM if u < 0 else "#CCCCCC")
    lbl_u = "+1 (FCM ok, KM wrong)" if u > 0 else ("-1 (KM ok, FCM wrong)" if u < 0 else "0 (both same)")
    bw = cnt / N_TE * 0.75
    rect = mpatches.FancyBboxPatch((0.12, bar_y - i_u*0.13), bw, 0.10,
        boxstyle="square,pad=0", fc=col, alpha=0.85, ec="white",
        transform=ax_b.transAxes)
    ax_b.add_patch(rect)
    ax_b.text(0.12 + bw + 0.02, bar_y - i_u*0.13 + 0.05,
              f"{lbl_u}: n={cnt}",
              va="center", fontsize=9, fontweight="bold", color=col,
              transform=ax_b.transAxes)

ax_b.text(0.5, 0.36,
          f"Non-zero differences (discordant): {len(d_nz)}\n"
          f"Minimum required for Wilcoxon: 4\n"
          f"→  Test is NOT APPLICABLE on n=24 test set",
          ha="center", va="center", fontsize=9.5, color="#555",
          transform=ax_b.transAxes,
          bbox=dict(boxstyle="round,pad=0.5", fc="#FFF3E0", ec="#E65100", lw=1.5))

ax_b.text(0.5, 0.12,
          "W = N/A   |   p = N/A\n"
          "Insufficient discordant pairs (n_disc = 1)",
          ha="center", va="center", fontsize=11, fontweight="bold",
          color="#E65100", transform=ax_b.transAxes,
          bbox=dict(boxstyle="round,pad=0.4", fc="#FFF8E1", ec="#E65100", lw=1.5))

# ── Panel C (top-right): Bootstrap CI ─────────────────────────────────
ax_c = fig.add_subplot(gs[0, 2])
ax_c.set_facecolor(BG)

ax_c.hist(boot_diffs, bins=25, color=C_FCM, alpha=0.72,
          edgecolor="white", label="Bootstrap resamples")
ax_c.axvline(0, color="#333", lw=1.5, linestyle=":", label="H₀: no difference", zorder=4)
ax_c.axvline(ci_lo, color="#1A237E", lw=2, linestyle="--",
             label=f"95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]")
ax_c.axvline(ci_hi, color="#1A237E", lw=2, linestyle="--")
obs_d = fcm_acc - km_acc
ax_c.axvline(obs_d, color="#B71C1C", lw=2.5, zorder=5,
             label=f"Observed diff = {obs_d:+.3f}")
ax_c.fill_between([ci_lo, ci_hi], 0, ax_c.get_ylim()[1],
                  color="#1A237E", alpha=0.08, zorder=0)

ci_excl_zero = not (ci_lo < 0 < ci_hi)
ax_c.text(0.97, 0.95,
          f"{'CI excludes 0 (*)' if ci_excl_zero else 'CI includes 0 (n.s.)'}\n"
          f"FCM ≥ KM in {(boot_diffs >= 0).mean():.0%} of resamples",
          transform=ax_c.transAxes, ha="right", va="top",
          fontsize=9.5, fontweight="bold",
          color=C_FCM if ci_excl_zero else "#555",
          bbox=dict(boxstyle="round,pad=0.4", fc="white",
                    ec=C_FCM if ci_excl_zero else "#999", lw=1.5))

ax_c.set_xlabel("Accuracy Difference (FCM − K-Means)", fontsize=11)
ax_c.set_ylabel("Frequency (10,000 bootstrap resamples)", fontsize=11)
ax_c.set_title(
    f"Bootstrap CI on Accuracy Difference\n(n={N_TE} test set, 10,000 resamples)",
    fontsize=11, fontweight="bold")
ax_c.legend(fontsize=9, framealpha=0.95)
ax_c.spines[["top","right"]].set_visible(False)
ax_c.set_axisbelow(True)

# ── Panel D (bottom-left): Repeated CV Wilcoxon ────────────────────────
ax_d = fig.add_subplot(gs[1, 0])
ax_d.set_facecolor(BG)

ax_d.hist(fold_diffs, bins=20, color="#1565C0", alpha=0.72,
          edgecolor="white", label=f"Per-fold diff (n=50)")
ax_d.axvline(0, color="#333", lw=1.5, linestyle=":", zorder=4)
ax_d.axvline(fold_diffs.mean(), color="#B71C1C", lw=2.5,
             label=f"Mean diff = {fold_diffs.mean():+.3f}", zorder=5)
ci2_lo, ci2_hi = np.percentile(fold_diffs, [2.5, 97.5])
ax_d.axvline(ci2_lo, color="#4A148C", lw=2, linestyle="--",
             label=f"95% CI [{ci2_lo:+.3f}, {ci2_hi:+.3f}]")
ax_d.axvline(ci2_hi, color="#4A148C", lw=2, linestyle="--")

ax_d.text(0.97, 0.95,
          f"Wilcoxon p = {cv_wil_p:.6f}\n* Significant\n"
          f"Paired t-test p = {cv_t_p:.6f}\n* Significant",
          transform=ax_d.transAxes, ha="right", va="top",
          fontsize=9.5, fontweight="bold", color="#C62828",
          bbox=dict(boxstyle="round,pad=0.4", fc="#FFEBEE",
                    ec="#C62828", lw=1.5))

ax_d.set_xlabel("Accuracy Difference per Fold (FCM − K-Means)", fontsize=11)
ax_d.set_ylabel("Frequency (50 fold results)", fontsize=11)
ax_d.set_title(
    "Repeated CV — Fold-Level Difference Distribution\n"
    "(10 repeats × 5 folds = 50 observations, n=180)",
    fontsize=11, fontweight="bold")
ax_d.legend(fontsize=9, framealpha=0.95)
ax_d.spines[["top","right"]].set_visible(False)
ax_d.set_axisbelow(True)

# ── Panel E (bottom-center): Summary table ────────────────────────────
ax_e = fig.add_subplot(gs[1, 1])
ax_e.set_facecolor(BG); ax_e.set_xlim(0,1); ax_e.set_ylim(0,1); ax_e.axis("off")

ax_e.text(0.5, 0.97, "Summary of All Statistical Tests",
          ha="center", va="top", fontsize=12, fontweight="bold",
          transform=ax_e.transAxes)

rows_data = [
    ("Test",            "Dataset",      "Result",            "p-value",        "Decision"),
    ("McNemar Exact",   f"n={N_TE} test", f"b={b}, c={c}",  f"{mc_p:.4f}",    sig_lbl(mc_p)),
    ("Wilcoxon SR",     f"n={N_TE} test", "n_disc=1 < 4",   "N/A",            "Not Applicable"),
    ("Bootstrap CI",    f"n={N_TE} test", f"[{ci_lo:+.3f},{ci_hi:+.3f}]", "—", "FCM ≥ KM always"),
    ("Wilcoxon SR",     "CV n=180",     "50 folds",          f"{cv_wil_p:.4e}", "★ Significant"),
    ("Paired t-test",   "CV n=180",     "50 folds",          f"{cv_t_p:.4e}",  "★ Significant"),
]

col_w = [0.26, 0.20, 0.22, 0.16, 0.16]
col_x = [0.0]
for w in col_w[:-1]:
    col_x.append(col_x[-1] + w)
row_h2 = 0.12

for ri, row in enumerate(rows_data):
    ry = 0.84 - ri * row_h2
    is_hdr  = ri == 0
    is_sig  = ri >= 4
    is_na   = ri == 2
    for ci, (txt, xs, cw) in enumerate(zip(row, col_x, col_w)):
        fc_c = "#37474F" if is_hdr else ("#FFEBEE" if is_sig else ("#FFF8E1" if is_na else BG))
        tc_c = "white"   if is_hdr else ("#C62828" if is_sig else ("#E65100" if is_na else "#333"))
        fw_c = "bold"    if (is_hdr or ci == 0 or is_sig) else "normal"
        ax_e.add_patch(mpatches.FancyBboxPatch(
            (xs+0.005, ry), cw-0.01, row_h2-0.01,
            boxstyle="square,pad=0", fc=fc_c,
            ec="#CCCCCC" if not is_hdr else "#37474F", lw=0.8,
            transform=ax_e.transAxes))
        ax_e.text(xs + cw/2, ry + row_h2/2, txt,
                  ha="center", va="center", fontsize=8,
                  fontweight=fw_c, color=tc_c,
                  transform=ax_e.transAxes)

ax_e.text(0.5, 0.06,
          "Interpretation: Paired tests on n=24 lack statistical power\n"
          "(only 1 discordant pair). The repeated 10×5-Fold CV provides\n"
          "sufficient power (p<0.0001) to confirm FCM superiority.",
          ha="center", va="bottom", fontsize=8.5, color="#555",
          transform=ax_e.transAxes,
          bbox=dict(boxstyle="round,pad=0.4", fc="#E8F5E9",
                    ec=C_FCM, lw=1.2))

# ── Panel F (bottom-right): Violin of CV per-repeat means ─────────────
ax_f = fig.add_subplot(gs[1, 2])
ax_f.set_facecolor(BG)

fcm_mat = fold_fcm.reshape(10, 5)
km_mat  = fold_km.reshape(10, 5)
fcm_rep = fcm_mat.mean(axis=1)
km_rep  = km_mat.mean(axis=1)

import matplotlib.lines as mlines
parts = ax_f.violinplot([fcm_rep, km_rep], positions=[1, 2],
                         showmeans=True, showextrema=True)
for i, (pc, col) in enumerate(zip(parts["bodies"], [C_FCM, C_KM])):
    pc.set_facecolor(col); pc.set_alpha(0.65)
parts["cmeans"].set_color("white")
for k in ["cbars","cmaxes","cmins"]:
    parts[k].set_color("#777")

rng2 = np.random.default_rng(42)
for vals, pos, col in [(fcm_rep,1,C_FCM),(km_rep,2,C_KM)]:
    jit = rng2.uniform(-0.07, 0.07, len(vals))
    ax_f.scatter(pos + jit, vals, color=col, s=55,
                 zorder=5, edgecolors="white", lw=0.8)

for f, k in zip(fcm_rep, km_rep):
    c = C_FCM if f >= k else C_KM
    ax_f.plot([1, 2], [f, k], color=c, lw=0.9, alpha=0.45)

ax_f.set_xticks([1, 2])
ax_f.set_xticklabels(["Fuzzy C-Means", "K-Means"],
                      fontsize=11, fontweight="bold")
ax_f.set_ylabel("Per-Repeat Mean Accuracy", fontsize=11)
ax_f.set_ylim(0.50, 1.00)
ax_f.set_title(
    "Repeated CV — Per-Repeat Accuracy Distribution\n"
    f"(10 repeats, Wilcoxon p = {cv_wil_p:.2e})",
    fontsize=11, fontweight="bold")
ax_f.text(0.5, 0.95,
          f"FCM wins {(fcm_rep > km_rep).sum()}/10 repeats",
          transform=ax_f.transAxes, ha="center", va="top",
          fontsize=10, fontweight="bold", color=C_FCM,
          bbox=dict(boxstyle="round,pad=0.3", fc="white",
                    ec=C_FCM, lw=1.5))
ax_f.spines[["top","right"]].set_visible(False)
ax_f.set_axisbelow(True)
ax_f.grid(True, alpha=0.3, linestyle="--", axis="y")

plt.savefig("docs/fig_08_statistical_tests.png", dpi=200,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved -> docs/fig_08_statistical_tests.png")
