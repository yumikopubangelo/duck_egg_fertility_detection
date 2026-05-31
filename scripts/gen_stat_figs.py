"""Generate correct/incorrect bar + statistical test figures."""
import json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.stats import wilcoxon
from scipy.stats import binom
import warnings
warnings.filterwarnings("ignore")

BG    = "#FAFAFA"
C_FCM = "#2A9D8F"
C_KM  = "#F4A261"
C_AWC = "#2166AC"
ALPHA = 0.05

# ── Load n=24 test predictions ─────────────────────────────────────
with open("results/evaluation/full_vascular_glcm_mobilenet/"
          "full_evaluation_20260427_004032.json") as f:
    ev24 = json.load(f)

N_TEST = ev24["meta"]["n_test"]
yt24   = np.array(ev24["results"]["FCM"]["y_true"])
yp_fcm24 = np.array(ev24["results"]["FCM"]["y_pred"])
yp_km24  = np.array(ev24["results"]["KMeans"]["y_pred"])

# ── Load n=68 test+val predictions ────────────────────────────────
df = pd.read_csv("results/evaluation/full_evaluation_predictions.csv")
N_STAT = len(df)

def lbl2int(col):
    return (df[col] == "fertile").astype(int).values

y68      = lbl2int("true_label")
yp68_fcm = lbl2int("FCM")
yp68_km  = lbl2int("KMeans")
yp68_awc = lbl2int("AWC")

print(f"n_test={N_TEST}  n_stat={N_STAT}")

# ── Helpers ────────────────────────────────────────────────────────
def mcnemar_exact(pa, pb, yt_):
    ca = (pa == yt_); cb = (pb == yt_)
    b  = int(( ca & ~cb).sum())
    c  = int((~ca &  cb).sum())
    n  = b + c
    if n == 0:
        return 1.0, b, c, 0
    k  = min(b, c)
    p  = float(2 * sum(binom.pmf(i, n, 0.5) for i in range(k + 1)))
    return min(p, 1.0), b, c, n

def wilcoxon_diff(pa, pb, yt_):
    d = (pa == yt_).astype(float) - (pb == yt_).astype(float)
    d_nz = d[d != 0]
    if len(d_nz) < 4:
        return float("nan"), float("nan"), int(len(d_nz))
    stat, p = wilcoxon(d_nz, alternative="two-sided")
    return float(stat), float(p), int(len(d_nz))

# Pairwise comparisons on n=68
pairs_68 = [
    ("FCM",  "K-Means", yp68_fcm, yp68_km,  C_FCM),
    ("AWC",  "K-Means", yp68_awc, yp68_km,  C_AWC),
    ("AWC",  "FCM",     yp68_awc, yp68_fcm, C_KM),
]

stats_68 = []
for na, nb, pa, pb, pc in pairs_68:
    mc_p, b, c, nd = mcnemar_exact(pa, pb, y68)
    wil_s, wil_p, nz = wilcoxon_diff(pa, pb, y68)
    acc_a = (pa == y68).mean()
    acc_b = (pb == y68).mean()
    stats_68.append(dict(
        pair=f"{na} vs {nb}", na=na, nb=nb, pc=pc,
        acc_a=acc_a, acc_b=acc_b, diff=acc_a - acc_b,
        mc_p=mc_p, mc_b=b, mc_c=c, mc_nd=nd,
        mc_sig=mc_p < ALPHA,
        wil_s=wil_s, wil_p=wil_p, wil_nz=nz,
        wil_sig=(not np.isnan(wil_p)) and (wil_p < ALPHA),
    ))
    print(f"{na} vs {nb}: acc={acc_a:.3f} vs {acc_b:.3f} | "
          f"McNemar p={mc_p:.4f} | Wilcoxon p={wil_p:.4f}")

# ══════════════════════════════════════════════════════════════════
# FIG 1: Correct vs Incorrect (n=24, test split)
# ══════════════════════════════════════════════════════════════════
n_cor_km  = int((yp_km24  == yt24).sum())
n_cor_fcm = int((yp_fcm24 == yt24).sum())
n_wrg_km  = N_TEST - n_cor_km
n_wrg_fcm = N_TEST - n_cor_fcm

fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=BG)
ax.set_facecolor(BG)

models   = ["K-Means", "FCM"]
colors   = [C_KM, C_FCM]
corrects = [n_cor_km,  n_cor_fcm]
wrongs   = [n_wrg_km,  n_wrg_fcm]
x = np.arange(2); w = 0.50

b1 = ax.bar(x, corrects, w, color=colors, alpha=0.88,
            edgecolor="white", linewidth=1.2)
b2 = ax.bar(x, wrongs, w, bottom=corrects,
            color=colors, alpha=0.30,
            edgecolor=colors, linewidth=1.2, hatch="///")

for bar, v in zip(b1, corrects):
    ax.text(bar.get_x() + bar.get_width() / 2, v / 2,
            str(v), ha="center", va="center",
            fontsize=20, fontweight="bold", color="white")

for bar, v, bot, col in zip(b2, wrongs, corrects, colors):
    if v > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, bot + v / 2,
                str(v), ha="center", va="center",
                fontsize=14, fontweight="bold", color=col)

for i, (cor, col) in enumerate(zip(corrects, colors)):
    total_h = corrects[i] + wrongs[i]
    ax.text(i, total_h + 0.6, f"{cor/N_TEST:.1%}",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=col)

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=13, fontweight="bold")
ax.set_ylabel(f"Number of Predictions (n={N_TEST})", fontsize=11)
ax.set_ylim(0, N_TEST + 5)
ax.set_title(
    f"Correct vs Incorrect Predictions per Model\n"
    f"(Test Split: n={N_TEST} | fertile={N_TEST//2}, infertile={N_TEST//2})",
    fontsize=12, fontweight="bold")
ax.yaxis.grid(True, alpha=0.35, linestyle="--")
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(handles=[
    mpatches.Patch(color="#888", alpha=0.85, label="Correct"),
    mpatches.Patch(color="#888", alpha=0.35, hatch="///", label="Incorrect"),
], fontsize=10, framealpha=0.9, loc="upper right")

plt.tight_layout()
plt.savefig("docs/fig_06_prediction_correct.png", dpi=200,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved -> fig_06_prediction_correct.png")

# ══════════════════════════════════════════════════════════════════
# FIG 2: Statistical Significance Tests (n=68 test+val)
# ══════════════════════════════════════════════════════════════════
n_pairs = len(stats_68)
fig = plt.figure(figsize=(16, 11), facecolor=BG)
fig.suptitle(
    f"Statistical Significance Tests — Pairwise Model Comparison\n"
    f"(n={N_STAT} samples: test+val combined | alpha={ALPHA})",
    fontsize=14, fontweight="bold")

gs = gridspec.GridSpec(2, n_pairs, figure=fig,
                       hspace=0.50, wspace=0.30,
                       top=0.90, bottom=0.05)

for col_i, st in enumerate(stats_68):
    pc        = st["pc"]
    sig_lbl   = "* Significant" if st["mc_sig"] else "n.s. Not Significant"
    sig_col   = "#C62828"       if st["mc_sig"] else "#555555"
    sig_bg    = "#FFEBEE"       if st["mc_sig"] else "#F5F5F5"

    # ── McNemar panel ────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, col_i])
    ax0.set_facecolor(BG)
    ax0.set_xlim(0, 1); ax0.set_ylim(0, 1)
    ax0.axis("off")

    # Title
    ax0.text(0.5, 0.97, "McNemar Exact Test",
             ha="center", va="top", fontsize=11, fontweight="bold", color=pc,
             transform=ax0.transAxes)
    ax0.text(0.5, 0.88, st["pair"],
             ha="center", va="top", fontsize=9.5, color="#333",
             transform=ax0.transAxes)

    # 2x2 contingency table (in axis coords 0-1)
    x0, y0, cw, ch = 0.10, 0.20, 0.38, 0.26
    col_hdr = [f"{st['nb']}\nCorrect", f"{st['nb']}\nWrong"]
    row_hdr = [f"{st['na']}\nCorrect", f"{st['na']}\nWrong"]
    cells   = [
        [("a", "#E8F5E9"),       (f"b={st['mc_b']}", "#FFEBEE")],
        [(f"c={st['mc_c']}", "#E3F2FD"), ("d", "#F5F5F5")],
    ]

    for j, hdr in enumerate(col_hdr):
        rx = x0 + j * cw
        ry = y0 + 2 * ch
        rect = mpatches.FancyBboxPatch((rx, ry), cw - 0.01, ch - 0.01,
                                       boxstyle="square,pad=0",
                                       fc=pc + "20", ec=pc, lw=1.5,
                                       transform=ax0.transAxes)
        ax0.add_patch(rect)
        ax0.text(rx + cw / 2, ry + ch / 2, hdr,
                 ha="center", va="center", fontsize=8.5,
                 fontweight="bold", color=pc, transform=ax0.transAxes)

    for i2, (rhdr, row) in enumerate(zip(row_hdr, cells)):
        ry = y0 + (1 - i2) * ch
        ax0.text(x0 - 0.03, ry + ch / 2, rhdr,
                 ha="right", va="center", fontsize=8.5,
                 fontweight="bold", color=pc, transform=ax0.transAxes)
        for j2, (lbl, fc) in enumerate(row):
            rx = x0 + j2 * cw
            is_disc = (i2 == 0 and j2 == 1) or (i2 == 1 and j2 == 0)
            rect = mpatches.FancyBboxPatch((rx, ry), cw - 0.01, ch - 0.01,
                                           boxstyle="square,pad=0",
                                           fc=fc,
                                           ec=pc if is_disc else "#CCCCCC",
                                           lw=2.0 if is_disc else 0.8,
                                           transform=ax0.transAxes)
            ax0.add_patch(rect)
            fw = "bold" if is_disc else "normal"
            tc = pc if is_disc else "#444"
            ax0.text(rx + cw / 2, ry + ch / 2, lbl,
                     ha="center", va="center", fontsize=9,
                     fontweight=fw, color=tc, transform=ax0.transAxes)

    # Summary text
    ax0.text(0.5, 0.14,
             f"Discordant pairs (b+c) = {st['mc_nd']}",
             ha="center", va="center", fontsize=9,
             color="#555", transform=ax0.transAxes)
    ax0.text(0.5, 0.05,
             f"p = {st['mc_p']:.4f}  ({sig_lbl})",
             ha="center", va="center", fontsize=10, fontweight="bold",
             color=sig_col, transform=ax0.transAxes,
             bbox=dict(boxstyle="round,pad=0.3",
                       fc=sig_bg, ec=sig_col, lw=1.2))

    # ── Wilcoxon panel ───────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, col_i])
    ax1.set_facecolor(BG)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.axis("off")

    ax1.text(0.5, 0.97, "Wilcoxon Signed-Rank Test",
             ha="center", va="top", fontsize=11, fontweight="bold", color=pc,
             transform=ax1.transAxes)
    ax1.text(0.5, 0.88, st["pair"],
             ha="center", va="top", fontsize=9.5, color="#333",
             transform=ax1.transAxes)

    # Accuracy bars
    bar_data = [(st["na"], st["acc_a"], pc),
                (st["nb"], st["acc_b"], "#AAAAAA")]
    for i_b, (nm, acc, col_b) in enumerate(bar_data):
        bw = acc * 0.75
        by = 0.62 - i_b * 0.18
        rect = mpatches.FancyBboxPatch((0.10, by), bw, 0.13,
                                       boxstyle="square,pad=0",
                                       fc=col_b, alpha=0.85, ec="white",
                                       transform=ax1.transAxes)
        ax1.add_patch(rect)
        ax1.text(0.10 + bw + 0.02, by + 0.065,
                 f"{nm}: {acc:.3f}",
                 va="center", fontsize=9.5,
                 fontweight="bold", color=col_b,
                 transform=ax1.transAxes)

    diff_col = "#2E7D32" if st["diff"] > 0 else "#B71C1C"
    ax1.text(0.5, 0.40,
             f"Accuracy difference: {st['diff']:+.3f} ({st['diff']*100:+.1f}%)",
             ha="center", va="center", fontsize=9.5, color=diff_col,
             fontweight="bold", transform=ax1.transAxes)

    if np.isnan(st["wil_p"]):
        wil_txt = (f"W = n/a  |  p = n/a\n"
                   f"Insufficient non-tied differences\n"
                   f"(n_discordant = {st['wil_nz']})")
        wil_col = "#777"
        wil_bg  = "#F0F0F0"
    else:
        wsig = "* Significant" if st["wil_sig"] else "n.s. Not Significant"
        wil_txt = f"W = {st['wil_s']:.1f}  |  p = {st['wil_p']:.4f}\n{wsig}"
        wil_col = "#C62828" if st["wil_sig"] else "#555"
        wil_bg  = "#EEEEEE"

    ax1.text(0.5, 0.14, wil_txt,
             ha="center", va="center", fontsize=9.5,
             fontweight="bold", color=wil_col,
             transform=ax1.transAxes,
             bbox=dict(boxstyle="round,pad=0.4",
                       fc=wil_bg, ec=wil_col, lw=1.2))

plt.savefig("docs/fig_08_statistical_tests.png", dpi=200,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved -> fig_08_statistical_tests.png")
