"""Report figures (E7, 10/24): Fig. 4 instrument vs LB calibration, Fig. 5 production-order stack, Fig. 7 X1 gain vs k.
Numbers from work/t2/g85_calibv3.csv (calibration pairs), the judge outputs recorded in DECISION_LEDGER #30-#32 and the
G93 final-pair frame (stack), work/t2/g99_x1_k.csv (k curve). Validated palette (dataviz skill): blue #2a78d6 = main
series, orange #eb6834 = the exception / the hold-out; text #0b0b0b / #52514e; recessive grid.
usage: python make_report_figs.py  -> Desktop/trafficflowbench/report/figs/fig{3,4,5}.{png,pdf}"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
_P = (lambda p: "/mnt/c" + p[2:]) if os.name == "posix" else (lambda p: p)   # rendered with WSL's matplotlib (not installed on Windows)
OUT = _P("C:/Users/user/OneDrive/Desktop/trafficflowbench/report/figs/"); W = _P("C:/Users/user/tfb/work/t2/")
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e6e3"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "savefig.dpi": 300, "figure.dpi": 150})


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT + name + ".png", facecolor="white"); fig.savefig(OUT + name + ".pdf", facecolor="white"); plt.close(fig)


# Fig. 4: instrument-predicted vs LB difference of S for submission pairs
C = pd.read_csv(W + "g85_calibv3.csv")
short = {"memory -> no memory": "no memory", "evidence rule": "evidence rule", "far gate": "far gate", "sim ongoing (s60->s70)": "synthetic ongoing",
         "B1 skip (s70->s72)": "B1 skip", "negative control (s49)": "negative control", "s75 TTA + cone": "TTA + cone", "s76 X1 ongoing": "X1 ongoing"}
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(3.4, 5.5), gridspec_kw=dict(height_ratios=[3.0, 2.3]))
x, y = C["main"].values * 1e3, C["lb"].values * 1e3
lim = (-6.5, 5.0); ax.plot(lim, lim, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=1); ax.axhline(0, color=INK2, lw=0.5); ax.axvline(0, color=INK2, lw=0.5)
big = np.abs(y) >= 0.5
ax.scatter(x[big], y[big], s=28, color=BLUE, edgecolor="white", linewidth=1.0, zorder=3, label="pairs, |ΔLB| ≥ 0.5 (sign agrees 6/6)")
ax.scatter(x[~big], y[~big], s=26, facecolor="white", edgecolor=BLUE, linewidth=1.3, zorder=3, label="pairs, |ΔLB| < 0.5 (not counted)")
OFF = {"no memory": (0, -11, "center"), "evidence rule": (0, 7, "center"), "far gate": (0, -11, "center"), "synthetic ongoing": (-5, 2, "right"),
       "B1 skip": (-5, -3, "right"), "negative control": (0, 9, "center"), "TTA + cone": (-38, 22, "right"), "X1 ongoing": (34, -20, "left")}
for xi, yi, n in zip(x, y, C["pair"]):
    dx, dy, ha = OFF[short[n]]; lead = short[n] in ("TTA + cone", "X1 ongoing")
    ax.annotate(short[n], (xi, yi), xytext=(dx, dy), textcoords="offset points", fontsize=6.5, color=INK2, ha=ha, va="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5, shrinkA=0, shrinkB=3) if lead else None)
ax.scatter([0.8], [-2.7], s=34, facecolor="white", edgecolor=ORANGE, linewidth=1.6, zorder=4, label="E3′ (missed: double use with X1)")
ax.annotate("E3′", (0.8, -2.7), xytext=(5, -2), textcoords="offset points", fontsize=6.5, color=INK2)
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel("ΔS predicted by the March instrument (×10⁻³)"); ax.set_ylabel("ΔS on the public leaderboard (×10⁻³)")
ax.legend(loc="lower right", fontsize=6, frameon=False); ax.set_title("(a) Task 2 and mixed pairs: the March instrument", loc="left", fontsize=7, color=INK)
# (b) Task-1-only pairs (ledger #68): the harness total 0.35 dS_state + 0.15 dS_phys against the LB, work/b22_fdsweep.log, x 10^-3
T1P = [("gapml", 10.39, 5.20, (-7, -7, "right")), ("FD 0.75 vs 0", 1.43, 1.00, (-3, -11, "left")), ("FD 0.5 vs 0.75", -0.22, 0.06, (8, 9, "right")), ("FD 1.0 vs 0.75", -0.22, -0.21, (26, -2, "left"))]
xs = np.linspace(-1.5, 14, 800); ax2.set_xscale("asinh", linear_width=0.5); ax2.set_yscale("asinh", linear_width=0.5)
ax2.fill_between(xs, np.minimum(xs, 0.5 * xs), np.maximum(xs, 0.5 * xs), color=BLUE, alpha=0.12, lw=0, label="LB / harness between 0.5 and 1.0"); ax2.plot(xs, xs, color=INK2, lw=0.8, ls=(0, (4, 3)))
ax2.axhline(0, color=INK2, lw=0.5); ax2.axvline(0, color=INK2, lw=0.5)
for n, xi, yi, (dx, dy, ha) in T1P:
    agree = xi * yi > 0; ax2.scatter([xi], [yi], s=26, zorder=3, facecolor=BLUE if agree else "white", edgecolor=BLUE if agree else ORANGE, linewidth=1.3)
    ax2.annotate(n, (xi, yi), xytext=(dx, dy), textcoords="offset points", fontsize=6.5, color=INK2, ha=ha, va="center")
ax2.set_xlim(-1.5, 14); ax2.set_ylim(-1.5, 14); ax2.set_xticks([-1, 0, 1, 10]); ax2.set_yticks([-1, 0, 1, 10]); ax2.set_xticklabels(["−1", "0", "1", "10"]); ax2.set_yticklabels(["−1", "0", "1", "10"]); ax2.minorticks_off()
ax2.set_xlabel("0.35 ΔS_state + 0.15 ΔS_phys on February (×10⁻³)"); ax2.set_ylabel("ΔS on the public leaderboard (×10⁻³)"); ax2.legend(loc="lower right", fontsize=6, frameon=False)
ax2.set_title("(b) Task-1-only pairs: the harness total", loc="left", fontsize=7, color=INK); save(fig, "fig4_calibration")

# Fig. 5: A's April components measured in production order on B's base (M3 v3), 95% intervals
rows = [("+ no-memory LightGBM\n(ongoing)", 1.0, 0.3, BLUE), ("+ cone on top\n(ongoing)", 0.2, 0.2, BLUE),   # one-condition steps: half the in-condition change
        ("+ onset ranker\n(onset)", 2.8, 1.55, BLUE), ("all of A's April\ncomponents", 3.8, 1.6, BLUE),
        ("same, hold-out\n(read once)", 3.2, 2.1, ORANGE)]
fig, ax = plt.subplots(figsize=(3.4, 2.4)); yy = np.arange(len(rows))[::-1]
for yi, (lab, m, se, col) in zip(yy, rows):
    ax.plot([m - 1.96 * se, m + 1.96 * se], [yi, yi], color=col, lw=2, solid_capstyle="round")
    ax.scatter([m], [yi], s=30, color=col, edgecolor="white", linewidth=1.0, zorder=3)
ax.axvline(0, color=INK2, lw=0.6); ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows], fontsize=6.5); ax.grid(axis="y", visible=False)
ax.set_xlabel("ΔT2 on M3 (×10⁻³), 95% interval"); save(fig, "fig5_stack")

# Fig. 7: X1 gain on ongoing windows vs the number of completed earlier days used
K = pd.read_csv(W + "g99_x1_k.csv")
fig, ax = plt.subplots(figsize=(3.4, 2.2))
ax.axhline(0, color=INK2, lw=0.6)
ax.errorbar(K.k, K["mean"] * 1e3, yerr=K.se * 1e3, color=BLUE, lw=2, marker="o", ms=5, mec="white", mew=1.0, capsize=0, elinewidth=1.0)
ax.set_xticks(K.k); ax.set_xlabel("earlier days of the month used for fine-tuning (k)"); ax.set_ylabel("ΔIoU, ongoing (×10⁻³)")
ax.set_ylim(-2, 15); save(fig, "fig7_x1_k")
# Fig. 1: the congestion map moves (per-link share of queued observed rows, training months vs March), two corridors
M = pd.read_csv(W + "g100_map_shift.csv")
fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.0), sharex=False)
for ax, (p, tv) in zip(axes, (("D12_I5_S", 0.364), ("D7_I10_E", 0.274))):
    g = M[M.panel == p]
    ax.plot(g.link, g.train * 100, color=BLUE, lw=1.4, label="training months")
    ax.plot(g.link, g.march * 100, color=ORANGE, lw=1.4, label="March")
    ax.set_title(f"{p}  (map distance TV = {tv:.2f})", fontsize=7, color=INK, loc="left")
    ax.set_ylabel("queued rows (%)"); ax.set_ylim(bottom=0)
axes[-1].set_xlabel("link (upstream → downstream)"); axes[0].legend(loc="upper left", fontsize=6, frameon=False)
save(fig, "fig1_map_shift")

# Fig. 2: where the onset loss is (M3, weighted share of onset windows vs share of onset loss, by error class)
L = pd.read_csv(W + "g52_m3_loss_map.csv"); O = L[L.cond == "queue_onset"]
CL = [("exact", "exact"), ("extent (overlap)", "right place,\nwrong extent"), ("missed (pred empty)", "no queue\npredicted"), ("wrong place (no overlap)", "wrong place\n(no overlap)")]
ws = [O.wt[O.cls == c].sum() / O.wt.sum() * 100 for c, _ in CL]; ls = [O.loss[O.cls == c].sum() / O.loss.sum() * 100 for c, _ in CL]
fig, ax = plt.subplots(figsize=(3.4, 2.3)); yy = np.arange(len(CL))[::-1]; h = 0.34; ax.set_axisbelow(True)
ax.barh(yy + 0.19, ws, height=h, color=BLUE, label="share of onset windows")
ax.barh(yy - 0.19, ls, height=h, color=ORANGE, label="share of onset loss")
for yi, a, b in zip(yy, ws, ls):
    ax.text(a + 1, yi + 0.19, f"{a:.0f}%", va="center", fontsize=6.5, color=INK2); ax.text(b + 1, yi - 0.19, f"{b:.0f}%", va="center", fontsize=6.5, color=INK2)
ax.set_yticks(yy); ax.set_yticklabels([c[1] for c in CL], fontsize=6.5); ax.set_xlim(0, 100); ax.grid(axis="y", visible=False)
ax.set_xlabel("% (M3, onset windows, weighted)"); ax.legend(loc="center right", fontsize=6, frameon=False); save(fig, "fig3_loss_map")

# Fig. 2: one breakdown on one link (D7_I10_E, the most often queued link; the median-drop morning switch), two panels
R = pd.read_csv(W + "g101_regime_switch.csv")
fig, axes = plt.subplots(2, 1, figsize=(3.4, 2.6), sharex=True)
for ax, col, lab in ((axes[0], "v_over_vf", "speed / free-flow speed"), (axes[1], "q_over_cap", "flow / capacity")):
    ax.axvspan(0, R.minute.max() + 2.5, color="#eeeeeb", lw=0, zorder=0)
    ax.plot(R.minute, R[col], color=BLUE, lw=1.6, marker="o", ms=3, mec="white", mew=0.6, zorder=3)
    ax.set_ylabel(lab, fontsize=7)
axes[0].set_ylim(0, 1.15); axes[1].set_ylim(0.4, 0.9)
axes[0].annotate("one 5-min row: 0.99 → 0.35\n(median drop 0.64 over 22 mornings)", (0, 0.35), xytext=(4, 0.62), fontsize=6, color=INK2,
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
axes[1].set_xlabel("minutes from the first queued row (shaded: queued)")
save(fig, "fig2_regime_switch")
# Fig. 8: what a Task 2 window reads (schematic) and the organisers' check reproduced
from matplotlib.patches import FancyBboxPatch
fig = plt.figure(figsize=(7.0, 2.0)); ax = fig.add_axes([0.01, 0.02, 0.98, 0.96]); ax.set_xlim(0, 20); ax.set_ylim(0, 6); ax.axis("off")
def band(x0, x1, y, h, col, txt, tcol="white"):
    ax.add_patch(FancyBboxPatch((x0, y), x1 - x0, h, boxstyle="round,pad=0,rounding_size=0.15", fc=col, ec="none"))
    ax.text((x0 + x1) / 2, y + h / 2, txt, ha="center", va="center", fontsize=7, color=tcol)
band(0.2, 9.8, 4.6, 0.9, BLUE, "completed earlier days of the month")
band(10.0, 14.8, 4.6, 0.9, BLUE, "day D, up to the origin T")
band(15.0, 19.8, 4.6, 0.9, "#d9d8d4", "after T", tcol=INK)
for x, txt in ((0.3, "X1: fine-tuning windows of days 1 … D−1\nevidence rule: same slot on ≤ 3 earlier days\nonset ranker weight a: onset events so far"),
               (10.1, "network inputs: the visible hour\nsame-day evidence (05:00 … T)\ncone, far gate: rows < T"),
               (15.1, "deleted in the check;\nnever read by the pipeline")):
    ax.text(x, 4.35, txt, ha="left", va="top", fontsize=6.3, color=INK2, linespacing=1.35)
ax.add_patch(FancyBboxPatch((0.2, 0.2), 19.6, 1.3, boxstyle="round,pad=0,rounding_size=0.15", fc="#f3f2ef", ec="none"))
ax.text(10.0, 0.85, "Check: for each of the 160 windows, a copy of the release cut at T (every corridor of the month) is fed to the one-window entry.\n"
        "160 / 160 windows identical; X1 training windows identical on 22 / 22 days;\nX1 fine-tuned inside the cut copy gives the submitted predictions.",
        ha="center", va="center", fontsize=6.5, color=INK, linespacing=1.4)
fig.savefig(OUT + "fig8_causality.png", facecolor="white"); fig.savefig(OUT + "fig8_causality.pdf", facecolor="white"); plt.close(fig)   # no tight_layout: full-figure axes
print("figures written to", OUT)

# Fig. 6: the final pipeline (schematic). Grey = same in both candidates; blue = both on March, A on April; orange = A's April windows only
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(7.0, 2.4)); ax.set_xlim(0, 100); ax.set_ylim(-3, 32); ax.axis("off"); GREY = "#8a8985"
def chain(y, label, items, x0=14.0, gap=1.5, h=5.2):
    ax.text(0, y, label, ha="left", va="center", fontsize=7, color=INK, fontweight="bold"); x = x0
    for i, (w, t, col) in enumerate(items):
        ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.5", fc="white", ec=col, lw=1.0))
        ax.text(x + w / 2, y, t, ha="center", va="center", fontsize=6, color=INK, linespacing=1.05)
        if i:
            ax.annotate("", (x, y), (x - gap, y), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.6, shrinkA=0, shrinkB=0, mutation_scale=6))
        x += w + gap
NETS = "Networks (E4 on\nA's April windows)"
chain(28, "Tasks 1, 3", [(15, "Three LightGBM pairs\nwith physics features", GREY), (14, "Link statics from\nthe month (ADAPT)", GREY), (13, "Model for cells\nnext to the blanks", GREY), (17, "Speed ceiling, FD blend,\nboundary model", GREY), (15, "Task 3 computed from\nTask 1's densities", GREY)])
chain(20.5, "Task 4", [(14, "Prior b, link counts c", GREY), (18, "L2 projection onto\n{f : A f = c, f ≥ 0}", GREY)])
chain(13, "Task 2 onset", [(14, NETS, BLUE), (12, "Ranker with\nadaptive weight", ORANGE), (9, "Evidence\nrule", BLUE), (11, "Expected-IoU\nset", BLUE)])
chain(5.5, "Task 2 ongoing", [(14, NETS, BLUE), (6, "X1", BLUE), (9, "Attention\nnetwork 0.2", BLUE), (10, "No-memory\nLightGBM 0.2", ORANGE), (6, "Cone", BLUE), (9, "Evidence\nrule", BLUE), (11, "Expected-IoU\nset", BLUE), (7, "Far gate", BLUE)])
for x, t, col in ((14, "same in both candidates", GREY), (37, "both on March, A on April", BLUE), (61, "A's April windows only", ORANGE)):
    ax.plot([x, x + 3], [-1.5, -1.5], color=col, lw=1.6, solid_capstyle="round"); ax.text(x + 4, -1.5, t, ha="left", va="center", fontsize=6, color=INK2)
save(fig, "fig6_pipeline")
