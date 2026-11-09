"""Report figures added on 11/01 (numbering of report/R2): Fig. 2 pipeline, Fig. 5 sign flip across months, Fig. 6 where the
Task 2 loss sits. Figs. 1, 3 and 4 are copies of fig1_map_shift, fig4_calibration and fig5_stack (make_report_figs.py).
Fig. 5 numbers: far gate from work/t2/g119_month_postproc.csv (ledger #72), B 0.25 from audit_20261031/t201_evidence_grid.csv,
mgL March from work/t2/g147_mg.csv, the rest from the ledger rows named below; SE by clusters of corridor-day, as in judge.py
where the ledger gives none. Written to report/fig5_signflip.csv. Fig. 6: ledger #70, #75, #81 (M3 dev, final A flow).
usage (WSL matplotlib): python make_report_figs2.py [A7|A6]"""
import os, shutil, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

_P = (lambda p: "/mnt/c" + p[2:]) if os.name == "posix" else (lambda p: p)
ROOT = _P("C:/Users/user/OneDrive/Desktop/trafficflowbench/"); REP = ROOT + "report/"; OLD = REP + "figs/"; OUT = REP + "figs2/"; W = _P("C:/Users/user/tfb/work/t2/")
A = sys.argv[1] if len(sys.argv) > 1 else "A7"
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, INK, INK2, GRID, GREY = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e6e3", "#8a8985"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "savefig.dpi": 300, "figure.dpi": 150})


def save(fig, name):
    fig.savefig(OUT + name + ".png", facecolor="white"); plt.close(fig)


def cluster_se(x, d):
    c = x.assign(d=d).groupby(["panel", "day"]).d.sum()
    return float(np.sqrt(((c - c.mean()) ** 2).sum() * len(c) / (len(c) - 1)) / len(x))


for new, old in (("fig1_map_shift", "fig1_map_shift"), ("fig3_calibration", "fig4_calibration"), ("fig4_stack", "fig5_stack")):
    shutil.copyfile(OLD + old + ".png", OUT + new + ".png")

# Fig. 5 data
G = pd.read_csv(W + "g119_month_postproc.csv"); E = pd.read_csv(ROOT + "audit_20261031/t201_evidence_grid.csv"); M = pd.read_csv(W + "g147_mg.csv")
rows = []
for m, lab in (("2031-01", "January"), ("2031-02", "February")):
    x = G[(G.month == m) & (G.vis == "mask") & (G.cond == "queue_ongoing")]
    rows.append(("far gate", lab, (x.SG - x.S).mean(), cluster_se(x, x.SG - x.S), len(x), "g119, ledger #72"))
    x = E[(E.month == m) & (E.cond == "queue_onset")]
    rows.append(("evidence 0.25", lab, (x["b0.25"] - x["b1.0"]).mean(), cluster_se(x, x["b0.25"] - x["b1.0"]), len(x), "t201_evidence_grid"))
rows += [("X1", "January", -0.0113, 0.0039, 165, "ledger #95"), ("X1", "February", -0.0177, 0.0043, 151, "ledger #95"), ("X1", "March", 0.0118, 0.0033, 356, "ledger #96"),
         ("evidence 0.25", "March", 0.0223, 0.0114, 100, "t201_evidence_grid"),
         ("mgL", "January", -0.0077, 0.0077 / 1.6, 165, "ledger #108, SE back from the ledger's z"), ("mgL", "February", -0.0039, 0.0039 / 1.2, 151, "ledger #108, SE back from the ledger's z")]
x = G[(G.month == "2031-03") & (G.cond == "queue_ongoing")]; rows.append(("far gate", "March", (x.SG - x.S).mean(), cluster_se(x, x.SG - x.S), len(x), "g119, ledger #72"))
x = M[M.cond == "queue_ongoing"]; rows.append(("mgL", "March", (x.MG - x.B0).mean(), cluster_se(x, x.MG - x.B0), len(x), "g147, raw"))
F = pd.DataFrame(rows, columns=["component", "month", "delta", "se", "windows", "source"]); F.to_csv(REP + "fig5_signflip.csv", index=False)
print(F.round(4).to_string())

fig, ax = plt.subplots(figsize=(3.4, 2.5), layout="constrained")
COMP = [("X1", "X1\n(ongoing)"), ("far gate", "far gate\n(ongoing)"), ("mgL", "four network\nchanges (ongoing)"), ("evidence 0.25", "onset evidence\n× 0.25 (onset)")]
ST = {"January": (BLUE, "o", -0.22), "February": (BLUE, "s", 0.0), "March": (ORANGE, "D", 0.22)}
ax.axvline(0, color=INK2, lw=0.6, zorder=1); ax.grid(axis="y", visible=False)
for i, (c, _) in enumerate(COMP):
    y0 = len(COMP) - 1 - i
    for mn, (col, mk, dy) in ST.items():
        r = F[(F.component == c) & (F.month == mn)].iloc[0]
        ax.plot([(r.delta - r.se) * 1e3, (r.delta + r.se) * 1e3], [y0 + dy] * 2, color=col, lw=1.2, solid_capstyle="round", zorder=2)
        ax.scatter([r.delta * 1e3], [y0 + dy], s=20, marker=mk, color=col, edgecolor="white", linewidth=0.7, zorder=3, label=mn if i == 0 else None)
ax.set_yticks(range(len(COMP))); ax.set_yticklabels([c[1] for c in COMP[::-1]], fontsize=6.5)
ax.set_xlabel("change in IoU (×10⁻³), ± 1 SE"); ax.legend(loc="upper right", fontsize=6, frameon=False, handletextpad=0.2)
save(fig, "fig5_signflip")

# Fig. 6: loss of final A on M3 dev, by source
BARS = [("onset\n(100 windows)", [(0.071, GREY, "no precursor\n0.071"), (0.059, BLUE, "wrong place\n0.059"), (0.006, ORANGE, "")]),
        ("ongoing\n(356 windows)", [(0.030, GREY, "jam end\n0.030"), (0.026, BLUE, "place\n0.026"), (0.025, ORANGE, "extent\n0.025")])]
fig, ax = plt.subplots(figsize=(3.4, 1.55), layout="constrained")
for i, (lab, seg) in enumerate(BARS):
    y = 1 - i; x = 0.0
    for v, col, t in seg:
        ax.barh(y, v - 0.0006, left=x + 0.0003, height=0.6, color=col, zorder=2)
        if t:
            ax.text(x + v / 2, y, t, ha="center", va="center", fontsize=5.6, color="white", linespacing=1.05, zorder=3)
        x += v
    ax.text(x + 0.002, y, f"{x:.3f}", va="center", fontsize=6.5, color=INK2)
ax.text(0.1365, 1.40, "extent 0.006", ha="right", va="center", fontsize=5.6, color=INK2)
ax.set_yticks([1, 0]); ax.set_yticklabels([b[0] for b in BARS], fontsize=6.5); ax.set_xlim(0, 0.155); ax.set_ylim(-0.45, 1.55); ax.grid(axis="y", visible=False)
ax.set_xlabel("1 − IoU on M3 dev")
save(fig, "fig6_loss")

# Fig. 2: the final pipeline. Grey = both candidates on every window, blue = both on March and A on April, orange = A's April only
fig, ax = plt.subplots(figsize=(7.0, 2.4), layout="constrained"); ax.set_xlim(0, 100); ax.set_ylim(-3, 32); ax.axis("off")
def chain(y, label, items, x0=14.0, gap=1.5, h=5.2):
    ax.text(0, y, label, ha="left", va="center", fontsize=7, color=INK, fontweight="bold"); x = x0
    for i, (w, t, col) in enumerate(items):
        ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.5", fc="white", ec=col, lw=1.0))
        ax.text(x + w / 2, y, t, ha="center", va="center", fontsize=6, color=INK, linespacing=1.05)
        if i:
            ax.annotate("", (x, y), (x - gap, y), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.6, shrinkA=0, shrinkB=0, mutation_scale=6))
        x += w + gap
NETS = "No-memory\nnetworks" if A == "A7" else "Networks (E4 on\nA's April windows)"
chain(28, "Tasks 1, 3", [(15, "Three LightGBM pairs\nwith physics features", GREY), (14, "Link statics from\nthe month (ADAPT)", GREY), (13, "Model for cells\ninside the blanks", GREY), (17, "Speed ceiling, FD blend,\nboundary model", GREY), (15, "Task 3 computed from\nTask 1's densities", GREY)])
chain(20.5, "Task 4", [(14, "Prior b, link counts c", GREY), (18, "L2 projection onto\n{f : A f = c, f ≥ 0}", GREY)])
chain(13, "Task 2 onset", [(14, NETS, GREY), (12, "Ranker with\nadaptive weight", ORANGE), (9, "Evidence\nrule", GREY), (11, "Expected-IoU\nset", GREY)])
chain(5.5, "Task 2 ongoing", [(14, NETS, GREY), (6, "X1", BLUE), (9, "Attention\nnetwork 0.2", BLUE), (10, "No-memory\nLightGBM 0.2", ORANGE), (6, "Cone", BLUE), (9, "Evidence\nrule", BLUE), (11, "Expected-IoU\nset", GREY), (7, "Far gate", BLUE)])
for x, t, col in ((14, "both candidates on every window", GREY), (41, "both on March, A only on April", BLUE), (68, "A's April windows only", ORANGE)):
    ax.plot([x, x + 3], [-1.5, -1.5], color=col, lw=1.6, solid_capstyle="round"); ax.text(x + 4, -1.5, t, ha="left", va="center", fontsize=6, color=INK2)
save(fig, "fig2_pipeline")
print("figures written to", OUT)
