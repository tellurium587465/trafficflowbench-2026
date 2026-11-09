"""G135b (10/31, ledger #95, after the result): is the X1 harm on January / February an artefact of the probe? No fitting, reads work/t2/x1_months/*.pkl and g135_x1_months.csv.
  (a) per seed, (b) by day of month (later days fine-tune on more windows), (c) what the fine-tune did to the probabilities: mean p, cells decoded, truth cells, recall and precision.
usage: TFB_M3VER=v3 py g135b_x1_diag.py"""
import os, pickle, importlib.util, numpy as np, pandas as pd
os.environ.setdefault("TFB_M3VER", "v3")
R0 = r"C:/Users/user/tfb/"; OUT = R0 + "work/t2/x1_months/"
spec = importlib.util.spec_from_file_location("g119", R0 + "review/g119_month_postproc.py"); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g); T = g.T
TAGS = [f"capB0_s{s}_fold2030-12" for s in range(4)]; PR = [pickle.load(open(OUT + t + ".pkl", "rb")) for t in TAGS]
C = pd.read_csv(OUT + "g135_x1_months.csv"); C["d"] = C.raw_x1 - C.raw_base
print("(b) by day of month, raw x1 - base")
for mo, d in C.groupby("month"):
    for lo, hi in ((0, 6), (7, 13), (14, 20), (21, 31)):
        e = d[(d.day >= lo) & (d.day <= hi)]; print(f"  {mo} day {lo + 1:2d}-{hi + 1:2d}: windows {len(e):3d} base {e.raw_base.mean():.4f} x1 - base {e.d.mean():+.4f}")
print("(a) per seed, raw x1 - base")
rows = []
for k in C.w:
    lab, el = T.LAB[k]; y, e = lab.astype(bool), el.astype(bool); mo = g.MON[k]; r = dict(w=k, month=mo)
    for i, P in enumerate(PR):
        r[f"s{i}"] = g.iou(g.decode(P[k][1]), y, e, "queue_ongoing") - g.iou(g.decode(P[k][0]), y, e, "queue_ongoing")
    for nm, j in (("b", 0), ("x", 1)):
        pm = sum(P[k][j] for P in PR) / len(PR); q = g.decode(pm).astype(bool); r["p_" + nm] = float(pm[e].mean()) if e.any() else np.nan; r["n_" + nm] = int((q & e).sum()); r["hit_" + nm] = int((q & y & e).sum())
    r["n_y"] = int((y & e).sum()); r["changed"] = float(np.abs(sum(P[k][1] for P in PR) / len(PR) - sum(P[k][0] for P in PR) / len(PR)).max()); rows.append(r)
D = pd.DataFrame(rows)
for mo, d in D.groupby("month"):
    print(f"  {mo}", " ".join(f"s{i} {d[f's{i}'].mean():+.4f}" for i in range(len(PR))))
print("(c) what changed (eligible cells, seed mean): mean p, cells decoded, truth, precision, recall")
for mo, d in D.groupby("month"):
    for nm in ("b", "x"):
        print(f"  {mo} {'base' if nm == 'b' else 'x1  '}: p {d['p_' + nm].mean():.4f} decoded {d['n_' + nm].sum():5d} truth {d.n_y.sum():5d} precision {d['hit_' + nm].sum() / max(d['n_' + nm].sum(), 1):.4f} recall {d['hit_' + nm].sum() / max(d.n_y.sum(), 1):.4f}")
    print(f"    max |dp| per window: median {d.changed.median():.3f} 90% {d.changed.quantile(.9):.3f}")
