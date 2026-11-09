"""G346 (11/05, ledger #276, measurement only): how far the truth moves from persistence, per horizon row, on the 80 official June ongoing
windows (G342a labels) against the June replica selector ongoing windows (G321's 163, evaluator labels). Persistence = the true queue state at
row T-1 (queue_matrix of the full training data; t2_official_check's persistence). Per row r = 1..6 (T+5 .. T+30), on the scored cells:
IoU(persistence, truth), the share of scored cells switched on (0 -> 1) and off (1 -> 0) against T-1, and the true queued count / the
persistence count. Window means, official minus replica with SE. usage: py -B -X utf8 g346_persistence_shift.py"""
import os, sys, pickle
os.environ["TFB_T2CACHE"] = "2_noT_af"
import numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from t2_windows import queue_matrix
from t2_eval_oof import T
W = "C:/Users/user/tfb/work/t2/"; A = pickle.load(open(W + "g342_jun80.pkl", "rb"))
rep = pd.read_csv(W + "g321_jun_ongoing.csv").w.astype(int).tolist()
items = [("official", r["panel"], r["t"], r["lab"].astype(bool), r["el"].astype(bool)) for r in A["rows"] if r["cond"] == "queue_ongoing"]
items += [("replica", str(T.META["panel"][k]), int(T.META["t"][k]), T.LAB[k][0].astype(bool), T.LAB[k][1].astype(bool)) for k in rep]
QM = {}; out = []
for p in T2_PANELS:
    QM[p] = queue_matrix(load(p, "train")).astype(bool); load.cache_clear()
for src, p, t, y, e in items:
    ps = QM[p][t - 1]; r_ = dict(src=src)
    for r in range(6):
        sc = e[r]; u = ((ps | y[r]) & sc).sum(); r_[f"iou{r + 1}"] = 1.0 if u == 0 else float((ps & y[r] & sc).sum() / u)
        n = max(int(sc.sum()), 1); r_[f"on{r + 1}"] = float((~ps & y[r] & sc).sum() / n); r_[f"off{r + 1}"] = float((ps & ~y[r] & sc).sum() / n)
        r_[f"cnt{r + 1}"] = float((y[r] & sc).sum() / max((ps & sc).sum(), 1))
    out.append(r_)
D = pd.DataFrame(out); D.to_csv(W + "g346_persist.csv", index=False)
a, b = D[D.src == "official"], D[D.src == "replica"]; print(f"official {len(a)} windows, replica {len(b)}", flush=True)
for nm in ("iou", "on", "off", "cnt"):
    cells = []
    for r in range(1, 7):
        x, z = a[f"{nm}{r}"], b[f"{nm}{r}"]; se = np.sqrt(x.var(ddof=1) / len(x) + z.var(ddof=1) / len(z)); cells.append(f"T+{5 * r}: {x.mean():.3f} / {z.mean():.3f} ({x.mean() - z.mean():+.3f} +- {se:.3f})")
    print(f"{nm} (official / replica): " + " | ".join(cells), flush=True)
