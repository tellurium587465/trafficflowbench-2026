"""G270 (11/04, ledger #228 stage 1, declared by the strategy before the results, auditor L4): M3 v3 dev ongoing windows split by the
selector's persistence filter computed as the official windows are reproduced (auditor's rep5.py, 36/40 ongoing: queue states with
gaps of <= 6 rows filled by the 'and' of forward and backward fills, links that are ever finite, base = the row at T, horizon T+1 ..
T+6). Uses the March (validation) release panel at the window's origin. Per stratum (persistence IoU <= 0.9 / > 0.9): the share and
A13's flow IoU (G212's dev a12; on dev a13 = a12), with the old filter's value for reference. Dev windows only (no hold-out).
usage: TFB_M3VER=v3 py -B -X utf8 g270_m3_persistence_strata.py"""
import os, sys, pickle
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
import m3v2
W = "C:/Users/user/tfb/work/t2/"; F = 6
META = {m["window_id"]: m for m in pickle.load(open(W + "m3v3_meta.pkl", "rb"))["meta"]}
D = pd.read_csv(W + "g212_dev.csv"); assert not m3v2.is_holdout(D.day.to_numpy()).any() and len(D) == 356


def impute_and(V, vcut, E, maxgap=6):
    q = np.where(np.isfinite(V), (V <= vcut[None, :]).astype(np.float32), np.nan); df = pd.DataFrame(q)
    f = df.ffill(limit=maxgap).to_numpy(); b = df.bfill(limit=maxgap).to_numpy()
    est = np.where(np.isfinite(f) & np.isfinite(b), np.minimum(f, b), np.where(np.isfinite(f), f, b)); out = np.where(np.isfinite(q), q, est)
    return (np.nan_to_num(out, nan=0.0) >= 0.5) & E


def piou(base, fut, el):
    p = np.broadcast_to(base, fut.shape) & el; t = fut & el; u = (p | t).sum(); return 1.0 if u == 0 else (p & t).sum() / u


LP = {}; rows = []
for w in D.w:
    m = META[w]; p = m["panel"]; t = int(m["t"])
    if p not in LP:
        P = load(p, "validation"); ever = np.isfinite(P.V).any(axis=0); LP[p] = (impute_and(P.V, P.vcut, P.E) & ever[None, :], ever, (np.nan_to_num(P.V, nan=1e9) <= P.vcut[None, :]) & P.E, np.isfinite(P.V) & P.E)
    Lp, ever, Qm, obs = LP[p]
    new = piou(Lp[t], Lp[t + 1:t + F + 1], np.broadcast_to(ever, (F, len(ever)))); old = piou(Qm[t - 1], Qm[t + 1:t + F + 1], obs[t + 1:t + F + 1])
    rows.append(dict(w=w, new_piou=new, old_piou=old))
S = D.merge(pd.DataFrame(rows), on="w"); S.to_csv(W + "g270_dev.csv", index=False)
S["lay"] = np.where(S.new_piou <= 0.9, "<= 0.9", "> 0.9")
for lay, g in S.groupby("lay"):
    print(f"new filter {lay}: {len(g)} windows ({len(g) / len(S):.1%}) | A13 flow IoU {g.a12.mean():.4f} (SE {g.a12.std() / np.sqrt(len(g)):.4f}) | mean new piou {g.new_piou.mean():.3f}, old piou {g.old_piou.mean():.3f}", flush=True)
print(f"old filter: windows with old piou > 0.9: {(S.old_piou > 0.9).sum()} | all {len(S)}: A13 flow IoU {S.a12.mean():.4f}", flush=True)
