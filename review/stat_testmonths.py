"""Statistician review (10/07): label-free comparison of train / validation(=public LB) / private months.
(a) window timing (public window_index), (b) do March and April share the same per-link re-draw (free-flow speed,
congestion map)?, (c) model self-assessed expected IoU and between-model disagreement per split.
Read-only; prints to stdout."""
import sys, os, glob, pickle, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src"); os.chdir(r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS, PANELS

REL = r"C:/Users/user/tfb/data/kaggle_public/task2"
# ---------- (a) window timing ----------
wi = pd.concat([pd.read_csv(f) for f in glob.glob(REL + "/*/*/window_index.csv")])
wi["ts"] = pd.to_datetime(wi.forecast_origin.str[:19]); wi["hour"] = wi.ts.dt.hour + wi.ts.dt.minute / 60
wi["dow"] = wi.ts.dt.dayofweek; wi["dom"] = wi.ts.dt.day; wi["wkend"] = wi.dow >= 5
print("# (a) window timing by split x condition")
g = wi.groupby(["split", "condition"])
print(g.agg(n=("hour", "size"), hour_med=("hour", "median"), am_share=("hour", lambda h: np.mean(h < 10)),
            weekend=("wkend", "mean"), dom_max=("dom", "max")).round(3).to_string())
print("onset hour histogram (2h bins):")
for s in ("train", "validation", "private"):
    h = wi[(wi.split == s) & (wi.condition == "queue_onset")].hour
    print(f"  {s:10s}", np.histogram(h, bins=range(0, 26, 2))[0])

# ---------- (b) shared re-draw? per-link free-flow level and congestion map ----------
print("\n# (b) per-link free-flow speed offset vs train mean, and congestion-map deviation (March vs April)")
rows = []
for p in PANELS:
    tr = load(p, "train"); va = load(p, "validation"); pv = load(p, "private")
    def ff(P, rows_=None):
        V = P.V if rows_ is None else P.V[rows_]
        ok = np.isfinite(V) & (V >= 0.85 * P.vf[None, :])
        return np.nanmedian(np.where(ok, V, np.nan), axis=0)
    def qrate(P, rows_=None):
        V = P.V if rows_ is None else P.V[rows_]; E = P.E if rows_ is None else P.E[rows_]
        obs = np.isfinite(V) & E
        return ((np.nan_to_num(V, nan=1e9) <= P.vcut[None, :]) & obs).sum(0) / np.maximum(obs.sum(0), 1)
    months = np.array([t[:7] for t in tr.times]); um = sorted(set(months))
    ffm = np.array([ff(tr, months == m) for m in um]); qm = np.array([qrate(tr, months == m) for m in um])
    base_ff = np.nanmean(ffm, 0); base_q = qm.mean(0)
    dM = ff(va) / base_ff - 1; dA = ff(pv) / base_ff - 1
    dT = ffm[-1] / np.nanmean(ffm[:-1], 0) - 1           # a train month vs the other train months
    qM, qA = qrate(va), qrate(pv)
    tv = lambda a, b: 0.5 * np.abs(a / a.sum() - b / b.sum()).sum()
    tvT = np.mean([tv(qm[i], qm[np.arange(len(um)) != i].mean(0)) for i in range(len(um))])
    ok = np.isfinite(dM) & np.isfinite(dA)
    rows.append(dict(panel=p, sd_dTrain=np.nanstd(dT) * 100, sd_dMar=np.nanstd(dM) * 100, sd_dApr=np.nanstd(dA) * 100,
                     r_ff_MarApr=np.corrcoef(dM[ok], dA[ok])[0, 1],
                     tv_trm=tvT, tv_Mar=tv(qM, base_q), tv_Apr=tv(qA, base_q), tv_MarApr=tv(qM, qA),
                     r_qdev_MarApr=np.corrcoef(qM - base_q, qA - base_q)[0, 1],
                     qrate_tr=base_q.mean() * 100, qrate_Mar=qM.mean() * 100, qrate_Apr=qA.mean() * 100))
    del tr, va, pv
b = pd.DataFrame(rows).set_index("panel")
print(b.round(3).to_string())
print("means:", b.mean().round(3).to_dict())

# ---------- (c) self-assessed expected IoU and disagreement on the test windows ----------
print("\n# (c) model self-assessed E[IoU] (best_set) and set agreement, per split x condition")
W = r"C:/Users/user/tfb/work/t2/"
M = {"v8(prior)": "probs_v8.pkl", "v9npp": "probs_v9npp.pkl", "s17": "probs_s17_t1f3ens3c_t2v12nppdem_nnDnp6_t4l2.pkl",
     "s17L": "probs_s17L_t1f3ens3c_t2v12nppdem_t4l2.pkl", "s23": "probs_s23_t1GapML_t2v14af_nnAf6_t4l2.pkl"}
PR = {m: pickle.load(open(W + f, "rb")) for m, f in M.items()}
cond = dict(zip(wi.window_id, wi.condition))
def bs(o):
    pe = o.ravel(); order = np.argsort(-pe); cs = np.cumsum(pe[order]); kk = np.arange(1, pe.size + 1)
    ei = cs / (kk + pe.sum() - cs); kb = int(np.argmax(ei)) + 1
    s = np.zeros(pe.size, bool); s[order[:kb]] = True
    return s, ei[kb - 1]
SET, EI = {}, {}
for m, d in PR.items():
    for w, o in d.items():
        s, e = bs(np.asarray(o)); SET[(m, w)] = s; EI[(m, w)] = e
wids = sorted(PR["v9npp"])
out = []
for w in wids:
    sp = "validation" if "_validation_" in w else "private"
    r = dict(split=sp, cond=cond[w])
    for m in M:
        r["E_" + m] = EI[(m, w)]
    for a, c in (("v8(prior)", "v9npp"), ("v9npp", "s17"), ("s17", "s23"), ("s17L", "s17")):
        A, C = SET[(a, w)], SET[(c, w)]; r[f"agree_{a}|{c}"] = (A & C).sum() / max((A | C).sum(), 1)
    out.append(r)
o = pd.DataFrame(out)
print(o.groupby(["split", "cond"]).mean(numeric_only=True).round(3).T.to_string())
print("\nshare of windows with agreement IoU < 0.5:")
print(o.groupby(["split", "cond"])[[c for c in o if c.startswith("agree")]].agg(lambda x: np.mean(x < 0.5)).round(3).T.to_string())
