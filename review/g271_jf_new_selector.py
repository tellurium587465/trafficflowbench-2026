"""G271 (11/04, ledger #228 stage 2a): January / February ongoing selector windows rebuilt with the persistence filter as the official
windows are reproduced (auditor's rep5.py: 'and' fill of gaps <= 6 rows, ever-finite links, base = the row at T); the rest as
t2_windows.candidates (same calendar day, history coverage >= 0.7, >= 2 queued history cells with >= 2 on one link, a queue in the
horizon, 6 h spacing in random order, seed 0). Counts the new windows, their overlap with the old selection (T.META is_sel) and which
of them have saved OOF probabilities (capB0 fold 2 / q25_nomem). Training split only. Writes work/t2/g271_new_windows.csv.
usage: TFB_M3VER=v3 py -B -X utf8 g271_jf_new_selector.py"""
import os, sys, pickle, importlib.util
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
R = r"C:/Users/user/tfb/"; a_ = sys.argv; sys.argv = ["g119", "run"]
spec = importlib.util.spec_from_file_location("g119", R + "review/g119_month_postproc.py"); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g); sys.argv = a_
T, MON, W = g.T, g.MON, g.W; H, F = 12, 6; M = T.META
KEY = {(str(p), int(t), str(c)): k for k, (p, t, c) in enumerate(zip(M["panel"], M["t"], M["cond"]))}
PC = set(pickle.load(open(W + "probs_jfmh0__capB0_s0_fold2030-12.pkl", "rb"))); PQ = set(pickle.load(open(W + "oof_q25_nomem_s3.pkl", "rb")))


def impute_and(V, vcut, E, maxgap=6):
    q = np.where(np.isfinite(V), (V <= vcut[None, :]).astype(np.float32), np.nan); df = pd.DataFrame(q)
    f = df.ffill(limit=maxgap).to_numpy(); b = df.bfill(limit=maxgap).to_numpy()
    est = np.where(np.isfinite(f) & np.isfinite(b), np.minimum(f, b), np.where(np.isfinite(f), f, b)); out = np.where(np.isfinite(q), q, est)
    return (np.nan_to_num(out, nan=0.0) >= 0.5) & E


def piou(base, fut, el):
    p = np.broadcast_to(base, fut.shape) & el; t = fut & el; u = (p | t).sum(); return 1.0 if u == 0 else (p & t).sum() / u


rows = []; rng = np.random.default_rng(0)
for p in sorted({str(x) for x in M["panel"]}):
    P = g.load(p, "train"); Qm = (np.nan_to_num(P.V, nan=1e9) <= P.vcut[None, :]) & P.E; obs = np.isfinite(P.V) & P.E; ever = np.isfinite(P.V).any(axis=0)
    Lp = impute_and(P.V, P.vcut, P.E) & ever[None, :]; mon = np.array([s[:7] for s in P.times]); cand = []
    for t in range(H, P.T - F):
        if mon[t] not in g.MONTHS or P.times[t - H][:10] != P.times[t + F][:10] or obs[t - H:t].mean() < 0.7:
            continue
        hq = Qm[t - H:t]
        if not hq.any() or not Qm[t + 1:t + F + 1].any() or not (hq.sum() >= 2 and hq.sum(axis=0).max() >= 2):
            continue
        new = piou(Lp[t], Lp[t + 1:t + F + 1], np.broadcast_to(ever, (F, len(ever)))); old = piou(Qm[t - 1], Qm[t + 1:t + F + 1], P.E[t + 1:t + F + 1])
        cand.append((t, new, old))
    for lab, keep in (("new", lambda r: r[1] <= 0.9), ("old", lambda r: r[2] <= 0.9)):
        ts = [r for r in cand if keep(r)]; chosen = []
        for i in rng.permutation(len(ts)):
            if all(abs(ts[i][0] - o[0]) >= 72 for o in chosen):
                chosen.append(ts[i])
        for t, new, old in chosen:
            k = KEY.get((p, t, "queue_ongoing")); rows.append(dict(sel=lab, panel=p, t=t, month=mon[t], new_piou=new, old_piou=old, k=k, is_sel=bool(M["is_sel"][k]) if k is not None else False, has_capB0=(str(k) in PC) if k is not None else False, has_q25=(k in PQ) if k is not None else False))
    g.load.cache_clear(); print(p, len(cand), "candidates", flush=True)
D = pd.DataFrame(rows); D.to_csv(W + "g271_new_windows.csv", index=False)
for lab, d in D.groupby("sel"):
    print(f"{lab} filter: {len(d)} windows (Jan {int((d.month == '2031-01').sum())}, Feb {int((d.month == '2031-02').sum())}) | in the old selection (is_sel) {int(d.is_sel.sum())} | with capB0 OOF {int(d.has_capB0.sum())} | with q25 OOF {int(d.has_q25.sum())} | mean new piou {d.new_piou.mean():.3f} old {d.old_piou.mean():.3f}", flush=True)
O = D[D.sel == "old"]; print("old-filter windows whose new piou > 0.9:", int((O.new_piou > 0.9).sum()), "of", len(O))
