"""Learned evidence fusion (10/12): a small model maps (NN probability, same-month evidence) -> adjusted probability.
The fixed rules (onset x0.25 / ongoing x0.85 for links without evidence) generalised. Fitting it on the train months
failed before (q27/q28: the map does not move there, so evidence carries no extra information); the twin's randomized
worlds have moved maps, so the rule can be learned where it matters.
Fit on the R worlds (A-trained NN predictions, twR_nomem), score on the held-out test worlds T (tw2_nomem), against the
fixed rules; far gate with the contiguous-growth exemption applied to both. Evidence = the masked layer, <= 3 earlier
days of the same month, as in t2_onset_evidence.py.
usage: python adjuster.py"""
import sys, pickle, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load

W = "C:/Users/user/tfb/work/t2/"; PC = {}


def probs(spec):
    P = [pickle.load(open(W + f"probs_{t}.pkl", "rb")) for t in spec.split("+")]
    return {k: sum(p[k] for p in P) / len(P) for k in P[0]}


def world(panel, wd):
    if (panel, wd) not in PC:
        P = load(panel, f"twin{wd}")
        q = (np.nan_to_num(P.V, nan=1e9) <= P.vcut[None, :]) & np.isfinite(P.V)
        PC[(panel, wd)] = (P, q & ~P.blank, (np.nan_to_num(P.V, nan=1e9) <= P.vcut[None, :]) & P.E)
    return PC[(panel, wd)]


def dil(x, d):
    e = x.copy()
    for o in range(1, d + 1):
        e[o:] |= x[:-o]; e[:-o] |= x[o:]
    return e


def feats(wid, rec, p):
    """per-window feature blocks: onset -> (n links) rows for T+30; ongoing -> (6 x n) rows"""
    y, el, panel, wd, t, cond, date = rec
    P, obs, full = world(panel, wd); n = P.n
    days_on, days_og = [], []
    for b in (1, 2, 3):
        dd = t - b * 288
        if dd - 18 >= 0:
            days_on.append(obs[dd - 6:dd + 12].any(axis=0)); days_og.append(obs[dd - 18:dd + 12].any(axis=0))
    nd = len(days_on)
    on0 = np.sum(days_on, axis=0) if nd else np.zeros(n); on2 = np.sum([dil(x, 2) for x in days_on], axis=0) if nd else np.zeros(n)
    og0 = np.sum(days_og, axis=0) if nd else np.zeros(n); og1 = np.sum([dil(x, 1) for x in days_og], axis=0) if nd else np.zeros(n)
    lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    if cond == "queue_onset":
        pl = p[-1]; rk = pl.argsort().argsort() / max(n - 1, 1)
        X = np.column_stack([lg[-1], on0, on2, np.full(n, nd), rk, np.full(n, pl.max()), on2 / max(nd, 1)])
        return X, y.T[-1].astype(bool), el.T[-1].astype(bool)
    lq = full[t - 1]; dist = np.full(n, 99.0)
    for j in np.flatnonzero(lq):
        dist = np.minimum(dist, np.abs(np.arange(n) - j))
    rows = []
    for s in range(6):
        rows.append(np.column_stack([lg[s], np.full(n, s), og0, og1, np.full(n, nd), np.minimum(dist, 20), lq.astype(float), og1 / max(nd, 1)]))
    return np.vstack(rows), y.T.astype(bool).ravel(), el.T.astype(bool).ravel()


def dataset(labels, pr):
    D = {"queue_onset": [], "queue_ongoing": []}
    for wid, rec in labels.items():
        if wid in pr:
            D[rec[5]].append((wid, *feats(wid, rec, pr[wid])))
    return D


def bestset(p):
    pe = p.ravel(); o = np.argsort(-pe); cs = np.cumsum(pe[o]); kk = np.arange(1, pe.size + 1)
    kb = int(np.argmax(cs / (kk + pe.sum() - cs))) + 1; s = np.zeros(pe.size, bool); s[o[:kb]] = True
    return s.reshape(p.shape)


def gate(s, rec):
    y, el, panel, wd, t, cond, date = rec; P, obs, full = world(panel, wd); n = P.n
    seen = np.zeros(n, bool)
    for b in (1, 2, 3):
        dd = t - b * 288
        if dd - 18 >= 0:
            seen |= obs[dd - 18:dd + 12].any(axis=0)
    d0 = t - (t % 288) + 60; seen |= obs[d0:t].any(axis=0) if t > d0 else False; gev = dil(seen, 2)
    near = np.zeros(n, bool)
    for j in np.flatnonzero(full[t - 1]):
        near[max(0, j - 3):j + 4] = True
    keep = np.zeros_like(s)
    for r in range(s.shape[0]):
        j = 0
        while j < n:
            if s[r, j]:
                e = j
                while e + 1 < n and s[r, e + 1]:
                    e += 1
                if near[j:e + 1].any():
                    keep[r, j:e + 1] = True
                j = e + 1
            else:
                j += 1
    return s & ~((~near & ~gev)[None, :] & ~keep)


def rule_probs(rec, p):
    y, el, panel, wd, t, cond, date = rec; P, obs, full = world(panel, wd); n = P.n; p = p.copy()
    seen_on = np.zeros(n, bool); seen_og = np.zeros(n, bool); nd = 0
    for b in (1, 2, 3):
        dd = t - b * 288
        if dd - 18 >= 0:
            seen_on |= obs[dd - 6:dd + 12].any(axis=0); seen_og |= obs[dd - 18:dd + 12].any(axis=0); nd += 1
    if cond == "queue_onset":
        p[:-1] = 0
        if nd:
            p[-1] = p[-1] * np.where(dil(seen_on, 2), 1.0, 0.25)
    elif nd:
        p = p * np.where(dil(seen_og, 1), 1.0, 0.85)[None, :]
    return p


if __name__ == "__main__":
    LR = pickle.load(open("E:/tfb_nn/twin_labels_twR.pkl", "rb")); LT = pickle.load(open("E:/tfb_nn/twin_labels.pkl", "rb"))
    PRr = probs("twR_nomem_s0+twR_nomem_s1"); PRt = probs("tw2_nomem_s0+tw2_nomem_s1")
    Dr, Dt = dataset(LR, PRr), dataset(LT, PRt)
    models = {}
    for cond, mono in (("queue_onset", [1, 1, 1, 0, 1, 0, 1]), ("queue_ongoing", [1, 0, 1, 1, 0, 0, 0, 1])):
        X = np.vstack([b[1] for b in Dr[cond]]); y = np.concatenate([b[2] for b in Dr[cond]]); m = np.concatenate([b[3] for b in Dr[cond]])
        prm = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, monotone_constraints=mono,
                   feature_fraction=1.0, verbose=-1, num_threads=8)
        models[cond] = lgb.train(prm, lgb.Dataset(X[m], y[m]), 300)
        print(cond, "fit on R worlds:", int(m.sum()), "cells", flush=True)
    rows = []
    for cond in ("queue_onset", "queue_ongoing"):
        for wid, X, yv, mv in Dt[cond]:
            rec = LT[wid]; y, el = rec[0].T.astype(bool), rec[1].T.astype(bool); p0 = PRt[wid]
            pa = models[cond].predict(X)
            if cond == "queue_onset":
                pad = np.zeros_like(p0); pad[-1] = pa
            else:
                pad = pa.reshape(6, -1)
            out = {}
            for name, pp in (("rule", rule_probs(rec, p0)), ("learned", pad)):
                s = bestset(pp)
                if cond == "queue_ongoing":
                    s = gate(s, rec)
                u = ((s | y) & el).sum(); out[name] = 1.0 if u == 0 else (s & y & el).sum() / u
            if cond == "queue_ongoing":
                P, obs, full = world(rec[2], rec[3]); pers = full[rec[4]]; fut = y & el; pp_ = np.broadcast_to(pers, fut.shape) & el; uu = (pp_ | fut).sum()
                if uu and (pp_ & fut).sum() / uu > 0.9:
                    continue
            rows.append(dict(cond=cond, cl=rec[2] + rec[3], day=int(rec[6][8:10]), d=out["learned"] - out["rule"], rule=out["rule"]))
    R = pd.DataFrame(rows); rng = np.random.default_rng(0)
    for cond in ("queue_onset", "queue_ongoing"):
        for lab, m in (("all", np.ones(len(R), bool)), ("days<=4", R.day.to_numpy() <= 4)):
            x = R[(R.cond == cond).to_numpy() & m]; u = x.cl.unique(); g = {c: x[x.cl == c].d.to_numpy() for c in u}
            se = np.std([np.concatenate([g[c] for c in rng.choice(u, len(u))]).mean() for _ in range(500)])
            print(f"{cond} {lab}: n={len(x)} rule {x.rule.mean():.4f} learned - rule {x.d.mean():+.4f} (SE {se:.4f})")
    pickle.dump(models, open("C:/Users/user/tfb/work/twin/adjuster_models.pkl", "wb"))
