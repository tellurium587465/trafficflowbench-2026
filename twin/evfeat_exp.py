"""Evidence-as-feature under map shift (10/13; MAP.md 5-4). Hypothesis: the within-month evidence features failed (q27/q28,
F29/F31) because (i) they were judged on the month CV, where the map does not move, and (ii) in the train months the
site memory (fingerprints) answers the same question more reliably, so a model never learns to follow the evidence.
Trained on worlds whose maps move (R), a model must use the evidence, and that use transfers to a new month.

One procedure for every world (so the comparison is apples to apples):
  onset event per panel-day = first morning row r (05:00-12:00) with a queue in the truth layer, no queue in the
  observed layer over r-18..r-7 (the visible hour); T = r - 6; label = truth queue at r (T+30), eligible cells only.
  Release blanks: rows T+1..T+18 of every day's window are hidden from the observed layer (as in the test months,
  where the windows sit on days 1-5, one per morning), before any evidence is counted.
Features per (window, link): dynamic state up to T (no time of day, no position), the training-config prior
(share of world-A6 days whose first morning queue covers the link +-2 = the site memory), and within-month
evidence from up to 3 earlier days (any morning queue / queue head at the link +-2, outside the blanks).
Models (LightGBM, rows = window x link): M1 dyn+prior, M2 dyn+evidence, M3 dyn+prior+evidence; trained on A6 or R.
Test: T, U, S worlds, days 1-5 (the test-month window days) and all days; expected-IoU set at T+30; baseline
'M1(A6) x rule' = M1 with links lacking evidence x0.25 (the production rule).
usage: python evfeat_exp.py build [family [lo hi]] | python evfeat_exp.py   (fit + report)"""
import sys, os, glob, json, pickle, time, numpy as np, pandas as pd, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/twin")
from cachelib import load, T2_PANELS

OUT = "E:/tfb_nn/evfeat/"; os.makedirs(OUT, exist_ok=True)
FAM = {"A6": ["A6"], "R": [f"R{k}" for k in range(48)], "T": [f"T{k}" for k in range(8)], "U": [f"U{k}" for k in range(8)],
       "S": [f"S{k}" for k in range(8)]}


def dil(x, d=2):
    e = x.copy()
    for o in range(1, d + 1):
        e[o:] |= x[:-o]; e[:-o] |= x[o:]
    return e


def world_layers(P):
    V = P.V; fin = np.isfinite(V)
    truth = (np.nan_to_num(V, nan=1e9) <= P.vcut[None, :]) & P.E
    obs_q = (V <= P.vcut[None, :]) & fin & ~P.blank
    return truth, obs_q, fin & ~P.blank


def events(P, truth, obs_q):
    D = P.T // 288; ev = {}
    for d in range(D):
        for s in range(72, 144):                                    # 06:00-12:00 so the visible hour exists
            r = d * 288 + s
            if truth[r].any():
                if not obs_q[r - 18:r - 6].any():
                    ev[d] = r
                break
    return ev


PRIOR = {}


def prior(p, exclude_month=None):
    """site memory: share of A6 days whose first morning queue covers link +-2; exclude_month leaves that month out
    (review 10/13: for A6 windows the prior must not contain the window's own label)"""
    key = (p, exclude_month)
    if key not in PRIOR:
        P = load(p, "twinA6"); truth, obs_q, _ = world_layers(P); ev = events(P, truth, obs_q)
        f = np.zeros(P.n); n = 0
        for d, r in ev.items():
            if exclude_month is not None and str(P.day[d])[:7] == exclude_month:
                continue
            f += dil(truth[r], 2); n += 1
        PRIOR[key] = f / max(n, 1)
    return PRIOR[key]


def build_world(p, w):
    P = load(p, f"twin{w}"); truth, obs_q, seen = world_layers(P); ev = events(P, truth, obs_q)
    hide = np.zeros(P.T, bool)
    for r in ev.values():
        hide[r - 6 + 1:r - 6 + 19] = True                          # T+1..T+18 blank, as the release does
    obs_q = obs_q & ~hide[:, None]; seen = seen & ~hide[:, None]
    R_ = np.where(seen, P.V / P.vcut[None, :], np.nan); Qc = np.where(seen, P.Q / P.cap[None, :], np.nan)
    pr = prior(p).astype(np.float32); n = P.n
    days = sorted(ev); morning = {}
    for d in range(P.T // 288):
        lo, hi = d * 288 + 60, d * 288 + 144
        q = obs_q[lo:hi]
        head = q & ~np.concatenate([q[:, 1:], np.zeros((q.shape[0], 1), bool)], axis=1)   # queued, downstream free
        morning[d] = (dil(q.any(axis=0), 2), dil(head.any(axis=0), 2))
    X, Y, G = [], [], []
    for d in days:
        r = ev[d]; t = r - 6
        y = truth[r] & P.E[r]
        with np.errstate(all="ignore"):
            v1 = R_[t - 1]; v6 = np.nanmean(R_[t - 6:t], axis=0); vmin = np.nanmin(R_[t - 12:t], axis=0)
            q6 = np.nanmean(Qc[t - 6:t], axis=0); q12 = np.nanmean(Qc[t - 12:t - 6], axis=0)
        def nb(a, f):
            out = np.full(n, np.nan)
            for o in (-2, -1, 1, 2):
                sh = np.roll(a, o); sh[(np.arange(n) - o < 0) | (np.arange(n) - o >= n)] = np.nan
                out = f(np.vstack([out, sh]), axis=0)
            return out
        vnb = nb(v6, np.nanmin); qdn = nb(q6, np.nanmax)
        prevs = [d - k for k in (1, 2, 3) if (d - k) in morning and (d - k) >= 0 and str(P.day[d - k])[:7] == str(P.day[d])[:7]]
        nd = len(prevs)
        e_any = np.mean([morning[k][0] for k in prevs], axis=0) if nd else np.full(n, np.nan)
        e_head = np.mean([morning[k][1] for k in prevs], axis=0) if nd else np.full(n, np.nan)
        e1 = morning[prevs[0]][0].astype(np.float32) if nd else np.full(n, np.nan)
        dom = int(str(P.day[d])[8:10])
        prd = prior(p, str(P.day[d])[:7] if w in ("A6", "B") else None).astype(np.float32)   # training worlds: month left out
        F = np.column_stack([v1, v6, vmin, q6, q6 / np.maximum(q12, 1e-3), vnb, qdn,             # dynamic
                             prd,                                                                   # site memory
                             e_any, e_head, e1, np.full(n, nd)]).astype(np.float32)                # evidence
        X.append(F); Y.append(y); G.append(np.full(n, len(G)))
    meta = dict(panel=p, world=w, days=[int(str(P.day[d])[8:10]) for d in days], n=n)
    if not X:
        return None
    return np.vstack(X), np.concatenate(Y), meta, [len(x) for x in X]


def build_pm(p, n_pm=60, L=8, dom=0.7, seed=0):
    """Pseudo-months from the single-configuration world A6 (the real-train analogue): each pseudo-month draws a
    dominant first-queue site and L days (dom of them from that site's days), in random order; the evidence features
    are recomputed from the previous pseudo-days. Decouples the evidence from the site memory without simulation."""
    rng = np.random.default_rng(seed + sum(map(ord, p)))
    P = load(p, "twinA6"); truth, obs_q, seen = world_layers(P); ev = events(P, truth, obs_q)
    hide = np.zeros(P.T, bool)
    for r in ev.values():
        hide[r - 6 + 1:r - 6 + 19] = True
    obs_q = obs_q & ~hide[:, None]; seen = seen & ~hide[:, None]
    R_ = np.where(seen, P.V / P.vcut[None, :], np.nan); Qc = np.where(seen, P.Q / P.cap[None, :], np.nan)
    pr = prior(p).astype(np.float32); n = P.n
    base, mor, site = {}, {}, {}
    for d, r in ev.items():
        t = r - 6; y = truth[r] & P.E[r]
        if not y.any():
            continue
        with np.errstate(all="ignore"):
            v1 = R_[t - 1]; v6 = np.nanmean(R_[t - 6:t], axis=0); vmin = np.nanmin(R_[t - 12:t], axis=0)
            q6 = np.nanmean(Qc[t - 6:t], axis=0); q12 = np.nanmean(Qc[t - 12:t - 6], axis=0)
        def nb(a, f):
            out = np.full(n, np.nan)
            for o in (-2, -1, 1, 2):
                sh = np.roll(a, o); sh[(np.arange(n) - o < 0) | (np.arange(n) - o >= n)] = np.nan
                out = f(np.vstack([out, sh]), axis=0)
            return out
        base[d] = (np.column_stack([v1, v6, vmin, q6, q6 / np.maximum(q12, 1e-3), nb(v6, np.nanmin), nb(q6, np.nanmax), prior(p, str(P.day[d])[:7])]), y)
        lo, hi = d * 288 + 60, d * 288 + 144; q = obs_q[lo:hi]
        head = q & ~np.concatenate([q[:, 1:], np.zeros((q.shape[0], 1), bool)], axis=1)
        mor[d] = (dil(q.any(axis=0), 2), dil(head.any(axis=0), 2)); site[d] = int(np.flatnonzero(y).max()) // 3
    days = sorted(base)
    if len(days) < L:
        return None
    by = {}
    for d in days:
        by.setdefault(site[d], []).append(d)
    cands = [b for b, v in by.items() if len(v) >= 2]
    X, Y, dd_ = [], [], []
    for m in range(n_pm):
        b = cands[rng.integers(len(cands))]
        # review 10/13: distinct days (sampling with replacement put a day among its own 'earlier days' = label leak)
        k_dom = min(len(by[b]), int(round(dom * L))); others = [d for d in days if d not in set(by[b])]
        if k_dom + len(others) < L:
            continue
        pick = list(rng.choice(by[b], k_dom, replace=False)) + list(rng.choice(others, L - k_dom, replace=False))
        pick = [int(x) for x in rng.permutation(pick)]
        for j, d in enumerate(pick):
            prevs = pick[max(0, j - 3):j][::-1]; nd = len(prevs)
            e_any = np.mean([mor[k][0] for k in prevs], axis=0) if nd else np.full(n, np.nan)
            e_head = np.mean([mor[k][1] for k in prevs], axis=0) if nd else np.full(n, np.nan)
            e1 = mor[prevs[0]][0].astype(np.float32) if nd else np.full(n, np.nan)
            F = np.column_stack([base[d][0], e_any, e_head, e1, np.full(n, nd)]).astype(np.float32)
            X.append(F); Y.append(base[d][1]); dd_.append(j + 1)
    return np.vstack(X), np.concatenate(Y), dict(panel=p, world="A6pm", days=dd_, n=n), [len(x) for x in X]


COLS = ["v1", "v6", "vmin", "q6", "qgrow", "vnb", "qdn", "prior", "e_any", "e_head", "e1", "nd"]
SETS = {"M1": ["v1", "v6", "vmin", "q6", "qgrow", "vnb", "qdn", "prior"],
        "M2": ["v1", "v6", "vmin", "q6", "qgrow", "vnb", "qdn", "e_any", "e_head", "e1", "nd"],
        "M3": COLS}


def build(only=None, lo=0, hi=None):
    for fam, ws in FAM.items():
        if only and fam != only:
            continue
        if hi is not None:
            ws = ws[lo:hi]; fam_out = f"{fam}_{lo}_{hi}"
        else:
            fam_out = fam
        f = OUT + f"{fam_out}.pkl"
        if os.path.exists(f):
            continue
        t0 = time.time(); blocks = []
        for p in T2_PANELS:
            for w in ws:
                b = build_world(p, w)
                if b is not None:
                    blocks.append(b)
        pickle.dump(blocks, open(f, "wb")); print(fam_out, len(blocks), "panel-worlds", f"{time.time()-t0:.0f}s", flush=True)


def bestset(p):
    o = np.argsort(-p); cs = np.cumsum(p[o]); kk = np.arange(1, p.size + 1)
    kb = int(np.argmax(cs / (kk + p.sum() - cs))) + 1; s = np.zeros(p.size, bool); s[o[:kb]] = True; return s


def stack(fam):
    B = [b for f in sorted(glob.glob(OUT + f"{fam}.pkl") + glob.glob(OUT + f"{fam}_*_*.pkl")) for b in pickle.load(open(f, "rb"))]
    X = np.vstack([b[0] for b in B]); Y = np.concatenate([b[1] for b in B])
    win = []                                                          # (start, len, panel, world, day)
    off = 0
    for b in B:
        for L, dd in zip(b[3], b[2]["days"]):
            win.append((off, L, b[2]["panel"], b[2]["world"], dd)); off += L
    return pd.DataFrame(X, columns=COLS), Y, win


def fit():
    import lightgbm as lgb
    Xa, Ya, _ = stack("A6"); Xr, Yr, winR = stack("R")
    if os.environ.get("EVF_R_MATCH", "0") == "1":                # review 10/13: same template months as A6 and same size
        keep = [k for k, (st, L, p, w, dd) in enumerate(winR)
                if json.load(open(f"C:/Users/user/tfb/work/twin/{p}_twin{w}.json"))["template"] < "2030-12"]
        na = len(stack("A6")[2]); keep = sorted(np.random.default_rng(0).choice(keep, min(na, len(keep)), replace=False))
        rows = np.concatenate([np.arange(winR[k][0], winR[k][0] + winR[k][1]) for k in keep])
        Xr, Yr = Xr.iloc[rows].reset_index(drop=True), Yr[rows]
        print("R matched to A6:", len(keep), "windows", flush=True)
    prm = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, num_threads=16, verbose=-1, seed=0)
    models = {}
    Xp, Yp, _ = stack("A6pm")
    for tr_name, (X, Y) in (("A6", (Xa, Ya)), ("R", (Xr, Yr)), ("A6pm", (Xp, Yp))):
        for m, cols in SETS.items():
            models[(m, tr_name)] = lgb.train(prm, lgb.Dataset(X[cols], Y.astype(int)), 400)
            print("fit", m, tr_name, flush=True)
    res = []
    for fam in ("T", "U", "S"):
        X, Y, win = stack(fam)
        P = {k: mdl.predict(X[SETS[k[0]]], num_threads=16) for k, mdl in models.items()}
        for (st, L, p, w, dd) in win:
            y = Y[st:st + L]
            if not y.any():
                continue
            r = dict(fam=fam, panel=p, world=w, day=dd, nd=float(X.nd.iloc[st]))
            for k, pp in P.items():
                s = bestset(pp[st:st + L]); r[f"{k[0]}_{k[1]}"] = (s & y).sum() / (s | y).sum()
            pr = P[("M1", "A6")][st:st + L].copy(); ev = X.e_any.to_numpy()[st:st + L]
            if np.isfinite(ev).all():
                pr = pr * np.where(ev > 0, 1.0, 0.25)
            s = bestset(pr); r["M1_A6_rule"] = (s & y).sum() / (s | y).sum()
            res.append(r)
    R = pd.DataFrame(res); R.to_csv(OUT + "results.csv", index=False)
    cols = [c for c in R.columns if c.startswith("M")]
    print("all days (mean IoU at T+30):"); print(R.groupby("fam")[cols].mean().round(4).to_string())
    m5 = R.day <= 5
    print("days 1-5:"); print(R[m5].groupby("fam")[cols].mean().round(4).to_string())
    print("day 1 only:"); print(R[R.day == 1].groupby("fam")[cols].mean().round(4).to_string())
    rng = np.random.default_rng(0); R["cl"] = R.panel + R.world
    base = "M1_A6_rule"
    for c in cols:
        if c == base:
            continue
        d = (R[c] - R[base]).to_numpy(); cl = R.cl.to_numpy(); u = np.unique(cl); idx = {x: np.flatnonzero(cl == x) for x in u}
        se = np.std([d[np.concatenate([idx[x] for x in rng.choice(u, len(u))])].mean() for _ in range(300)])
        print(f"{c:10s} - {base}: {d.mean():+.4f} (SE {se:.4f}); days 1-5 {d[m5.to_numpy()].mean():+.4f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build_pm":
        blocks = [b for b in (build_pm(p) for p in T2_PANELS) if b is not None]
        pickle.dump(blocks, open(OUT + "A6pm.pkl", "wb")); print("A6pm", len(blocks), sum(len(b[3]) for b in blocks), "windows")
    elif len(sys.argv) > 1 and sys.argv[1] == "build":
        a = sys.argv[2:]
        build(a[0] if a else None, int(a[1]) if len(a) > 2 else 0, int(a[2]) if len(a) > 2 else None)
    else:
        fit()
