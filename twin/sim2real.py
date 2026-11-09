"""Zero-shot sim -> real for onset site choice (10/13): does the improved twin (gen2.py: stochastic site capacities)
teach a mapping 'state at T -> onset site' that holds on REAL data better than the current twin does?
Same onset-event procedure, release blanks and features as evfeat_exp.py. All training material uses the train months
0-5 (Jun-Nov) as templates / identification; the test is the REAL months 6-8 (Dec-Feb).
  base     : twinA6 + R worlds whose template is in months 0-5 (current twin)
  improved : twinB + twinQ0..15 (gen2.py)
  real     : real months 0-5 (reference: what in-distribution real training gives)
Models: M0 = dynamics only (the pure mapping), M1 = dynamics + the site prior of the model's own training world
(at test: the real months 0-5 prior). Score: mean IoU at T+30 on the real test onsets, cluster bootstrap over panels.
usage: python sim2real.py build [panel,panel] | python sim2real.py   (fit + report over all built blocks)"""
import sys, os, glob, json, pickle, time, numpy as np, pandas as pd, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/twin")
from cachelib import load, T2_PANELS
_argv = list(sys.argv); sys.argv = ["x"]
exec(open(r"C:/Users/user/tfb/twin/evfeat_exp.py", encoding="utf-8").read().split("COLS = [")[0])
sys.argv = _argv
OUT2 = "E:/tfb_nn/sim2real/"; os.makedirs(OUT2, exist_ok=True)
DYN = ["v1", "v6", "vmin", "q6", "qgrow", "vnb", "qdn"]
TEST_M = ("2030-12", "2031-01", "2031-02")


def bestset(p):
    o = np.argsort(-p); cs = np.cumsum(p[o]); kk = np.arange(1, p.size + 1)
    kb = int(np.argmax(cs / (kk + p.sum() - cs))) + 1; s = np.zeros(p.size, bool); s[o[:kb]] = True; return s


PRC = {}


def prior_of(P, day_ok, exclude_month=None):
    """site prior of a world (review 10/13: exclude_month leaves the window's own month out for training worlds)"""
    key = (P.panel, P.split, id(day_ok), exclude_month)        # id: the day filters are lambdas (same __name__)
    if key in PRC:
        return PRC[key]
    truth, obs_q, _ = world_layers(P); ev = events(P, truth, obs_q)
    f = np.zeros(P.n); n = 0
    for d, r in ev.items():
        if day_ok(P, d) and (exclude_month is None or str(P.day[d])[:7] != exclude_month):
            f += dil(truth[r], 2); n += 1
    PRC[key] = (f / max(n, 1)).astype(np.float32)
    return PRC[key]


def build_split(p, split, pr, day_ok):
    P = load(p, split); truth, obs_q, seen = world_layers(P); ev = {d: r for d, r in events(P, truth, obs_q).items() if day_ok(P, d)}
    hide = np.zeros(P.T, bool)
    for r in ev.values():
        hide[r - 5:r + 13] = True
    seen = seen & ~hide[:, None]
    R_ = np.where(seen, P.V / P.vcut[None, :], np.nan); Qc = np.where(seen, P.Q / P.cap[None, :], np.nan); n = P.n
    X, Y = [], []
    for d, r in sorted(ev.items()):
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
        prw = pr(str(P.day[d])[:7]) if callable(pr) else pr
        X.append(np.column_stack([v1, v6, vmin, q6, q6 / np.maximum(q12, 1e-3), nb(v6, np.nanmin), nb(q6, np.nanmax), prw]).astype(np.float32))
        Y.append(y)
    return X, Y


def build(panels=None):
    allday = lambda P, d: True
    real_tr = lambda P, d: str(P.day[d])[:7] not in TEST_M
    real_te = lambda P, d: str(P.day[d])[:7] in TEST_M
    out = {k: [] for k in ("base", "improved", "real", "test")}
    for p in (panels or T2_PANELS):
        t0 = time.time()
        PA6, PB, PT_ = load(p, "twinA6"), load(p, "twinB"), load(p, "train")
        prA_all = prior_of(PA6, allday); prB_all = prior_of(PB, allday); prR = prior_of(PT_, real_tr)
        prA = lambda m, P=PA6: prior_of(P, allday, m); prB = lambda m, P=PB: prior_of(P, allday, m)   # own month out (training worlds)
        prRtr = lambda m, P=PT_: prior_of(P, real_tr, m)
        Rw = [f"twinR{k}" for k in range(48) if os.path.exists(f"C:/Users/user/tfb/work/twin/{p}_twinR{k}.json")
              and json.load(open(f"C:/Users/user/tfb/work/twin/{p}_twinR{k}.json"))["template"] < "2030-12"]
        for sp in ["twinA6"] + Rw:
            out["base"].append((p, sp) + build_split(p, sp, prA if sp == "twinA6" else prA_all, allday))
        for sp in ["twinB"] + [f"twinQ{k}" for k in range(16)]:
            out["improved"].append((p, sp) + build_split(p, sp, prB if sp == "twinB" else prB_all, allday))
        out["real"].append((p, "train") + build_split(p, "train", prRtr, real_tr))
        out["test"].append((p, "train") + build_split(p, "train", prR, real_te))
        print(p, f"{time.time()-t0:.0f}s", {k: sum(len(b[2]) for b in v if b[0] == p) for k, v in out.items()}, flush=True)
    pickle.dump(out, open(OUT2 + f"blocks_{'_'.join(panels) if panels else 'ALL'}.pkl", "wb"))


def fit():
    import lightgbm as lgb
    B = {k: [] for k in ("base", "improved", "real", "test")}
    for f in glob.glob(OUT2 + "blocks_*.pkl"):
        for k, v in pickle.load(open(f, "rb")).items():
            B[k] += v
    cols = DYN + ["prior"]
    def XY(blocks):
        X = np.vstack([x for b in blocks for x in b[2]]); Y = np.concatenate([y for b in blocks for y in b[3]])
        return pd.DataFrame(X, columns=cols), Y
    prm = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, num_threads=16, verbose=-1, seed=0)
    models = {}
    for src in ("base", "improved", "real"):
        X, Y = XY(B[src])
        for m, c in (("M0", DYN), ("M1", cols)):
            models[(src, m)] = (lgb.train(prm, lgb.Dataset(X[c], Y.astype(int)), 400), c)
        print(src, "windows", sum(len(b[2]) for b in B[src]), flush=True)
    rows = []
    for (p, sp, Xs, Ys) in B["test"]:
        for X, y in zip(Xs, Ys):
            D = pd.DataFrame(X, columns=cols); r = dict(panel=p)
            for k, (mdl, c) in models.items():
                s = bestset(mdl.predict(D[c], num_threads=16)); r[f"{k[0]}_{k[1]}"] = (s & y).sum() / (s | y).sum()
            rows.append(r)
    R = pd.DataFrame(rows); ks = [c for c in R.columns if c != "panel"]
    print(f"real test onsets (months 6-8): {len(R)}"); print(R[ks].mean().round(4).to_string())
    print(R.groupby("panel")[ks].mean().round(3).to_string())
    rng = np.random.default_rng(0); u = R.panel.unique(); idx = {x: np.flatnonzero(R.panel.to_numpy() == x) for x in u}
    for a, b in (("improved_M0", "base_M0"), ("improved_M1", "base_M1"), ("real_M1", "improved_M1"), ("real_M0", "improved_M0")):
        d = (R[a] - R[b]).to_numpy(); se = np.std([d[np.concatenate([idx[x] for x in rng.choice(u, len(u))])].mean() for _ in range(500)])
        print(f"{a} - {b}: {d.mean():+.4f} (panel-bootstrap SE {se:.4f})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build(sys.argv[2].split(",") if len(sys.argv) > 2 else None)
    else:
        fit()
