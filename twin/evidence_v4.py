"""Third instrument for the onset evidence rule B1 (10/14; pending since 10/13: +0.0095, 1.8 sigma in two twin families).
Question: when the release blanks hide all same-month evidence, should the production rule still scale every link by B
(a uniform shrink), or skip (B1)? Setting, all in memory on the CPU:
  model  : per panel, LightGBM WITH link position trained on the real months 0-5 onset windows (a site-memory model, as
           the production network memorises sites from the corridor itself; twin/sim2real_diag.py)
  months : per panel 12 twin v4 worlds (templates months 0-5, the clean C-world demand redraw: the sites re-rank)
  blanks : scored windows = the first onset window on each of days 1-5 (mornings) + up to 5 ongoing windows on days
           1-5; rows T+1..T+18 of every scored window are blanked on all links (as the release does)
  rules  : none | cur (production: links without a queue in slots T-30..T+60 on <= 3 earlier days, +-2 links, x B at T+30)
           | skip (B1: as cur, but no scaling when no link has evidence) ; B = 0.25
IoU at T+30 (expected-IoU set) of the scored onset windows; days 2-5 (evidence can exist) and day 1 reported apart;
cluster bootstrap over panel x world. Pre-registered (10/13): B1 goes to the LB if skip - cur >= +2 SE here.
usage: python evidence_v4.py [panels=ALL T2] [worlds=12]"""
import sys, zlib, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/twin")
src = open(r"C:/Users/user/tfb/twin/sim2real_v4.py", encoding="utf-8").read()
ns = {"__name__": "lib"}; exec(src.split("data = {\"REAL\"")[0], ns)                 # window / feature functions only
from cachelib import load, T2_PANELS
from onset_shape3_lib import raw_matrices
from ctm import junction_ratios
from t2_model import best_set
import gen, gen_v4

H, F, B = 12, 6, 0.25
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=0)
ns["COND"] = "onset"; ns["CAPW"][0] = 250; ns["NOPOS"] = False


def scored_windows(V, E, vcut, n_days, rng):
    """first onset window per morning on days 1-5 + up to 5 ongoing windows on days 1-5 (selector conditions)"""
    ns["VCUT"] = vcut; ns["COND"] = "onset"; ons, Qm = ns["windows"](V, E, n_days)
    first = {}
    for t in ons:
        d = t // 288
        if d < 5 and d not in first:
            first[d] = t
    ns["COND"] = "ongoing"; ns["CAPW"][0] = 10 ** 9; ogs, _ = ns["windows"](V, E, n_days); ns["COND"] = "onset"; ns["CAPW"][0] = 250
    ogs = [t for t in ogs if t // 288 < 5]
    ogs = sorted(rng.choice(ogs, min(5, len(ogs)), replace=False).tolist()) if ogs else []
    return first, ogs, Qm


def rule(pb_last, V, t, vcut, kind):
    if kind == "none":
        return pb_last
    q = (V <= vcut[None, :]) & np.isfinite(V); seen = np.zeros(V.shape[1], bool); nd = 0
    for back in (1, 2, 3):
        dd = t - back * 288
        if dd - 6 >= 0:
            seen |= q[dd - 6:dd + 12].any(axis=0); nd += 1
    if not nd or (kind == "skip" and not seen.any()):
        return pb_last
    ev = seen.copy()
    for o in (1, 2):
        ev[o:] |= seen[:-o]; ev[:-o] |= seen[o:]
    return pb_last * np.where(ev, 1.0, B)


rows = []
pans = T2_PANELS if len(sys.argv) < 2 or sys.argv[1] == "ALL" else sys.argv[1].split(",")
NW = int(sys.argv[2]) if len(sys.argv) > 2 else 12
for p in pans:
    PT = load(p, "train"); D = PT.T // 288; PT.V, PT.Q, PT.E = raw_matrices(PT); ns["VCUT"] = PT.vcut
    months = sorted(set(str(x)[:7] for x in PT.day[:D])); keep = [d for d in range(D) if str(PT.day[d])[:7] in months[:6]]
    vf, cap = PT.vf.astype(float), PT.cap.astype(float); r = ns["month_rows"](keep)
    xs, ys, es, _ = ns["collect"](PT.V[r], PT.Q[r], PT.E[r], len(keep), vf, cap, "REAL", p)
    X = np.vstack([x[e] for x, e in zip(xs, es)]); y = np.concatenate([y_[e] for y_, e in zip(ys, es)]).astype(np.float32)
    mdl = lgb.train(PARAMS, lgb.Dataset(X, y), 300)
    S = ns["structure_k"](PT, keep); jr = np.isin(np.arange(PT.T) // 288, keep)
    jr_k = lambda P_, rows=None, _jr=jr: junction_ratios(P_, _jr); gen.junction_ratios = gen_v4.junction_ratios = jr_k
    base_lr = np.log(jr_k(PT)).mean(axis=(0, 1)); rng = np.random.default_rng([11, zlib.crc32(p.encode())])
    for k in range(NW):
        tm = months[k % 6]; days = [d for d in keep if str(PT.day[d]).startswith(tm)][:30]
        s_rat, s_up = ns["RATS"][rng.integers(3)], ns["UPS"][rng.integers(3)]
        cm = np.exp(gen.SIG_CAP * rng.standard_normal(PT.n)); rat = s_rat * np.abs(base_lr) * rng.standard_normal(PT.n - 1)
        upm = float(np.exp(s_up * rng.standard_normal())); rr = ns["month_rows"](days); nanm = ~np.isfinite(PT.V[rr]); Et = PT.E[rr]
        Vt, Qt, _ = gen_v4.simulate_v4(PT, days, rng, S, rat, upm, cm)
        Vt = Vt * (1 + gen.NOISE_V * rng.standard_normal(Vt.shape)); Qt = Qt * (1 + gen.NOISE_Q * rng.standard_normal(Qt.shape))
        Vt[nanm] = np.nan; Qt[nanm] = np.nan
        first, ogs, Qm = scored_windows(Vt, Et, PT.vcut, len(days), rng)
        Vb, Qb = Vt.copy(), Qt.copy()
        for t in list(first.values()) + ogs:                  # the release blanks T+1..T+18 of every scored window
            Vb[t + 1:t + 19] = np.nan; Qb[t + 1:t + 19] = np.nan
        for d, t in first.items():
            x = ns["feats"](Vb, Qb, t, vf, cap, PT.n); pb = mdl.predict(x, num_threads=8)
            yl, el = Qm[t + F], Et[t + F]
            row = dict(panel=p, world=k, day=d + 1, cl=f"{p}_{k}")
            for kind in ("none", "cur", "skip"):
                q = rule(pb.copy(), Vb, t, PT.vcut, kind); s = best_set(np.where(el, q, 0.0), el)
                u = (s | yl) & el; row[kind] = 1.0 if not u.any() else float((s & yl & el).sum() / u.sum())
            rows.append(row)
    print(p, "scored onset windows", sum(1 for r_ in rows if r_["panel"] == p), flush=True)
    load.cache_clear()
Dd = pd.DataFrame(rows); Dd.to_csv(r"C:/Users/user/tfb/work/evidence_v4.csv", index=False); rng = np.random.default_rng(0)
for nm, g in (("all days", Dd), ("days 2-5", Dd[Dd.day >= 2]), ("day 1", Dd[Dd.day == 1])):
    u = g.cl.unique(); idx = {x: np.flatnonzero(g.cl.to_numpy() == x) for x in u}; msg = []
    for a, b in (("cur", "none"), ("skip", "cur"), ("skip", "none")):
        d_ = (g[a] - g[b]).to_numpy()
        se = np.std([d_[np.concatenate([idx[x] for x in rng.choice(u, len(u))])].mean() for _ in range(1000)])
        msg.append(f"{a}-{b} {d_.mean():+.4f} (SE {se:.4f})")
    print(f"{nm}: windows {len(g)}, none {g.none.mean():.4f}, cur {g.cur.mean():.4f}, skip {g.skip.mean():.4f} | " + " | ".join(msg), flush=True)
