"""Go/no-go for onset training on synthetic months (A2) before the GPU phase (10/14): does the twin v4 (block cells,
block-level daily thresholds, congested-branch recording) teach an onset mapping that holds on REAL months, where the
v2 twin failed (F57: -0.024 locally)? CPU only, everything in memory (raw C: data; nothing written but a small CSV).
Identification (blocks, sites, thresholds, discharge, junction ratios, capacity multipliers) on the real months 0-5;
synthetic worlds copy a month 0-5 template with the clean C-world randomization (gen_clean / gen_v4); the test is the
real months 6-8. Onset windows = the selector's onset condition (no queued eligible cell in T-12..T+5, one at T+6,
same day, history coverage >= 0.7), mornings ((T+6) % 288 < 144). Features from the history rows T-12..T-1 only, no
link identity: speed and flow levels / trends at the link and +-1..2 neighbours, flow / capacity, relative position.
LightGBM (8 threads); decoder = expected-IoU set at T+30 (as production). Train sets: REAL (months 0-5), V2 (twin v2),
V4 (twin v4), REAL+V4. Score: mean IoU on the real test windows, cluster bootstrap over panel x day.
usage: python sim2real_v4.py [panels=ALL T2 panels] [worlds per panel=12] [cond=onset|ongoing]
cond=ongoing (10/14): the selector's ongoing condition (history T-12..T-1 holds >= 2 queued obs with >= 2 on one link,
a queue in T+1..T+6, persistence IoU <= 0.9, coverage >= 0.7), all six steps scored, step and history-queue features
(queued at T-1, queued rows in the history, distance to the nearest queued link at T-1); train sets REAL, REAL+V2,
REAL+V4 (augmentation is how the twin is used for ongoing)."""
import sys, time, zlib, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/twin")
from cachelib import load, T2_PANELS
from ctm import junction_ratios
from onset_shape3_lib import raw_matrices, blocks
from t2_model import best_set
import gen, gen2, gen_v4
from twin3_check import block_sites

H, F = 12, 6
RATS, UPS = gen_v4.RATS, gen_v4.UPS


def structure_k(PT, keep):
    rows = np.isin(np.arange(PT.T) // 288, keep); bid = blocks(PT.V, PT, rows); bs = block_sites(PT, keep, bid)
    dists = gen2.site_dists(PT, keep); cm = gen2.base_cap_mult(PT, keep, dists)
    cg = (PT.V[rows] <= PT.vcut[None, :]) & np.isfinite(PT.Q[rows]); Cd = np.full(bid.max() + 1, np.nan)
    for j in range(bid.max() + 1):
        m = cg[:, bid == j]
        if m.sum() >= 30:
            Cd[j] = np.median(PT.Q[rows][:, bid == j][m])
    return bid, bs, cm, Cd


COND = sys.argv[3] if len(sys.argv) > 3 else "onset"


def windows(V, E, n_days):
    """origins t of the selector's condition COND on a (n_days*288, n) month; returns list of t"""
    Qm = (V <= VCUT[None, :]) & np.isfinite(V) & E; obs = np.isfinite(V) & E
    any_q = Qm.any(axis=1); qrow = np.concatenate([[0], np.cumsum(any_q)]); ocnt = np.concatenate([[0], np.cumsum(obs.sum(axis=1))])
    out = []
    for t in range(H, n_days * 288 - F):
        if (t - H) // 288 != (t + F) // 288 or (COND == "onset" and (t + F) % 288 >= 144):
            continue
        if (ocnt[t] - ocnt[t - H]) / (H * V.shape[1]) < 0.7:
            continue
        if COND == "onset":
            if qrow[t + F] - qrow[t - H] == 0 and any_q[t + F]:
                out.append(t)
        else:
            hq = Qm[t - H:t].sum(axis=0)
            if hq.sum() >= 2 and hq.max() >= 2 and qrow[t + F + 1] - qrow[t + 1] > 0:
                fut = Qm[t + 1:t + F + 1]; el = E[t + 1:t + F + 1]; last = np.broadcast_to(Qm[t - 1], fut.shape)
                u = ((last | fut) & el).sum(); iou_p = 1.0 if u == 0 else (last & fut & el).sum() / u
                if iou_p <= 0.9:
                    out.append(t)
    if COND == "ongoing":                                    # thin to one origin per 6 rows (overlapping windows)
        out = out[::6]
        cap_w = CAPW[0]
        if len(out) > cap_w:                                 # memory cap (4 GB rule): a fixed random subset
            out = sorted(np.random.default_rng(len(out)).choice(out, cap_w, replace=False).tolist())
    return out, Qm


CAPW = [250]                                                 # windows per collect call (real months / a twin world)


def feats(V, Q, t, vf, cap, n):
    h = slice(t - H, t)
    v = V[h] / vf[None, :]; q = Q[h] / cap[None, :]
    v1 = np.nanmean(v[-1:], axis=0); v3 = np.nanmean(v[-3:], axis=0); vmin = np.nanmin(v, axis=0)
    q1 = np.nanmean(q[-2:], axis=0); q6 = np.nanmean(q[-6:], axis=0); qg = q1 - np.nanmean(q[:6], axis=0)
    sh = lambda a, k: np.concatenate([np.full(k, np.nan), a[:-k]]) if k > 0 else np.concatenate([a[-k:], np.full(-k, np.nan)])
    cols = [v1, v3, vmin, q1, q6, qg, sh(v1, 1), sh(v1, -1), sh(v1, 2), sh(v1, -2), sh(q1, -1) - q1, sh(q1, 1) - q1]
    if not NOPOS:                                            # 10/14: production NN zeroes position (ZERO=pos+...)
        cols.append(np.arange(n) / n)
    return np.column_stack(cols).astype(np.float32)


import os
NOPOS = os.environ.get("NOPOS", "0") == "1"


def collect(V, Q, E, n_days, vf, cap, tag, cl_prefix):
    ts, Qm = windows(V, E, n_days); Xs, Ys, Es, meta = [], [], [], []
    for t in ts:
        X = feats(V, Q, t, vf, cap, V.shape[1])
        if COND == "onset":
            Xs.append(X); Ys.append(Qm[t + F]); Es.append(E[t + F])
        else:
            n = V.shape[1]; last = Qm[t - 1].astype(np.float32); cnt = Qm[t - H:t].sum(axis=0).astype(np.float32)
            ql = np.flatnonzero(Qm[t - 1]); dist = (np.abs(np.arange(n)[:, None] - ql[None, :]).min(axis=1) if len(ql) else np.full(n, 99)).astype(np.float32)
            Xs.append(np.vstack([np.column_stack([X, last, cnt, np.minimum(dist, 20), np.full(n, s_, np.float32)]) for s_ in range(F)]))
            Ys.append(Qm[t + 1:t + F + 1].ravel()); Es.append(E[t + 1:t + F + 1].ravel())
        meta.append((tag, f"{cl_prefix}{t // 288}"))
    return Xs, Ys, Es, meta


def month_rows(days):
    return np.concatenate([np.arange(d * 288, (d + 1) * 288) for d in days])


data = {"REAL": [], "V2": [], "V4": [], "TEST": []}
pans = T2_PANELS if len(sys.argv) < 2 or sys.argv[1] == "ALL" else sys.argv[1].split(",")
NW = int(sys.argv[2]) if len(sys.argv) > 2 else 12
t0 = time.time()
for p in pans:
    PT = load(p, "train"); D = PT.T // 288; PT.V, PT.Q, PT.E = raw_matrices(PT); VCUT = PT.vcut
    months = sorted(set(str(x)[:7] for x in PT.day[:D])); keep = [d for d in range(D) if str(PT.day[d])[:7] in months[:6]]
    test = [d for d in range(D) if d not in set(keep)]
    vf, cap = PT.vf.astype(float), PT.cap.astype(float)
    CAPW[0] = 250
    for tag, days in (("REAL", keep), ("TEST", test)):
        r = month_rows(days); X, Y, E_, M = collect(PT.V[r], PT.Q[r], PT.E[r], len(days), vf, cap, tag, p + tag)
        data[tag].append((X, Y, E_, M))
    S = structure_k(PT, keep); jr = np.isin(np.arange(PT.T) // 288, keep)
    jr_k = lambda P_, rows=None, _jr=jr: junction_ratios(P_, _jr)          # identification on months 0-5 only
    gen.junction_ratios = gen_v4.junction_ratios = jr_k
    base_lr = np.log(jr_k(PT)).mean(axis=(0, 1))
    rng = np.random.default_rng([7, zlib.crc32(p.encode())])
    for k in range(NW):
        tm = months[k % 6]; days = [d for d in keep if str(PT.day[d]).startswith(tm)][:30]
        s_rat, s_up = RATS[rng.integers(3)], UPS[rng.integers(3)]
        cap_mult = np.exp(gen.SIG_CAP * rng.standard_normal(PT.n))
        rat = s_rat * np.abs(base_lr) * rng.standard_normal(PT.n - 1); upm = float(np.exp(s_up * rng.standard_normal()))
        r = month_rows(days); nanm = ~np.isfinite(PT.V[r]); Et = PT.E[r]
        CAPW[0] = 25                                         # 12 worlds x 25 = 300 twin windows per panel and variant
        for tag in ("V2", "V4"):
            if tag == "V4":
                Vt, Qt, _ = gen_v4.simulate_v4(PT, days, rng, S, rat, upm, cap_mult)
            else:
                Vt, Qt, _ = gen.simulate(PT, days, rng, cap_mult, np.ones(PT.n), rat, upm)
            Vt = Vt * (1 + gen.NOISE_V * rng.standard_normal(Vt.shape)); Qt = Qt * (1 + gen.NOISE_Q * rng.standard_normal(Qt.shape))
            Vt[nanm] = np.nan; Qt[nanm] = np.nan
            X, Y, E_, M = collect(Vt, Qt, Et, len(days), vf, cap, tag, f"{p}{tag}{k}_")
            data[tag].append((X, Y, E_, M))
    print(p, {k: sum(len(x[0]) for x in v) for k, v in data.items()}, f"{time.time() - t0:.0f}s", flush=True)
    load.cache_clear()


def stack(tag):
    X, Y, E_ = [], [], []
    for xs, ys, es, _ in data[tag]:
        for x, y, e in zip(xs, ys, es):
            X.append(x[e]); Y.append(y[e])
    return np.vstack(X), np.concatenate(Y).astype(np.float32)


PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=0)
models = {}
SETS = (("REAL", ["REAL"]), ("V2", ["V2"]), ("V4", ["V4"]), ("REAL+V4", ["REAL", "V4"]), ("REAL+V4d", ["REAL", "V4"])) if COND == "onset" else     (("REAL", ["REAL"]), ("REAL+V2", ["REAL", "V2"]), ("REAL+V4", ["REAL", "V4"]), ("REAL+V4d", ["REAL", "V4"]))
DOM = {}                                                     # "d" = domain indicator (1 = twin) as a feature, 0 at test
for name, tags in SETS:
    parts = [stack(t) for t in tags]
    if name.endswith("d"):
        parts = [(np.column_stack([a, np.full(len(a), float(t != "REAL"), np.float32)]), b) for (a, b), t in zip(parts, tags)]
    DOM[name] = name.endswith("d")
    X = np.vstack([a for a, _ in parts]); y = np.concatenate([b for _, b in parts])
    models[name] = lgb.train(PARAMS, lgb.Dataset(X, y), 300)
    print(name, "train cells", len(y), "positive share", round(float(y.mean()), 4), f"{time.time() - t0:.0f}s", flush=True)
# 10/14: separate ranking from calibration - Platt (a, b) per model fitted on the REAL training windows (months 0-5,
# legitimate), then the within-window AUC (ranking only) and the IoU after recalibration
from scipy.optimize import minimize
from scipy.stats import rankdata
XR, YR = stack("REAL"); PLATT = {}
addd = lambda name, x: np.column_stack([x, np.zeros(len(x), np.float32)]) if DOM[name] else x
for name, mdl in models.items():
    z = np.log(np.clip(mdl.predict(addd(name, XR), num_threads=8), 1e-6, 1 - 1e-6)); z = z - np.log1p(-np.exp(z))
    A = np.column_stack([np.ones(len(z)), z])
    f = lambda w: (np.mean(np.logaddexp(0, A @ w) - YR * (A @ w)) + 1e-7 * (w @ w), A.T @ (1 / (1 + np.exp(-(A @ w))) - YR) / len(YR) + 2e-7 * w)
    PLATT[name] = minimize(f, np.array([0.0, 1.0]), jac=True, method="L-BFGS-B").x
print("Platt on REAL train windows:", {k: np.round(v, 3).tolist() for k, v in PLATT.items()}, flush=True)
rows = []
for xs, ys, es, ms in data["TEST"]:
    for x, y, e, m in zip(xs, ys, es, ms):
        row = dict(cl=m[1])
        for name, mdl in models.items():
            p_ = mdl.predict(addd(name, x), num_threads=8); s = best_set(np.where(e, p_, 0.0), e)
            u = (s | y) & e; row[name] = 1.0 if not u.any() else float((s & y & e).sum() / u.sum())
            lz = np.log(np.clip(p_, 1e-6, 1 - 1e-6) / (1 - np.clip(p_, 1e-6, 1 - 1e-6))); w = PLATT[name]
            pc = 1 / (1 + np.exp(-(w[0] + w[1] * lz))); s = best_set(np.where(e, pc, 0.0), e)
            u = (s | y) & e; row[name + "_cal"] = 1.0 if not u.any() else float((s & y & e).sum() / u.sum())
            yy, pp = y[e], p_[e]
            if 0 < yy.sum() < len(yy):
                r_ = rankdata(pp); row[name + "_auc"] = float((r_[yy].sum() - yy.sum() * (yy.sum() + 1) / 2) / (yy.sum() * (len(yy) - yy.sum())))
        rows.append(row)
D_ = pd.DataFrame(rows); D_.to_csv(time.strftime(rf"C:/Users/user/tfb/work/sim2real_v4_{COND}{'_nopos' if NOPOS else ''}_%Y%m%d_%H%M.csv"), index=False)
rng = np.random.default_rng(0); u = D_.cl.unique(); idx = {x: np.flatnonzero(D_.cl.to_numpy() == x) for x in u}
print(f"{COND} nopos={NOPOS}: real test windows {len(D_)}:", {k: round(float(D_[k].mean()), 4) for k in models})
print("  after Platt on REAL train windows:", {k: round(float(D_[k + "_cal"].mean()), 4) for k in models})
print("  within-window AUC (ranking only):", {k: round(float(D_[k + "_auc"].mean()), 4) for k in models if k + "_auc" in D_})
for a, b in ((("V4", "V2"), ("V4", "REAL"), ("REAL+V4", "REAL"), ("REAL+V4d", "REAL"), ("V2", "REAL")) if COND == "onset" else
             (("REAL+V4", "REAL+V2"), ("REAL+V4", "REAL"), ("REAL+V2", "REAL"), ("REAL+V4d", "REAL"))):
    d = (D_[a] - D_[b]).to_numpy()
    se = np.std([d[np.concatenate([idx[x] for x in rng.choice(u, len(u))])].mean() for _ in range(1000)])
    print(f"  {a} - {b}: {d.mean():+.4f} (SE {se:.4f})", flush=True)
