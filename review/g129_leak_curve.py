"""G129 (10/31, ledger #83, strategy P1, the user's request): DIAGNOSTIC ONLY, no submission. How much would the observed
cells after T buy on our path? Organiser ruling 749601 forbids data after T as an input or through fitting, selection or
values in the method: nothing here feeds the method, the constants, a candidate or the finals. Outputs stay in
work/t2/leak_curve/ (never work/subs, src/ or an as-of cache). Held-out days are not read.
M3 v3 DEV onset 100 / ongoing 356 windows, final A's flows (G120 / G121 paths), horizons h = T+5, T+15, T+30, rows r = 0, 2, 5:
  O  overlay: rows <= r set to the truth on the scored cells, the rest final A's selection
  OP overlay + persistence: rows > r take the T+h state where it is observed (ongoing)
  ST a small stack: rows > r re-predicted by LightGBM from logit p, its +-2 link max, the lead, the queue at T-5 and the T+h
     state (queue, v/vf of the link and +-1, +-2, distance to the nearest queue), then best_set, no far gate
  CT the same stack with every state feature blank (the recalibration of the stack alone)
  SD final A's logit + (logit ST - logit CT): only the state's evidence (added after the first run: CT onset -0.23, the
     stack's calibration does not carry over to final A's probabilities). T+25 was added at the same time.
The stacks train on the training months' selector-like windows (month-CV OOF q25_nomem s3/s4, Task 1 blank hidden).
LU (onset, added after the second run: ST / SD onset are not readable, CT -0.23): a logistic update fitted on the training
months, logit y ~ lp + d0 + dn, d0 = max(0, 0.95 - v/vf) of the link at T+h, dn = the same, max over +-1, +-2, applied as
logit p + c0 d0 + c1 dn on final A's probabilities (no drop = final A).
usage: TFB_M3VER=v3 py g129_leak_curve.py [fit | score | onset]"""
import os, sys, time, pickle, importlib.util
from collections import defaultdict
MODE = (sys.argv[1:] or ["fit"])[0]
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd, lightgbm as lgb
TM = time.perf_counter; T00 = TM(); R = r"C:/Users/user/tfb/"; OUT = R + "work/t2/leak_curve/"; os.makedirs(OUT, exist_ok=True)
HS = (5, 15, 25); CONDS = ("queue_onset", "queue_ongoing")
PRM = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, seed=0, verbose=-1)
NST = 13                                                                   # state features at the tail of a row


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def sh(a, o):
    out = np.full(a.shape, np.nan)
    if o > 0:
        out[o:] = a[:-o]
    else:
        out[:o] = a[-o:]
    return out


def state(y_r, e_r, v_r, vf):
    """(n, NST): queue and v/vf at T+h of the link, +-1, +-2 (NaN where not observed), distance to the nearest queue up / down"""
    n = len(y_r); q = np.where(e_r, y_r.astype(float), np.nan)
    with np.errstate(all="ignore"):
        vr = np.where(e_r, v_r / vf, np.nan)
    ix = np.flatnonzero(y_r & e_r); pos = np.arange(n); up = np.full(n, 99.0); dn = np.full(n, 99.0)
    if ix.size:
        i = np.searchsorted(ix, pos, side="right"); up = np.where(i > 0, pos - ix[np.maximum(i - 1, 0)], 99.0)
        j = np.searchsorted(ix, pos, side="left"); dn = np.where(j < ix.size, ix[np.minimum(j, ix.size - 1)] - pos, 99.0)
    return np.column_stack([q, vr] + [sh(q, o) for o in (-2, -1, 1, 2)] + [sh(vr, o) for o in (-2, -1, 1, 2)] + [up, dn, (ix.size > 0) * np.ones(n)])


def rows_x(pb, rr, r, lq, st):
    """feature rows for the probability rows rr: (len(rr) n, 4 + NST)"""
    with np.errstate(all="ignore"):
        lp = np.log(np.clip(pb, 1e-6, 1 - 1e-6)) - np.log1p(-np.clip(pb, 1e-6, 1 - 1e-6))
    out = []
    for j in rr:
        mx = np.nanmax(np.vstack([lp[j]] + [sh(lp[j], o) for o in (-2, -1, 1, 2)]), axis=0)
        out.append(np.column_stack([lp[j], mx, np.full(lp.shape[1], j - r), lq.astype(float), st]))
    return np.vstack(out)


def fit():
    g = module("g119", R + "review/g119_month_postproc.py"); T, G = g.T, g.G; CI = G.CI
    OOF = [pickle.load(open(g.W + f"oof_q25_nomem_s{s}.pkl", "rb")) for s in (3, 4)]; X, Y = defaultdict(list), defaultdict(list)
    ks = [k for k in np.flatnonzero(T.META["is_sel"]) if k in OOF[0] and k in OOF[1]]; print(f"windows {len(ks)} | {TM() - T00:.0f}s", flush=True)
    for p in sorted({str(T.META["panel"][k]) for k in ks}):
        P = g.load(p, "train")
        for k in [k for k in ks if str(T.META["panel"][k]) == p]:
            t = int(T.META["t"][k]); cond = str(T.META["cond"][k]); lab, el = T.LAB[k]; y, e = lab.astype(bool), el.astype(bool)
            if t + 6 >= P.T:
                continue
            p_ = (OOF[0][k] + OOF[1][k]) / 2; lq = np.asarray(g.XN[g.OFFN[k]:g.OFFN[k + 1]][:, CI], float) == 1
            for h in HS + (0,):
                r = h // 5 - 1; rr = [5] if cond == "queue_onset" else list(range(r + 1, 6))
                if h:
                    vis = e[r] & ~P.blank[t + r + 1]; st = state(y[r], vis, P.V[t + r + 1], P.vf)
                else:
                    st = np.full((P.n, NST), np.nan)
                Fm = rows_x(p_, rr, r, lq, st); yy = np.concatenate([y[j] for j in rr]); ee = np.concatenate([e[j] for j in rr])
                X[(cond, h)].append(Fm[ee].astype(np.float32)); Y[(cond, h)].append(yy[ee])
        g.load.cache_clear()
    for key in X:
        Xa = np.vstack(X[key]); Ya = np.concatenate(Y[key]); t0 = TM(); ds = lgb.Dataset(Xa, Ya.astype(int), free_raw_data=True)
        if key[0] == "queue_onset" and key[1]:
            np.savez(OUT + f"onset_rows_{key[1]}.npz", X=Xa, y=Ya)
        mdl = lgb.train(PRM, ds, 300); mdl.save_model(OUT + f"stack_{key[0]}_{key[1]}.txt")
        print(f"{key}: {Xa.shape} pos {Ya.mean():.3f} | fit {TM() - t0:.0f}s", flush=True)
        X[key] = Y[key] = None


def boot(d, cl, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed); ids = pd.factorize(cl)[0]; M = np.bincount(ids, d); N = np.bincount(ids)
    Rm = rng.multinomial(len(M), np.full(len(M), 1.0 / len(M)), size=n_boot); return float(((Rm @ M) / (Rm @ N)).std())


def known(pb, y, e, r):
    q = pb.copy(); q[:r + 1] = np.where(e[:r + 1], y[:r + 1].astype(float), pb[:r + 1]); return q


def stack(mdl, pb, rr, r, lq, st):
    q = pb.copy(); n = pb.shape[1]; pr = mdl.predict(rows_x(pb, rr, r, lq, st))
    for i, j in enumerate(rr):
        q[j] = pr[i * n:(i + 1) * n]
    return q


def delta(mS, mC, pb, rr, r, lq, st, nan0):
    lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6)) - np.log1p(-np.clip(x, 1e-6, 1 - 1e-6))
    q = pb.copy(); n = pb.shape[1]; a = lg(mS.predict(rows_x(pb, rr, r, lq, st))); b = lg(mC.predict(rows_x(pb, rr, -1, lq, nan0)))
    for i, j in enumerate(rr):
        q[j] = 1.0 / (1.0 + np.exp(-(lg(pb[j]) + a[i * n:(i + 1) * n] - b[i * n:(i + 1) * n])))
    return q


def score():
    MD = {(c, h): lgb.Booster(model_file=OUT + f"stack_{c}_{h}.txt") for c in CONDS for h in HS + (0,)}
    # ongoing: G121 path
    g = module("g121", R + "review/g121_ongoing_oracle.py"); G, m3v2, best_set = g.G, g.m3v2, g.best_set; dec = lambda q: best_set(q.ravel(), np.ones(q.size, bool)).reshape(q.shape); ROWS = []
    for w in g.DEV:
        m = G.META[w]; i = G.POS[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); pl = g.FS.lgb_ongoing(Xw); lq = np.asarray(Xw[:, G.CI], float) == 1
        y, e = m["y"].T.astype(bool), m["m"].T.astype(bool); P = m3v2._panel(m["panel"])[0]; t = m["t"]; r_ = defaultdict(float, w=w, cond="queue_ongoing", panel=m["panel"], day=m["day"])
        nan0 = np.full((P.n, NST), np.nan); ST = {h: state(y[h // 5 - 1], e[h // 5 - 1], P.V[t + h // 5], P.vf) for h in HS}
        for k, pr in m3v2.kdist(m["day"]).items():
            p = (0.8 * g.X1[(w, k)] + 0.2 * pl) * g.cone_f(m, k); sel0, pb = g.og_pred(p, m, k); r_["A"] += pr * G.iou(sel0, m)
            r_["CT"] += pr * G.iou(dec(stack(MD[("queue_ongoing", 0)], pb, list(range(6)), -1, lq, nan0)), m)
            for h in HS:
                r = h // 5 - 1; s = sel0.copy(); s[:r + 1] = (y[:r + 1] & e[:r + 1]) | (sel0[:r + 1] & ~e[:r + 1]); r_[f"O{h}"] += pr * G.iou(s, m)
                s2 = s.copy()
                for j in range(r + 1, 6):
                    s2[j] = np.where(e[r], y[r], s[j])
                r_[f"OP{h}"] += pr * G.iou(s2, m)
                kn = known(pb, y, e, r); rr = list(range(r + 1, 6))
                r_[f"ST{h}"] += pr * G.iou(dec(stack(MD[("queue_ongoing", h)], kn, rr, r, lq, ST[h])), m)
                r_[f"SD{h}"] += pr * G.iou(dec(delta(MD[("queue_ongoing", h)], MD[("queue_ongoing", 0)], kn, rr, r, lq, ST[h], nan0)), m)
            r_["O30"] += pr * G.iou(y & e, m)
        ROWS.append(dict(r_))
    print(f"ongoing scored {TM() - T00:.0f}s", flush=True)
    # onset: G120 path
    g2 = module("g120", R + "review/g120_onset_gate.py"); PO = g2.PO
    PN = g2.nn_blend(); phys = g2.lgb.Booster(model_file=g2.W + "physon_base.txt"); ev = PO.month_events("validation", PO.rarity_table())
    for w in g2.DEV:
        m = G.META[w]; i = G.POS[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); ph = phys.predict(PO.feats(Xw)); a = PO.adaptive_a(ev, m["day"]); lq = np.asarray(Xw[:, G.CI], float) == 1
        y, e = m["y"].T.astype(bool), m["m"].T.astype(bool); P = m3v2._panel(m["panel"])[0]; t = m["t"]; r_ = defaultdict(float, w=w, cond="queue_onset", panel=m["panel"], day=m["day"])
        nan0 = np.full((P.n, NST), np.nan); ST = {h: state(y[h // 5 - 1], e[h // 5 - 1], P.V[t + h // 5], P.vf) for h in HS}
        for k, pr in m3v2.kdist(m["day"]).items():
            pb = PN[w].copy(); pb[5] = (1 - a) * pb[5] + a * ph; s, nd = m3v2.seen(m["panel"], t, k, -6, 12)
            if nd:
                pb[-1] = pb[-1] * np.where(G.dil(s, 2), 1.0, 0.25)
            sel0 = dec(pb); io = G.iou(sel0, m); r_["A"] += pr * io; r_["O30"] += pr * G.iou(y & e, m)
            r_["CT"] += pr * G.iou(dec(stack(MD[("queue_onset", 0)], pb, [5], -1, lq, nan0)), m)
            for h in HS:
                r_[f"O{h}"] += pr * io; r_[f"OP{h}"] += pr * io; r_[f"ST{h}"] += pr * G.iou(dec(stack(MD[("queue_onset", h)], pb, [5], h // 5 - 1, lq, ST[h])), m)
                r_[f"SD{h}"] += pr * G.iou(dec(delta(MD[("queue_onset", h)], MD[("queue_onset", 0)], pb, [5], h // 5 - 1, lq, ST[h], nan0)), m)
        ROWS.append(dict(r_))
    D = pd.DataFrame(ROWS); D.to_csv(OUT + "g129_leak_curve.csv", index=False)
    ref = pd.read_csv(G.W + "g108_rectangles.csv").set_index("w")["A"]; d0 = float(np.max(np.abs(D.A - ref.loc[D.w].to_numpy())))
    print(f"scored {len(D)} windows in {TM() - T00:.0f}s | identity h = 0 vs G108 column A max |d| {d0:.2e}")
    if d0 > 1e-9:
        print("IDENTITY FAILED: stop"); sys.exit(1)
    Lo = {c: 1 - D[D.cond == c].A.mean() for c in CONDS}; RX = {"queue_onset": (1 - 0.829) / (1 - 0.864), "queue_ongoing": (1 - 0.881) / (1 - 0.919)}
    for col in ("CT", "O5", "OP5", "ST5", "SD5", "O15", "OP15", "ST15", "SD15", "O25", "OP25", "ST25", "SD25", "O30"):
        s_ = 0.89719; txt = []
        for c in CONDS:
            d = D[D.cond == c]; x = (d[col] - d.A).to_numpy(); se = boot(x, d.panel + "_" + d.day.astype(str)); s_ += 0.15 * RX[c] * x.mean()
            txt.append(f"{c.split('_')[1]} {d[col].mean():.4f} ({x.mean():+.4f}, SE {se:.4f})")
        print(f"{col:5s} " + " | ".join(txt) + f" | S {s_:.4f}")
    for h in HS:
        for c in CONDS:
            d = D[D.cond == c]; x = (d[f"SD{h}"] - d[f"O{h}"]).to_numpy(); print(f"SD{h} - O{h} {c.split('_')[1]}: {x.mean():+.4f} (SE {boot(x, d.panel + '_' + d.day.astype(str)):.4f})")


def drops(F):
    d0 = np.nan_to_num(np.clip(0.95 - F[:, 5], 0, None)); dn = np.nan_to_num(np.nanmax(np.clip(0.95 - F[:, 10:14], 0, None), axis=1)); return d0, dn


def onset():
    from sklearn.linear_model import LogisticRegression
    import warnings; warnings.simplefilter("ignore")
    C = {}
    for h in HS:
        z = np.load(OUT + f"onset_rows_{h}.npz"); F = z["X"]; d0, dn = drops(F); mdl = LogisticRegression(C=1e6, max_iter=500).fit(np.column_stack([F[:, 0], d0, dn]), z["y"].astype(int)); C[h] = mdl.coef_[0]
        print(f"T+{h}: coef lp {C[h][0]:.3f} d0 {C[h][1]:.2f} dn {C[h][2]:.2f} | cells with d0 > 0: {(d0 > 0).mean():.4f}, positive rate there {z['y'][d0 > 0].mean():.3f} vs {z['y'][d0 == 0].mean():.3f}", flush=True)
    g2 = module("g120", R + "review/g120_onset_gate.py"); G, m3v2, PO, best_set = g2.G, g2.m3v2, g2.PO, g2.best_set
    PN = g2.nn_blend(); phys = g2.lgb.Booster(model_file=g2.W + "physon_base.txt"); ev = PO.month_events("validation", PO.rarity_table()); rows = []
    for w in g2.DEV:
        m = G.META[w]; i = G.POS[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); ph = phys.predict(PO.feats(Xw)); a = PO.adaptive_a(ev, m["day"]); lq = np.asarray(Xw[:, G.CI], float) == 1
        y, e = m["y"].T.astype(bool), m["m"].T.astype(bool); P = m3v2._panel(m["panel"])[0]; t = m["t"]; r_ = defaultdict(float, w=w, panel=m["panel"], day=m["day"])
        for k, pr in m3v2.kdist(m["day"]).items():
            pb = PN[w].copy(); pb[5] = (1 - a) * pb[5] + a * ph; s, nd = m3v2.seen(m["panel"], t, k, -6, 12)
            if nd:
                pb[-1] = pb[-1] * np.where(G.dil(s, 2), 1.0, 0.25)
            r_["A"] += pr * G.iou(best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape), m)
            for h in HS:
                r = h // 5 - 1; F = rows_x(pb, [5], r, lq, state(y[r], e[r], P.V[t + h // 5], P.vf)); d0, dn = drops(F); q = pb.copy()
                q[5] = 1.0 / (1.0 + np.exp(-(F[:, 0] + C[h][1] * d0 + C[h][2] * dn))); r_[f"LU{h}"] += pr * G.iou(best_set(q.ravel(), np.ones(q.size, bool)).reshape(q.shape), m)
        rows.append(dict(r_))
    D = pd.DataFrame(rows); D.to_csv(OUT + "g129_onset_lu.csv", index=False); ref = pd.read_csv(G.W + "g108_rectangles.csv").set_index("w")["A"]
    print(f"identity A vs G108 {float(np.max(np.abs(D.A - ref.loc[D.w].to_numpy()))):.2e} | {TM() - T00:.0f}s")
    cl = D.panel + "_" + D.day.astype(str)
    for h in HS:
        x = (D[f"LU{h}"] - D.A).to_numpy(); print(f"LU T+{h}: onset {D[f'LU{h}'].mean():.4f} ({x.mean():+.4f}, SE {boot(x, cl):.4f}, windows changed {(np.abs(x) > 1e-9).sum()})")


if __name__ == "__main__":
    {"fit": fit, "score": score, "onset": onset}[MODE]()
