"""Location-agnostic onset model (strategy §16, 10/22; G67-G73): binary LightGBM on the within-window percentiles of the
non-memory NN inputs, trained on the onset windows with a T+30 queue, label = queued at T+30 on eligible cells.
  "base" model: the train months (nn_data_afdem)                 -> official validation (March) windows and M3
  "m"    model: the train months + March (nn_data_march, 748789) -> official private (April) windows only
Writes work/t2/probs_physon.pkl = {window_id: (6, n)} for every official onset window (T+30 row only, zeros elsewhere),
validation windows from "base", private from "m", plus probs_m3__physon.pkl (M3, "base"). Mixed into the onset blend at
T+30 as p = (1 - a) p_NN + a p_phys (a = 0.3 pre-declared, t2_build_variant.py TFB_T2_PHYS_A), before the evidence rule.
Identity: the "base" model reproduces G69's (review/g69_phys_blend2.py, same rows, labels, parameters) on M3.
usage: python t2_phys_onset.py            (models + probabilities)
       python t2_phys_onset.py adaptive   (strategy §16 (d): causal per-window a -> physon_a.pkl, physon_a_m3.pkl)
       python t2_phys_onset.py apply [out] (10/23 repro: the saved models -> probs_<out>.pkl, default physon)
       python t2_phys_onset.py rarity      (10/23: the train-month rarity table -> work/t2/rarity_table.pkl, shipped)"""
import sys, pickle, numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import rankdata
from common import NN, REL, WORK
from cachelib import T2_PANELS

W = WORK + "/t2/"
NAMES = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]
DROP = {"prior_on", "prior_og", "pos", "vf", "capl", "lanes", "length", "w", "tod", "dow", "arr5", "cond"}
COLS = [i for i, n in enumerate(NAMES) if n not in DROP]
PRM = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=8, seed=7)


def feats(Xw):
    X = Xw[:, COLS].astype(float); P = np.full_like(X, 0.5)
    for c in range(X.shape[1]):
        v = X[:, c]; ok = np.isfinite(v)
        if ok.sum() >= 2 and np.nanstd(v) > 0:
            P[ok, c] = (rankdata(v[ok]) - 0.5) / ok.sum()
    return P


def rows(d):
    """(features, labels) of the eligible T+30 cells of every onset window with a T+30 queue in train-layout dir d"""
    X = np.load(f"{NN}/{d}/X.npy", mmap_mode="r"); Y = np.load(f"{NN}/{d}/Y.npy", mmap_mode="r"); E = np.load(f"{NN}/{d}/EL.npy", mmap_mode="r")
    off = np.load(f"{NN}/{d}/off.npy"); cond = np.load(f"{NN}/{d}/cond.npy", allow_pickle=True); Xs, ys = [], []
    for k in np.flatnonzero(cond == "queue_onset"):
        a, b = off[k], off[k + 1]; y = np.asarray(Y[a:b, 5]).astype(bool); e = np.asarray(E[a:b, 5]).astype(bool)
        if (y & e).any():
            Xs.append(feats(np.asarray(X[a:b]))[e]); ys.append(y[e].astype(int))
    print(d, len(Xs), "onset windows", flush=True)
    return Xs, ys


def predict(mdl, test_dir, wanted=None):
    X = np.load(f"{NN}/{test_dir}/X.npy", mmap_mode="r"); off = np.load(f"{NN}/{test_dir}/off.npy")
    wid = [str(w) for w in np.load(f"{NN}/{test_dir}/window_id.npy", allow_pickle=True)]; cond = np.load(f"{NN}/{test_dir}/cond.npy", allow_pickle=True)
    out = {}
    for i, w in enumerate(wid):
        if cond[i] != "queue_onset" or (wanted is not None and w not in wanted):
            continue
        p = np.zeros((6, off[i + 1] - off[i]), np.float32); p[5] = mdl.predict(feats(np.asarray(X[off[i]:off[i + 1]]))); out[w] = p
    return out


A0, R0, M0 = 0.3, 0.30, 20.0     # strategy §16 (d): a = A0 x rate / R0, rate shrunk to March's R0 with M0 pseudo-events


def rarity_table():
    """train-month link frequency at T+30 per panel x weekend x 2-hour slot (G54/G76): {key: f (n links)}. 10/23: a static
    train-months artifact like a model's weights -> read from work/t2/rarity_table.pkl when it exists (written by
    `python t2_phys_onset.py rarity` from the train-month windows' labels; TFB_RARITY_REBUILD=1 recomputes)"""
    import os
    if os.path.exists(W + "rarity_table.pkl") and os.environ.get("TFB_RARITY_REBUILD", "0") != "1":
        return pickle.load(open(W + "rarity_table.pkl", "rb"))
    os.environ.setdefault("TFB_T2CACHE", "2_noT_af"); argv = sys.argv; sys.argv = [argv[0], "0"]
    from t2_eval_oof import T
    sys.argv = argv
    cnt, den = {}, {}
    for k in range(T.NWIN):
        if T.META["is_sel"][k] and str(T.META["cond"][k]) == "queue_onset":
            lab, el = T.LAB[k]; key = (str(T.META["panel"][k]), pd.Timestamp(str(T.META["date"][k])[:10]).dayofweek >= 5, (int(T.META["t"][k]) % 288) // 24)
            cnt[key] = cnt.get(key, 0) + (lab[5] & el[5]).astype(float); den[key] = den.get(key, 0) + 1
    return {k: cnt[k] / den[k] for k in cnt}


def month_events(split, F_):
    """onset events of the released month (G76 shape) -> list of (day0, new_site): day0 = 0-based day of the event row"""
    from cachelib import load
    ev = []
    for p in T2_PANELS:
        P = load(p, split); q = np.isfinite(P.V) & (P.V <= P.vcut[None, :]); anyq = q.any(axis=1); last = -10 ** 9
        for t in range(17, P.T):
            if anyq[t] and not anyq[t - 17:t].any() and t - last >= 24:
                last = t; o = t - 6; key = (p, pd.Timestamp(P.times[o][:10]).dayofweek >= 5, (o % 288) // 24)
                f = F_.get(key, np.zeros(P.n)); ev.append((t // 288, bool((f[np.flatnonzero(q[t])] <= 0.1).all())))
    return ev


def adaptive_a(ev, day0):
    """a for a window on 0-based day day0: only events on completed earlier days (day < day0) are counted"""
    e = [nw for d, nw in ev if d < day0]
    return float(np.clip(A0 * ((sum(e) + R0 * M0) / (len(e) + M0)) / R0, 0.0, 0.5))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "rarity":   # 10/23: the static table for the repro package
    import os
    os.environ["TFB_RARITY_REBUILD"] = "1"; F_ = rarity_table(); pickle.dump(F_, open(W + "rarity_table.pkl", "wb")); print("rarity table", len(F_), "keys")
    sys.exit()

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "adaptive":
    F_ = rarity_table(); out = {}
    for sp in ("validation", "private"):
        ev = month_events(sp, F_); tmap = {}
        if sp == "validation":
            ev_val = ev
        for p in T2_PANELS:
            P = __import__("cachelib").load(p, sp); ti = {s_: i for i, s_ in enumerate(P.times)}
            for r in pd.read_csv(REL / "task2" / p / sp / "window_index.csv").itertuples():
                if r.condition == "queue_onset":
                    tmap[r.window_id] = out[r.window_id] = adaptive_a(ev, ti[r.forecast_origin] // 288)
        print(sp, "events", len(ev), "new-site share (whole month, diagnostic) %.2f" % np.mean([nw for _, nw in ev]),
              "| a over its onset windows: min %.3f median %.3f max %.3f" % tuple(np.percentile(list(tmap.values()), [0, 50, 100])), flush=True)
    import pickle as _pk
    _pk.dump(out, open(W + "physon_a.pkl", "wb"))
    import os as _os
    if _os.path.exists(W + "m3_meta.pkl"):                   # the March instrument (not part of the repro package)
        m3a = {m["window_id"]: adaptive_a(ev_val, m["day"]) for m in _pk.load(open(W + "m3_meta.pkl", "rb"))["meta"] if m["cond"] == "queue_onset"}
        _pk.dump(m3a, open(W + "physon_a_m3.pkl", "wb")); print("M3 onset windows: a median %.3f, range %.3f-%.3f" % (np.median(list(m3a.values())), min(m3a.values()), max(m3a.values())))
    sys.exit()

if __name__ == "__main__":
    APPLY = sys.argv[1:2] == ["apply"]                          # 10/23 repro: the saved models -> probs_physon.pkl only
    if APPLY:
        base, mm = lgb.Booster(model_file=W + "physon_base.txt"), lgb.Booster(model_file=W + "physon_m.txt")
    else:
        Xa, ya = rows("nn_data_afdem")
        base = lgb.train(PRM, lgb.Dataset(np.vstack(Xa), np.concatenate(ya)), 400); base.save_model(W + "physon_base.txt")
        Xm, ym = rows("nn_data_march")
        mm = lgb.train(PRM, lgb.Dataset(np.vstack(Xa + Xm), np.concatenate(ya + ym)), 400); mm.save_model(W + "physon_m.txt")
    split = {}
    for sp in ("validation", "private"):
        for p in T2_PANELS:
            split.update({w: sp for w in pd.read_csv(REL / "task2" / p / sp / "window_index.csv").window_id})
    out = predict(base, "nn_test_afdem", {w for w, s in split.items() if s == "validation"})
    out.update(predict(mm, "nn_test_afdem", {w for w, s in split.items() if s == "private"}))
    pickle.dump(out, open(W + (f"probs_{sys.argv[2]}.pkl" if APPLY and len(sys.argv) > 2 else "probs_physon.pkl"), "wb"))
    if APPLY:
        print("official onset windows", len(out), "(saved models)"); sys.exit()
    m3 = predict(base, "nn_test_m3"); pickle.dump(m3, open(W + "probs_m3__physon.pkl", "wb"))
    ref = pickle.load(open(W + "probs_m3__g69phys_onset.pkl", "rb"))
    d = max(float(np.abs(m3[w][5] - ref[w]).max()) for w in ref)
    print(f"official onset windows {len(out)} (validation {sum(split[w] == 'validation' for w in out)}, private {sum(split[w] == 'private' for w in out)}); "
          f"identity vs G69 on M3 dev onset windows: max |diff| {d:.2e}")
