"""Train the final blank-cell model on every train month and fill the Task 1 targets inside the Task 2 blanks of
validation/private (rows T+5..T+90 of every scored window) in the given state parquets; density clip v >= q/k_jam last.

usage: python t1_gapml_apply.py <rounds_speed> <rounds_flow> <lr> <in1.parquet> <out1.parquet> [<in2.parquet> <out2.parquet> ...]
Feb harness (audit b1/b2): blank-cell RMSE v 8.9 -> 5.5 km/h, q/lane 102 -> 63 vs the linear interpolation.
"""
import sys, time, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
from cachelib import load, T2_PANELS
from common import REL, WORK
import t2_demand
from t2_demand import profile
from t2_arrival import _dmin
from t1_gapml import view_train, view_test, gap_features, G, PARAMS

W = WORK + "/"
rs, rq, lr = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]); pairs = list(zip(sys.argv[4::2], sys.argv[5::2]))
t0 = time.time()
# ---- training rows: every eval/selector-like train window, all months
import os                                                       # 10/24 repro: only the window list is read -> a 5.7 MB extract
Z = np.load(W + "t2/gapml_windows.npz" if os.path.exists(W + "t2/gapml_windows.npz") else W + "t2/featcache2_noT.npz", allow_pickle=True)
m = Z["is_eval"] | Z["is_sel"]
wins = {}
for p, t, c in zip(Z["panel"][m], Z["t"][m], Z["cond"][m]):
    wins.setdefault((str(p), int(t)), str(c))
del Z
rng = np.random.default_rng(11); TR = []; names = None
for p in T2_PANELS:
    P = load(p, "train"); x = np.concatenate([[0.0], np.cumsum(P.length[:-1])]); dm = _dmin(P, x)
    for t, c in sorted((t, c) for (pp, t), c in wins.items() if pp == p):
        if t + G + 31 > P.T:
            continue
        V, Q = view_train(P, t)
        X, names, bv, bq, kk, jj = gap_features(P, V, Q, t, c, profile(P, str(P.times[t])[:7]), dm, x)
        rr = t + kk; tv = P.V[rr, jj].astype(float); tq = (P.Q[rr, jj] / P.lanes[jj]).astype(float)
        ok = P.E[rr, jj] & np.isfinite(tv) & np.isfinite(tq) & np.isfinite(bv) & np.isfinite(bq)
        f = {nm: X[:, i] for i, nm in enumerate(names)}
        near = (np.minimum.reduce([f["dup_pre"], f["ddn_pre"], f["dup_post"], f["ddn_post"]]) <= 3.0) | (np.nan_to_num(f["v_lin"], nan=1) < 0.85)
        pk = np.where(near, 0.5, 0.08); keep = ok & (rng.random(len(ok)) < pk)
        TR.append((X[keep], tv[keep] - bv[keep], tq[keep] - bq[keep], 1.0 / pk[keep]))
    load.cache_clear(); t2_demand._PROF.clear()
Xtr = np.vstack([a[0] for a in TR]); wtr = np.concatenate([a[3] for a in TR])
prm = dict(PARAMS, learning_rate=lr); models = {}
for ch, yi, nr in (("speed", 1, rs), ("flow", 2, rq)):
    models[ch] = lgb.train(prm, lgb.Dataset(Xtr, np.concatenate([a[yi] for a in TR]), weight=wtr, feature_name=names), nr)
    models[ch].save_model(W + f"t1/gapml_{ch}_final.txt"); print(ch, "trained", Xtr.shape, f"{time.time()-t0:.0f}s", flush=True)
del Xtr, TR
# ---- fill table for validation/private
fills = []
for split in ("validation", "private"):
    for p in T2_PANELS:
        P = load(p, split); ti = {s: i for i, s in enumerate(P.times)}
        x = np.concatenate([[0.0], np.cumsum(P.length[:-1])]); dm = _dmin(P, x); prof = profile(load(p, "train"))
        n0 = 0
        for r in pd.read_csv(REL / "task2" / p / split / "window_index.csv").itertuples():
            t = ti[r.forecast_origin]
            V, Q = view_test(P, t)
            X, _, bv, bq, kk, jj = gap_features(P, V, Q, t, r.condition, prof, dm, x)
            rr = t + kk; inb = rr < P.T
            tg = np.zeros(len(rr), bool); tg[inb] = P.target[rr[inb], jj[inb]]
            tg &= np.isfinite(bv) & np.isfinite(bq)
            if not tg.any():
                continue
            v = np.clip(bv[tg] + models["speed"].predict(X[tg], num_threads=8), 3.0, 1.15 * P.vf[jj[tg]])
            q = np.clip((bq[tg] + models["flow"].predict(X[tg], num_threads=8)) * P.lanes[jj[tg]], 60.0, None)
            fills.append(pd.DataFrame({"panel": p, "_ts": P.times[rr[tg]].astype(str), "link_id": P.links[jj[tg]].astype(str), "_v": v, "_q": q}))
            n0 += int(tg.sum())
        print(split, p, "blank targets filled", n0, flush=True)
        load.cache_clear(); t2_demand._PROF.clear()
fill = pd.concat(fills, ignore_index=True)
assert not fill.duplicated(["panel", "_ts", "link_id"]).any()
fill.to_parquet(W + "t1/gapml_fill.parquet", index=False)
KJ = {p: dict(zip(load(p, "validation").links.astype(str), load(p, "validation").kj)) for p in T2_PANELS + ["D12_I405_N", "D12_I405_S"]}
load.cache_clear()
for src, dst in pairs:
    st = pd.read_parquet(src); st["_ts"] = st.timestamp.astype(str); st["link_id"] = st.link_id.astype(str)
    j = st[["panel", "_ts", "link_id"]].reset_index().merge(fill, on=["panel", "_ts", "link_id"], how="inner")
    st.loc[j["index"].values, "speed_kmh"] = j._v.values; st.loc[j["index"].values, "flow_vph"] = j._q.values
    kj = np.array([KJ[p][l] for p, l in zip(st.panel.values, st.link_id.values)])
    st["speed_kmh"] = np.maximum(st.speed_kmh.values, st.flow_vph.values / kj)          # density clip, applied last
    assert np.isfinite(st.speed_kmh).all() and np.isfinite(st.flow_vph).all()
    st.drop(columns="_ts").to_parquet(dst, index=False)
    print(src, "->", dst, "replaced", len(j), "of", len(fill), "fill rows;", len(st), "rows", f"{time.time()-t0:.0f}s", flush=True)
