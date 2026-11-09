"""Boundary specialist -> validation/private Task 1 states.  Cells = Task 1 targets outside the Task 2 blanks whose
released neighbourhood (+-2 links, +-2 steps) holds both congested and free observations.  New value =
w * specialist + (1 - w) * current, then the per-link speed ceiling (as t1_clip), FD 0.75 on those cells (as t1_fd),
and the density clip v >= q/k_jam on every row last.  Features use ADAPT statics (as the T1 inference).
Simulation (audit d2): specialist RMSE below the best T1 on the selected cells in 9/10 panels.

usage: python t1_bnd_apply.py <model tag: bnd|bnd2> <w> <in.parquet> <out.parquet> [<in2> <out2> ...]
"""
import os, sys, time
os.environ["TFB_T1_ADAPT"] = "1"
import numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
from cachelib import load, PANELS, T2_PANELS
from common import REL, WORK
from t1_features3 import grid_features3, rows3, base_of3
from t1_bnd import released, selector, build

W = WORK + "/t1/"
tag, w = sys.argv[1], float(sys.argv[2]); pairs = list(zip(sys.argv[3::2], sys.argv[4::2]))
M = {ch: lgb.Booster(model_file=W + f"{tag}_{ch}_final.txt") for ch in ("speed", "flow")}
t0 = time.time(); fills = []; CAP = {}
for p in PANELS:
    T = load(p, "train"); capv = np.nanmax(np.where(np.isfinite(T.V), T.V, np.nan), axis=0)
    for split in ("validation", "private"):
        S = load(p, split); capv = np.fmax(capv, np.nanmax(np.where(np.isfinite(S.V), S.V, np.nan), axis=0))
    CAP[p] = capv; load.cache_clear()
    for split in ("validation", "private"):
        P = load(p, split); ti = {s: i for i, s in enumerate(P.times)}
        gap = np.zeros((P.T, P.n), bool)
        if p in T2_PANELS:
            for r in pd.read_csv(REL / "task2" / p / split / "window_index.csv").itertuples():
                t = ti[r.forecast_origin]; gap[t + 1:t + 19] = True
        F, _, _, _ = grid_features3(P); bv, bq = base_of3(F)
        V, Q = released(P, P.V, P.Q); sel, cc, ff = selector(P, V)
        rr, cc_ = np.nonzero(P.target & ~gap & sel)
        X, _ = build(P, F, V, Q, rr, cc_, cc, ff, rows3); del F
        vb = np.clip((bv[rr, cc_] + M["speed"].predict(X, num_threads=8)) * P.vf[cc_], 3.0, 1.15 * P.vf[cc_])
        qb = np.clip((bq[rr, cc_] + M["flow"].predict(X, num_threads=8)) * P.lanes[cc_], 60.0, None)
        fills.append(pd.DataFrame({"panel": p, "_ts": P.times[rr].astype(str), "link_id": P.links[cc_].astype(str), "_v": vb, "_q": qb,
                                   "_vf": P.vf[cc_], "_kj": P.kj[cc_], "_w": P.w[cc_], "_cap": capv[cc_]}))
        print(p, split, "boundary targets", len(rr), f"{time.time()-t0:.0f}s", flush=True)
        load.cache_clear()
fill = pd.concat(fills, ignore_index=True); assert not fill.duplicated(["panel", "_ts", "link_id"]).any()
fill.to_parquet(W + f"{tag}_fill.parquet", index=False)
KJ = {}
for p in PANELS:
    P = load(p, "validation"); KJ[p] = dict(zip(P.links.astype(str), P.kj)); load.cache_clear()
for src, dst in pairs:
    st = pd.read_parquet(src); st["_ts"] = st.timestamp.astype(str); st["link_id"] = st.link_id.astype(str)
    j = st[["panel", "_ts", "link_id"]].reset_index().merge(fill, on=["panel", "_ts", "link_id"], how="inner")
    ix = j["index"].values
    v = w * j._v.values + (1 - w) * st.speed_kmh.values[ix]; q = w * j._q.values + (1 - w) * st.flow_vph.values[ix]
    v = np.maximum(np.minimum(v, j._cap.values), q / j._kj.values)                                   # ceiling (t1_clip)
    cong = v < 0.6 * j._vf.values; k = np.maximum(j._kj.values - q / j._w.values, 1e-3)             # FD 0.75 (t1_fd)
    v = np.where(cong, 0.25 * v + 0.75 * q / k, v)
    st.loc[ix, "speed_kmh"] = v; st.loc[ix, "flow_vph"] = q
    kj = np.array([KJ[p][l] for p, l in zip(st.panel.values, st.link_id.values)])
    st["speed_kmh"] = np.maximum(st.speed_kmh.values, st.flow_vph.values / kj)                      # density clip last
    assert np.isfinite(st.speed_kmh).all() and np.isfinite(st.flow_vph).all()
    st.drop(columns="_ts").to_parquet(dst, index=False)
    print(src, "->", dst, "replaced", len(j), "of", len(fill), f"{time.time()-t0:.0f}s", flush=True)
