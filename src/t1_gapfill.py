"""Task 1 targets inside the Task 2 blanks (rows T+5..T+90 of every scored window, all links blank in
validation/private, never seen in training) -> per-link time interpolation between the last released value
before the blank (rows T-60..T) and the first one after it (rows T+95..T+150). Then the density clip k <= k_jam.
Feb simulation (audit a5b): S_state +0.012, physics proxy +0.02 vs the model's own prediction in the blank.

usage: python t1_gapfill.py <state_in.parquet> <state_out.parquet>
"""
import sys, numpy as np, pandas as pd
from cachelib import load, T2_PANELS
from common import REL

src, dst = sys.argv[1], sys.argv[2]
st = pd.read_parquet(src)
st["_ts"] = st.timestamp.astype(str)
changed = 0
for split in ("validation", "private"):
    for p in T2_PANELS:
        P = load(p, split); ti = {s: i for i, s in enumerate(P.times)}
        ok = np.isfinite(P.V) & np.isfinite(P.Q)
        Vf = np.full((P.T, P.n), np.nan); Qf = np.full((P.T, P.n), np.nan)
        for r in pd.read_csv(REL / "task2" / p / split / "window_index.csv").itertuples():
            t = ti[r.forecast_origin]
            lo, hi = max(t - 12, 0), min(t + 31, P.T)
            pre = np.arange(t, lo - 1, -1); post = np.arange(t + 19, hi)
            okp = ok[pre]; okn = ok[post]
            has_p = okp.any(axis=0); has_n = okn.any(axis=0)
            a = pre[np.argmax(okp, axis=0)]; b = post[np.argmax(okn, axis=0)]
            cols = np.flatnonzero(has_p & has_n)
            for i in range(t + 1, min(t + 19, P.T)):
                c = cols[P.target[i, cols]]
                if not len(c):
                    continue
                w = (i - a[c]) / (b[c] - a[c])
                Vf[i, c] = P.V[a[c], c] * (1 - w) + P.V[b[c], c] * w
                Qf[i, c] = P.Q[a[c], c] * (1 - w) + P.Q[b[c], c] * w
        r_, c_ = np.nonzero(np.isfinite(Vf))
        if not len(r_):
            continue
        fill = pd.DataFrame({"panel": p, "_ts": P.times[r_].astype(str), "link_id": P.links[c_],
                             "_v": Vf[r_, c_], "_q": Qf[r_, c_],
                             "_kj": P.kj[c_]})
        m = st.panel.eq(p)
        sub = st.loc[m, ["_ts", "link_id"]].reset_index().merge(fill.drop(columns="panel"), on=["_ts", "link_id"], how="inner")
        v = np.maximum(sub._v.values, sub._q.values / sub._kj.values)          # density clip
        st.loc[sub["index"].values, "speed_kmh"] = v
        st.loc[sub["index"].values, "flow_vph"] = sub._q.values
        changed += len(sub)
        print(p, split, "filled", len(sub), flush=True)
st.drop(columns="_ts").to_parquet(dst, index=False)
print("total filled rows", changed, "of", len(st))
