"""M3 statistics (PLAN_20261022 §2.4, items 3-4): production-configuration weights, paired differences with a
panel x day cluster bootstrap, and P(A > B) over redrawn 80-window sets (40 onset + 40 ongoing, the size of one month's
LB). Weights: condition 50:50, panels equal within a condition, weekday 82.5 / weekend 17.5 within a panel (the private
month's mix; the k of each window is already drawn from the official distribution). March 2031 starts on a Saturday.
usage: from m3_stats import weights, summary"""
import numpy as np, pandas as pd

MARCH_FIRST_DOW = 5                                             # 2031-03-01 is a Saturday (synthetic calendar)


def weights(D):
    """D needs columns cond, panel, day -> weight per row (sums to 1 per condition)"""
    wk = ((MARCH_FIRST_DOW + D["day"]) % 7) < 5
    w = np.zeros(len(D))
    for cond in D.cond.unique():
        mc = (D.cond == cond).to_numpy(); pans = D.loc[mc, "panel"].unique()
        for p in pans:
            mp = mc & (D.panel == p).to_numpy()
            for flag, share in ((True, 0.825), (False, 0.175)):
                mm = mp & (wk.to_numpy() == flag)
                if mm.any():
                    other = (mp & (wk.to_numpy() != flag)).any()
                    w[mm] = (share if other else 1.0) / mm.sum() / len(pans)
    return w


def summary(D, a, b, n_boot=1000, n_draw=2000, seed=0):
    """paired difference b - a: raw mean per condition, weighted mean per condition and T2 (= mean of the two
    conditions), cluster-bootstrap SE (panel x day) of the weighted T2 difference, and P(T2 diff > 0) over redrawn
    40 + 40 window sets drawn with the production weights."""
    rng = np.random.default_rng(seed); D = D.copy(); D["d"] = (D[b] - D[a]).to_numpy(); D["w"] = weights(D)
    out = {}
    for cond in ("queue_onset", "queue_ongoing"):
        g = D[D.cond == cond]
        if len(g):
            out[cond] = dict(raw=float(g.d.mean()), weighted=float(np.sum(g.d * g.w) / g.w.sum()), n=len(g))
    conds = [c for c in ("queue_onset", "queue_ongoing") if c in out]
    t2 = float(np.mean([out[c]["weighted"] for c in conds]))
    cl = (D.panel + "_" + D.day.astype(str)).to_numpy(); u = np.unique(cl); idx = {x: np.flatnonzero(cl == x) for x in u}
    bs = []
    for _ in range(n_boot):
        S = D.iloc[np.concatenate([idx[x] for x in rng.choice(u, len(u))])]
        bs.append(np.mean([np.sum(S[S.cond == c].d * S[S.cond == c].w) / max(S[S.cond == c].w.sum(), 1e-12) for c in conds]))
    pos = []
    for _ in range(n_draw):
        v = []
        for c in conds:
            g = D[D.cond == c]; pr = g.w.to_numpy() / g.w.sum()
            v.append(g.d.to_numpy()[rng.choice(len(g), 40, p=pr)].mean())
        pos.append(np.mean(v) > 0)
    return dict(per_cond=out, t2_weighted=t2, se=float(np.std(bs)), p_a_lt_b=float(np.mean(pos)))


def fmt(name, s):
    pc = " ".join(f"{c.split('_')[1]} raw {v['raw']:+.4f} w {v['weighted']:+.4f} (n {v['n']})" for c, v in s["per_cond"].items())
    return f"{name}: T2(weighted) {s['t2_weighted']:+.4f} (SE {s['se']:.4f}, {s['t2_weighted'] / max(s['se'], 1e-9):+.1f} SE), P(B>A in 80 windows) {s['p_a_lt_b']:.2f} | {pc}"
