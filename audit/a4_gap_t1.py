"""Audit A4: the Task 2 horizon+buffer gap (T+5..T+90, all links blank) and Task 1.
On the official TRAIN windows (truth available):
 (1) masked-layer quality flags inside the gap == unmasked-layer flags? (=> horizon eligibility is known in val/private)
 (2) how hard are Task 1 targets inside the gap: RMSE of a two-sided linear-in-time interpolation (row T-1 / row T+19)
     vs the same interpolation at ordinary targets, speed and flow per lane.
 (3) are Task 1 targets present inside the train gaps (template)."""
import sys, glob, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from common import REL

out = []
for p in T2_PANELS:
    P = load(p, "train"); ti = {s: i for i, s in enumerate(P.times)}; li = {l: i for i, l in enumerate(P.links)}
    wi = pd.read_csv(REL / "task2" / p / "train" / "window_index.csv")
    for r in wi.itertuples():
        t = ti[r.forecast_origin]; day = r.date
        f = glob.glob(str(REL / "corridors" / p / "train" / "mainline_states_masked" / "*" / f"*{day.replace('-', '_')}.parquet"))[0]
        m = pd.read_parquet(f, columns=["timestamp", "link_id", "speed_kmh", "pct_observed", "is_score_eligible"])
        rr = m.timestamp.map(ti).to_numpy(); cc = m.link_id.astype(str).map(li).to_numpy()
        E_m = np.zeros((P.T, P.n), bool); E_m[rr, cc] = m.is_score_eligible.astype(bool).to_numpy()
        S_m = np.full((P.T, P.n), np.nan); S_m[rr, cc] = m.speed_kmh.to_numpy()
        g = slice(t + 1, t + 19)
        flag_eq = float((E_m[g] == P.E[g]).mean()); gap_fin = float(np.isfinite(S_m[g]).mean())
        # two-sided linear interpolation across the gap, rows t-1 (last released history row) and t+19
        a, b = t - 1, t + 19
        w = (np.arange(1, 19) + 1) / 20.0                           # position of rows t+1..t+18 between a and b
        ok = P.E[a] & P.E[b] & np.isfinite(P.V[a]) & np.isfinite(P.V[b])
        vi = P.V[a][None, :] * (1 - w[:, None]) + P.V[b][None, :] * w[:, None]
        qi = (P.Q[a][None, :] * (1 - w[:, None]) + P.Q[b][None, :] * w[:, None]) / P.lanes[None, :]
        tv, tq = P.V[g], P.Q[g] / P.lanes[None, :]
        sel = P.E[g] & np.isfinite(tv) & ok[None, :]
        tg = P.target[g] & sel
        cong = (tv < 0.6 * P.vf[None, :]) & sel
        out.append(dict(panel=p, cond=r.condition, flag_eq=flag_eq, gap_fin=gap_fin, n=int(sel.sum()), n_tgt=int(tg.sum()), cong_share=float(cong.sum() / max(sel.sum(), 1)),
                        se_v=float(((vi - tv)[sel] ** 2).sum()), se_q=float(((qi - tq)[sel] ** 2).sum())))
    load.cache_clear()
d = pd.DataFrame(out)
print("masked-layer eligibility flags inside the gap equal to the unmasked layer: min", d.flag_eq.min().round(4), "mean", d.flag_eq.mean().round(4))
print("finite share inside the train gaps (masked layer):", d.gap_fin.describe().round(3).to_dict())
print("Task 1 targets inside the official train gaps:", int(d.n_tgt.sum()), "of", int(d.n.sum()), "eligible gap cells")
g = d.groupby("cond")[["n", "se_v", "se_q"]].sum()
g["rmse_v_gap_lininterp"] = np.sqrt(g.se_v / g.n); g["rmse_qlane_gap_lininterp"] = np.sqrt(g.se_q / g.n)
print(g[["n", "rmse_v_gap_lininterp", "rmse_qlane_gap_lininterp"]].round(2))
print("congested share of gap cells:", d.groupby("cond").cong_share.mean().round(3).to_dict())
