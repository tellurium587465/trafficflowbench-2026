"""Audit A3: facts about the validation/private release that the pipeline relies on.
 - window_history.parquet vs our reconstruction (cache rows T-12..T-1)
 - observed share at row T, inside the horizon+buffer gap, and Task 1 targets inside the gap
 - finite-but-ineligible cells (dropped by PanelInputs' good = pct>=75 & eligible)
 - day-of-month of the scored windows (month-prior relevance)"""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from common import REL, PANELS

rows = []
for split in ("validation", "private"):
    for p in T2_PANELS:
        P = load(p, split); ti = {s: i for i, s in enumerate(P.times)}; li = {l: i for i, l in enumerate(P.links)}
        wi = pd.read_csv(REL / "task2" / p / split / "window_index.csv")
        h = pd.read_parquet(REL / "task2" / p / split / "window_history.parquet")
        gap_rows = np.zeros(P.T, bool)
        stats = dict(hist_cells=0, hist_fin=0, hist_fin_cache_nan=0, cache_fin_hist_nan=0, maxdiff_v=0.0, maxdiff_q=0.0, elig_mismatch=0,
                     hist_fin_inelig=0, rowT_elig=0, rowT_fin=0, gap_elig=0, gap_fin=0, gap_pct_known=0, gap_cells=0)
        for r in wi.itertuples():
            t = ti[r.forecast_origin]
            gap_rows[t + 1:t + 19] = True
            hw = h[h.window_id == r.window_id]
            rr = hw.timestamp.map(ti).to_numpy(); cc = hw.link_id.map(li).to_numpy()
            assert rr.min() == t - 12 and rr.max() == t - 1, (r.window_id, rr.min() - t, rr.max() - t)
            hv = hw.speed_kmh.to_numpy(); hq = hw.flow_vph.to_numpy(); he = hw.is_score_eligible.astype(bool).to_numpy()
            cv = P.V[rr, cc]; cq = P.Q[rr, cc]; ce = P.E[rr, cc]
            stats["hist_cells"] += len(hw); stats["hist_fin"] += int(np.isfinite(hv).sum())
            stats["hist_fin_cache_nan"] += int((np.isfinite(hv) & ~np.isfinite(cv)).sum())
            stats["cache_fin_hist_nan"] += int((~np.isfinite(hv) & np.isfinite(cv)).sum())
            both = np.isfinite(hv) & np.isfinite(cv)
            if both.any():
                stats["maxdiff_v"] = max(stats["maxdiff_v"], float(np.abs(hv[both] - cv[both]).max()))
                stats["maxdiff_q"] = max(stats["maxdiff_q"], float(np.abs(hq[both] - cq[both]).max()))
            stats["elig_mismatch"] += int((he != ce).sum())
            stats["hist_fin_inelig"] += int((np.isfinite(hv) & ~he).sum())
            stats["rowT_elig"] += int(P.E[t].sum()); stats["rowT_fin"] += int((np.isfinite(P.V[t]) & P.E[t]).sum())
            g = slice(t + 1, t + 19)
            stats["gap_cells"] += P.V[g].size; stats["gap_elig"] += int(P.E[g].sum()); stats["gap_fin"] += int(np.isfinite(P.V[g]).sum())
            stats["gap_pct_known"] += int((P.P[g] > 0).sum())
        tgt = P.target; n_t = int(tgt.sum()); n_gap = int(tgt[gap_rows].sum())
        fin = np.isfinite(P.V); inel_fin = int((fin & ~P.E).sum()); el_fin = int((fin & P.E).sum())
        days = sorted({int(s[8:10]) for s in wi.date.astype(str)})
        on_days = [int(s[8:10]) for s, c in zip(wi.date.astype(str), wi.condition) if c == "queue_onset"]
        og_days = [int(s[8:10]) for s, c in zip(wi.date.astype(str), wi.condition) if c == "queue_ongoing"]
        rows.append(dict(split=split, panel=p, **stats, targets=n_t, targets_in_gap=n_gap, gap_row_share=float(gap_rows.mean()),
                         finite_ineligible=inel_fin, finite_eligible=el_fin, onset_days=on_days, ongoing_days=og_days))
        load.cache_clear()
d = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(d[["split", "panel", "hist_cells", "hist_fin", "hist_fin_cache_nan", "cache_fin_hist_nan", "maxdiff_v", "maxdiff_q", "elig_mismatch", "hist_fin_inelig"]].to_string())
d["rowT_obs_share"] = d.rowT_fin / d.rowT_elig
d["gap_fin_share"] = d.gap_fin / d.gap_cells; d["gap_elig_share"] = d.gap_elig / d.gap_cells
d["tgt_gap_share"] = d.targets_in_gap / d.targets
d["inelig_fin_share"] = d.finite_ineligible / (d.finite_ineligible + d.finite_eligible)
print(d[["split", "panel", "rowT_obs_share", "gap_fin_share", "gap_elig_share", "gap_pct_known", "targets", "targets_in_gap", "tgt_gap_share", "gap_row_share", "inelig_fin_share"]].round(4).to_string())
print(d[["split", "panel", "onset_days", "ongoing_days"]].to_string())
