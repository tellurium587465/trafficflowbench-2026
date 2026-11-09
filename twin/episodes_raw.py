"""Real congestion episodes on all panels from the RAW released train data on C: (the caches live on the E: HDD, which
the RSNA job needs; only the link order is taken from the cache, a few KB with the lazy loader). Per panel:
  onset shape: v/vcut at rows r-3..r+2 around the first morning queue row (median), share of two-run first rows and
               whether the link between the runs is score-ineligible;
  episodes (8-connected congested cells, >= 6 cells): duration, span, first-row extent, extent trajectory, speed level
               (median v/vf) and its spread, the level by head link (most frequent level per head).
Writes work/twin/episodes_raw.csv (one row per episode) for the episode-type twin.
usage: python episodes_raw.py [panels=ALL T2 panels]"""
import sys, glob, numpy as np, pandas as pd, warnings
from scipy import ndimage
warnings.simplefilter("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from common import REL


def matrices(p):
    P = load(p, "train"); li = {l: i for i, l in enumerate(P.links)}; ti = {t: i for i, t in enumerate(P.times)}
    V = np.full((len(P.times), P.n), np.nan, np.float32); E = np.zeros((len(P.times), P.n), bool)
    for f in sorted(glob.glob(str(REL / "corridors" / p / "train" / "mainline_states" / "**" / "*.parquet"), recursive=True)):
        d = pd.read_parquet(f, columns=["timestamp", "link_id", "speed_kmh", "is_score_eligible"])
        g = d.groupby(["timestamp", "link_id"], sort=False).agg(v=("speed_kmh", "mean"), e=("is_score_eligible", "max")).reset_index()
        r = g.timestamp.map(ti).to_numpy(); c = g.link_id.map(li).to_numpy(); ok = ~(pd.isna(r) | pd.isna(c))
        r, c = r[ok].astype(int), c[ok].astype(int)
        V[r, c] = g.v.to_numpy()[ok]; E[r, c] = g.e.to_numpy()[ok] > 0
    return P, V, E


rows, summ = [], []
for p in (sys.argv[1].split(",") if len(sys.argv) > 1 else T2_PANELS):
    P, V, E = matrices(p); D = P.T // 288
    vr = V / P.vf[None, :]; R = V / P.vcut[None, :]
    cong = (V <= P.vcut[None, :]) & np.isfinite(V)
    truth = cong & E
    prof, two, gap_inel = [], 0, []
    for d in range(D):
        for s in range(72, 144):
            r = d * 288 + s
            if truth[r].any():
                idx = np.flatnonzero(truth[r]); prof.append(np.nanmedian(R[r - 3:r + 3][:, idx], axis=1))
                runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
                if len(runs) >= 2:
                    two += 1; g0, g1 = runs[0][-1] + 1, runs[1][0]
                    gap_inel.append(float(np.mean(~E[r, g0:g1])))
                break
    lab, n = ndimage.label(cong, structure=np.ones((3, 3))); sl = ndimage.find_objects(lab)
    for i, s_ in enumerate(sl, 1):
        m = lab[s_] == i
        if m.sum() < 6:
            continue
        vals = vr[s_][m]; ext = m.sum(1)
        rows.append(dict(panel=p, start=s_[0].start % 288, dur=s_[0].stop - s_[0].start, span=s_[1].stop - s_[1].start, first=int(ext[0]),
                         head=s_[1].stop - 1, level=float(np.round(np.nanmedian(vals), 2)),
                         spread=float(np.nanquantile(vals, 0.9) - np.nanquantile(vals, 0.1)), traj=",".join(map(str, ext[:13].tolist()))))
    Ep = pd.DataFrame([r for r in rows if r["panel"] == p])
    summ.append(dict(panel=p, onset_profile=np.round(np.nanmedian(np.array(prof), axis=0), 2).tolist(), two_run_share=round(two / max(len(prof), 1), 2),
                     gap_ineligible=round(float(np.mean(gap_inel)), 2) if gap_inel else np.nan, episodes=len(Ep),
                     dur_med=float(Ep.dur.median()), span_med=float(Ep.span.median()), first_med=float(Ep["first"].median()),
                     spread_med=round(float(Ep.spread.median()), 3), top_levels=Ep["level"].value_counts().head(4).to_dict()))
    print(summ[-1], flush=True)
    load.cache_clear(); del V, E, vr, R, cong, truth, lab
pd.DataFrame(rows).to_csv(r"C:/Users/user/tfb/work/twin_episodes_raw.csv", index=False)
print(pd.DataFrame(summ).drop(columns=["top_levels", "onset_profile"]).to_string(index=False))
