"""What starts a congestion episode? (raw C: data only; the generator study of 10/14). For each panel, morning episode
starts (first congested row of an 8-connected episode, 06:00-12:00, head = most downstream congested link) per site
(head link). For each frequent site h (>= 8 starts): the 30-min moving-average flow at links h..h+2 (max over the 3
links) in every morning row before any congestion at the site; the hazard P(start in the next row | flow bin) from
all such rows of all mornings (starts vs at-risk rows). A step-like hazard = deterministic threshold (predictable from
flows); a smooth one = random breakdown (limited predictability). Also the start-time spread per site.
usage: python episode_trigger.py [panels]"""
import sys, numpy as np, pandas as pd, warnings
from scipy import ndimage
warnings.simplefilter("ignore")
sys.argv += [] if len(sys.argv) > 1 else ["ALL"]
_a = list(sys.argv); sys.argv = ["x", "NONE"]
src = open(r"C:/Users/user/tfb/twin/episodes_raw.py", encoding="utf-8").read()
exec(src.split("rows, summ = [], []")[0].replace('V = np.full((len(P.times), P.n), np.nan, np.float32); E = np.zeros((len(P.times), P.n), bool)',
     'V = np.full((len(P.times), P.n), np.nan, np.float32); Qm = np.full((len(P.times), P.n), np.nan, np.float32); E = np.zeros((len(P.times), P.n), bool)')
     .replace('columns=["timestamp", "link_id", "speed_kmh", "is_score_eligible"]', 'columns=["timestamp", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"]')
     .replace('agg(v=("speed_kmh", "mean"), e=("is_score_eligible", "max"))', 'agg(v=("speed_kmh", "mean"), q=("flow_vph", "mean"), e=("is_score_eligible", "max"))')
     .replace('V[r, c] = g.v.to_numpy()[ok]; E[r, c] = g.e.to_numpy()[ok] > 0', 'V[r, c] = g.v.to_numpy()[ok]; Qm[r, c] = g.q.to_numpy()[ok]; E[r, c] = g.e.to_numpy()[ok] > 0')
     .replace('    return P, V, E', '    return P, V, E, Qm'))
sys.argv = _a
pans = T2_PANELS if sys.argv[1] == "ALL" else sys.argv[1].split(",")
out = []
for p in pans:
    P, V, E, Q = matrices(p); D = P.T // 288
    cong = (V <= P.vcut[None, :]) & np.isfinite(V)
    lab, n = ndimage.label(cong, structure=np.ones((3, 3))); sl = ndimage.find_objects(lab)
    starts = {}
    for i, s_ in enumerate(sl, 1):
        m = lab[s_] == i
        if m.sum() < 6:
            continue
        r0 = s_[0].start
        if not (72 <= r0 % 288 < 144):
            continue
        head = s_[1].start + int(np.flatnonzero(m[0]).max())
        starts.setdefault(head, []).append(r0)
    # 30-min moving average of flow (rows), max over h..h+2
    c = np.nancumsum(np.nan_to_num(Q), axis=0); k = np.cumsum(np.isfinite(Q), axis=0)
    c = np.vstack([np.zeros((1, P.n)), c]); k = np.vstack([np.zeros((1, P.n)), k])
    ma = (c[6:] - c[:-6]) / np.maximum(k[6:] - k[:-6], 1); ma[(k[6:] - k[:-6]) < 4] = np.nan       # ma[t] = mean of rows t-5..t
    ma = np.vstack([np.full((5, P.n), np.nan), ma])
    for h, rs in starts.items():
        if len(rs) < 8:
            continue
        f = np.nanmax(np.stack([ma[:, j] for j in range(h, min(h + 3, P.n))]), axis=0)
        start_rows = set(rs); atrisk, ev = [], []
        for d in range(D):
            lo, hi = d * 288 + 72, d * 288 + 144
            cong_site = cong[lo:hi, max(0, h - 4):h + 1].any(axis=1)
            stop = lo + (int(np.argmax(cong_site)) if cong_site.any() else hi - lo)
            for t in range(lo, stop + 1 if stop < hi else hi):
                if np.isfinite(f[t - 1]):
                    atrisk.append(f[t - 1]); ev.append(t in start_rows)
        a = np.array(atrisk); e = np.array(ev)
        if e.sum() < 5:
            continue
        fs = a[e]; lo_, hi_ = np.nanquantile(fs, 0.05), np.nanquantile(fs, 0.95)
        bins = np.quantile(a[a > np.nanquantile(fs, 0.01) * 0.95], np.linspace(0, 1, 9)) if (a > np.nanquantile(fs, 0.01) * 0.95).sum() > 50 else None
        hz = []
        if bins is not None:
            for b0, b1 in zip(bins[:-1], bins[1:]):
                mm = (a >= b0) & (a < b1); hz.append(round(float(e[mm].mean()), 3) if mm.sum() else np.nan)
        out.append(dict(panel=p, site=h, starts=int(e.sum()), flow_start_q05_q95=(round(float(lo_)), round(float(hi_))),
                        width_rel=round(float((hi_ - lo_) / np.nanmedian(fs)), 3), start_time_sd_min=round(float(np.std([r % 288 for r in rs]) * 5), 1),
                        hazard_by_flow_octile=hz))
        print(out[-1], flush=True)
    load.cache_clear(); del V, E, Q, cong, lab, ma, c, k
R = pd.DataFrame(out); R.to_csv(r"C:/Users/user/tfb/work/twin_episode_trigger.csv", index=False)
print(R[["panel", "site", "starts", "width_rel", "start_time_sd_min"]].to_string(index=False))
