"""Twin fidelity repair for site choice (10/13). fidelity_site.py: world A replays pick the wrong regular bottleneck on
4 of 8 panels and are too regular (dominance 0.92 vs real 0.70). Two causes in the CTM:
  (cap) cell capacities come from fd_parameters (q0.999 of observed flow), while the real breakdown flows at the
        regular sites are 0.42-0.97 of it (G34);
  (day) connector flows enter as weekday/weekend x time-of-day profiles, while the real generator redraws demand
        every day.
Replay every train day (real upstream demand, no incidents) under: base | cap | day | cap+day, and score the
first-morning-queue site against the real data (same site within +-2 links, time difference, dominance).
  cap: links b..b+2 of each regular site (G34 events, n >= 3) get capacity = the median breakdown flow; other links
       max(fd capacity, 1.02 x the largest 30-min mean flow they carried on any train day).
  day: per day and junction, the morning (06:00-10:00) median deviation of the free-flow log ratio from the profile
       (0 where fewer than 12 free-flow pairs).
usage: python fid_replay.py [panels=ALL]"""
import sys, time, numpy as np, pandas as pd, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/twin")
from cachelib import load, T2_PANELS
from ctm import Corridor, junction_ratios

E = pd.read_csv(r"C:/Users/user/tfb/work/t2/g34_events.csv")


def first_sites(V, vcut, Emask, day):
    truth = (np.nan_to_num(V, nan=1e9) <= vcut[None, :]) & Emask
    out = {}
    for d in range(V.shape[0] // 288):
        for s in range(72, 144):
            r = d * 288 + s
            if truth[r].any():
                out[str(day[d])] = (s, int(np.flatnonzero(truth[r]).max())); break
    return out


def cap_mult_eff(P):
    D = P.T // 288; m30 = np.nanmean(P.Q[:D * 288].reshape(D * 48, 6, P.n), axis=1)
    cap = np.maximum(P.cap, 1.02 * np.nanmax(m30, axis=0))
    for b, g in E[E.panel == P.panel].groupby("b"):
        if len(g) >= 3:
            for i in range(b, min(b + 3, P.n)):
                cap[i] = min(cap[i], np.nanmedian(g.bf)) if i > b else np.nanmedian(g.bf)
    return cap / P.cap


def day_ratios(P, RAT):
    """(D, 288, n-1): the day-type profile x exp(per-day morning deviation)"""
    D = P.T // 288
    ff = np.isfinite(P.V) & np.isfinite(P.Q) & (P.V > 0.85 * P.vf[None, :]) & (P.Q > 100)
    ok = ff[:, :-1] & ff[:, 1:]
    lr = np.where(ok, np.log(np.maximum(P.Q[:, 1:], 1) / np.maximum(P.Q[:, :-1], 1)), np.nan)[:D * 288].reshape(D, 288, -1)
    dayt = np.array([pd.Timestamp(str(x)).dayofweek >= 5 for x in P.day[:D]]).astype(int)
    prof = np.log(RAT)[dayt]                                                   # (D, 288, n-1)
    dev = lr[:, 72:120] - prof[:, 72:120]
    cnt = np.isfinite(dev).sum(axis=1); med = np.nanmedian(dev, axis=1)
    med = np.where(cnt >= 12, np.nan_to_num(med), 0.0)
    return np.exp(prof + med[:, None, :])


def replay(P, cap_mult, per_day):
    C = Corridor(P, cap_mult=cap_mult); D = P.T // 288
    RAT = junction_ratios(P)
    dayt = np.array([pd.Timestamp(str(x)).dayofweek >= 5 for x in P.day[:D]]).astype(int)
    RD = day_ratios(P, RAT) if per_day else RAT[dayt]                         # (D, 288, n-1)
    CR = C.cell_ratios(RD)
    up = pd.Series(P.Q[:, 0].astype(float)).interpolate(limit_direction="both").to_numpy()[:D * 288].reshape(D, 288)
    Vc, Fc = C.run(up, CR, None)
    return Vc[:, :, C.cell_of].reshape(-1, P.n)


def score(a, b):
    common = sorted(set(a) & set(b))
    same = np.mean([abs(a[d][1] - b[d][1]) <= 2 for d in common]) if common else np.nan
    hb = pd.Series([b[d][1] // 3 for d in common]).value_counts(normalize=True)
    return same, len(common), len(set(a) - set(b)), len(set(b) - set(a)), (hb.iloc[0] if len(hb) else np.nan)


pans = T2_PANELS if len(sys.argv) < 2 or sys.argv[1] == "ALL" else sys.argv[1].split(",")
rows = []
for p in pans:
    t0 = time.time(); P = load(p, "train"); D = P.T // 288
    real = first_sites(P.V[:D * 288], P.vcut, P.E[:D * 288], P.day)
    hr = pd.Series([v[1] // 3 for v in real.values()]).value_counts(normalize=True)
    cm = cap_mult_eff(P)
    for nm, capm, pdays in (("base", None, False), ("cap", cm, False), ("day", None, True), ("cap+day", cm, True)):
        V = replay(P, capm, pdays)
        tw = first_sites(V, P.vcut, P.E[:D * 288], P.day)
        same, nc, only_r, only_t, dom = score(real, tw)
        rows.append(dict(panel=p, variant=nm, same_site=same, common=nc, only_real=only_r, only_twin=only_t, dom_twin=dom, dom_real=hr.iloc[0]))
        print(p, nm, f"same {same:.3f} common {nc} only_real {only_r} only_twin {only_t} dom {dom:.3f} (real {hr.iloc[0]:.3f}) {time.time()-t0:.0f}s", flush=True)
R = pd.DataFrame(rows)
print(R.pivot(index="panel", columns="variant", values="same_site").round(3).to_string())
print(R.groupby("variant")[["same_site", "only_real", "only_twin", "dom_twin"]].mean().round(3).to_string())
R.to_csv(r"C:/Users/user/tfb/work/twin/fid_replay.csv", index=False)
