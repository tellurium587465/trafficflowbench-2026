"""G117 (10/29, ledger #70, strategy J11): an onset ceiling from an oracle split of the M3 v3 DEV onset windows (the
held-out days are not read: they belong to the 11/4 final CV) into "no precursor" windows and the rest.
Label set Y = the links queued at T+30 (y[:, 5] & m[:, 5]). No precursor = all of
  (a) speed: in each of the rows T-6..T-1 (the last 30 min, March as released = the masked layer, NaN skipped) the mean
      over Y of V / vff >= 0.98, vff = the link's free-flow level (median of the training months' eligible speeds
      >= 0.85 x fd free speed). Also printed with vff = the fd free speed (q0.99), the strategy's literal reading.
  (b) upstream: the mean over rows T-6..T-1 and over Y plus the 3 links upstream of Y (direction from the topology's
      incoming links) of flow / the training months' weekday x time-of-day mean flow (t2_demand.profile) in [0.85, 1.15]
  (c) site: the rarity table's T+30 frequency (t2_phys_onset.rarity_table, panel x weekend x 2-hour slot of T) <= 0.1 on
      every link of Y (the "new site" test of month_events).
IoU per window = the final A flow's onset IoU (k expectation, simulated blanks), column A of work/t2/g108_rectangles.csv.
u = the share of windows with no precursor (the mean IoU is a window mean; the label-mass share is printed too).
Ceiling = 1 - u (1 - IoU_u) (the precursor windows made perfect, the others left as they are); room = (1 - u)(1 - IoU_p).
usage: TFB_M3VER=v3 py g117_onset_oracle.py"""
import os, sys
os.environ.setdefault("TFB_M3VER", "v3")
sys.path.insert(0, r"C:/Users/user/tfb/src"); argv0 = sys.argv[0]; sys.argv = [argv0, "0"]
import numpy as np, pandas as pd
import t2_m3_gate as G, m3v2, t2_phys_onset as PO
from cachelib import load
from t2_demand import profile

A = pd.read_csv(G.W + "g108_rectangles.csv").set_index("w")["A"]
F_ = PO.rarity_table()
VFF, PROF, UP = {}, {}, {}
for p in sorted({G.META[w]["panel"] for w in G.WID}):
    PT = load(p, "train"); ff = np.where(PT.E & (PT.V >= 0.85 * PT.vf[None, :]), PT.V, np.nan)
    VFF[p] = np.nanmedian(ff, axis=0); PROF[p] = profile(PT)
    li = {l: i for i, l in enumerate(PT.links)}; sg = []
    for j, s in enumerate(PT.topo.incoming_link_ids.fillna("").astype(str)):
        sg += [np.sign(li[x] - j) for x in s.split(";") if x in li]
    UP[p] = int(np.sign(np.median(sg)))                          # -1: upstream = lower index
    load.cache_clear()
print("upstream direction per panel:", UP)
rows = []
for w in G.WID:
    m = G.META[w]
    if m["cond"] != "queue_onset" or m3v2.is_holdout(np.array([m["day"]]))[0]:
        continue
    P, _, _ = m3v2._panel(m["panel"]); t = m["t"]; Y = np.flatnonzero(m["y"][:, 5].astype(bool) & m["m"][:, 5].astype(bool))
    r = dict(w=w, panel=m["panel"], day=m["day"], nY=len(Y), iou=float(A[w]))
    if len(Y):
        H = P.V[t - 6:t][:, Y]
        with np.errstate(all="ignore"):
            r["v_ff"] = np.nanmin(np.nanmean(H / VFF[m["panel"]][Y][None, :], axis=1))
            r["v_q99"] = np.nanmin(np.nanmean(H / P.vf[Y][None, :], axis=1))
        lo, hi = Y.min(), Y.max(); R = np.r_[Y, np.arange(lo - 3, lo) if UP[m["panel"]] < 0 else np.arange(hi + 1, hi + 4)]
        R = R[(R >= 0) & (R < P.n)]
        dow = pd.Timestamp(P.times[t][:10]).dayofweek; pr = PROF[m["panel"]][dow][(np.arange(t - 6, t) % 288)][:, R]
        with np.errstate(all="ignore"):
            r["flow_ratio"] = np.nanmean(P.Q[t - 6:t][:, R]) / np.nanmean(pr)
        key = (m["panel"], dow >= 5, (t % 288) // 24); f = F_.get(key, np.zeros(P.n))
        r["fmax"] = float(f[Y].max())
    rows.append(r)
D = pd.DataFrame(rows); D.to_csv(G.W + "g117_onset_oracle.csv", index=False)
D = D[D.nY > 0].copy()
print(f"dev onset windows with a label at T+30: {len(D)} (mean IoU {D.iou.mean():.4f}); windows with no label: {len(rows) - len(D)}")
print("v_ff (min over rows of the mean V / vff over Y) quantiles:", D.v_ff.quantile([.05, .25, .5, .75, .95]).round(3).tolist())
print("flow ratio quantiles:", D.flow_ratio.quantile([.05, .25, .5, .75, .95]).round(3).tolist(), "| rare site (fmax <= 0.1):", f"{(D.fmax <= 0.1).mean():.3f}")
a = (D.v_ff >= 0.98); a_lit = (D.v_q99 >= 0.98); b = D.flow_ratio.between(0.85, 1.15); c = D.fmax <= 0.1
for nm, u_ in (("(a) speed", a), ("(a) speed, literal vf q0.99", a_lit), ("(a)+(c)", a & c), ("(a)+(b)+(c) primary", a & b & c), ("(c) rare site only", c)):
    u = u_.mean(); iu = D.iou[u_].mean() if u_.any() else float("nan"); ip = D.iou[~u_].mean()
    um = D.nY[u_].sum() / D.nY.sum()
    print(f"{nm:28s} u {u:.3f} (n {int(u_.sum())}, label mass {um:.3f}) | IoU_u {iu:.3f} IoU_p {ip:.3f} | ceiling 1 - u(1 - IoU_u) = {1 - u * (1 - iu):.3f} | room (1-u)(1-IoU_p) = {(1 - u) * (1 - ip):.3f}"
          f" | of IoU<0.5 windows: {(u_ & (D.iou < 0.5)).sum()}/{(D.iou < 0.5).sum()}")
u_ = a & b & c; u = u_.mean(); iu = D.iou[u_].mean(); ip = D.iou[~u_].mean()
for nm, s in (("private (incidents 7 vs 5)", 7 / 5), ("training months (rare sites 0.28 vs 0.40)", 0.28 / 0.40)):
    uu = min(1.0, u * s); print(f"{nm}: u {uu:.3f} -> ceiling {1 - uu * (1 - iu):.3f}, the current mix at IoU_u / IoU_p {uu * iu + (1 - uu) * ip:.3f}")
