"""G361 (11/05, ledger #288 stage 0, measurement only): do release blanks (Task 2 horizons of earlier official windows) hide the evidence that
the production rules read as "no queue"? March / April official windows; reads only rows < T and window_index origins < T.
  onset evidence (t2_onset_evidence): rows dd-6 .. dd+11 of up to 3 earlier days (dd = T - back*288), +-2 links; a link-day is "blind"
      when every row of its range is non-finite;
  ongoing OGB (x0.85): rows dd-18 .. dd+11, +-1 link; far gate: the same day 05:00 .. T-5 and rows dd-18 .. dd+11, +-2 links.
Per window: link-days blind / all; the blind link-days whose range overlaps (>= half of its rows) a previous official window's horizon + 1 h
(rows T'+1 .. T'+18) of the same panel; links penalised (no evidence after the dilation) that are blind on every earlier day (penalised for
lack of sight, not for a seen absence). By day of month 1-2 / 3+. Also the release's horizon rows of the official windows (finite share of
rows T+1 .. T+6 and T+7 .. T+18 against the month), and on the training split's June the P.blank share of those rows.
usage: py -B -X utf8 g361_blank_evidence.py"""
import sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from common import REL


def dil(m, r):
    out = m.copy()
    for o in range(1, r + 1):
        out[o:] |= m[:-o]; out[:-o] |= m[o:]
    return out


rows = []; hz = []
for sp in ("validation", "private"):
    for p in T2_PANELS:
        P = load(p, sp); fin = np.isfinite(P.V); q = fin & (np.nan_to_num(P.V, nan=1e9) <= P.vcut[None, :]); ti = {s: i for i, s in enumerate(P.times)}
        wi = pd.read_csv(REL / "task2" / p / sp / "window_index.csv"); wi["t"] = [ti[o] for o in wi.forecast_origin]; wi = wi.sort_values("t")
        mf = float(fin.mean())
        for r in wi.itertuples():
            t = r.t; prior = wi.t[wi.t < t].to_numpy(); blocks = [(u + 1, u + 19) for u in prior]
            hz.append(dict(split=sp, h1=float(fin[t + 1:t + 7].mean()), h2=float(fin[t + 7:t + 19].mean()), month=mf))
            specs = [("onset_ev", -6, 12, 2)] if r.condition == "queue_onset" else [("og_OGB", -18, 12, 1), ("far_gate", -18, 12, 2)]
            for nm, lo, hi, rad in specs:
                blind_all = np.ones(P.n, bool); seen = np.zeros(P.n, bool); nb = nld = nov = 0; nd = 0
                for back in (1, 2, 3):
                    dd = t - back * 288
                    if dd + lo < 0:
                        break
                    a, b = dd + lo, dd + hi; bl = ~fin[a:b].any(axis=0); nb += int(bl.sum()); nld += P.n; nd += 1
                    ov = any(min(b, e) - max(a, s) >= (b - a) / 2 for s, e in blocks); nov += int(bl.sum()) if ov else 0
                    blind_all &= bl; seen |= q[a:b].any(axis=0)
                if nm == "far_gate":
                    d0 = t - (t % 288) + 60
                    if t > d0:
                        seen |= q[d0:t].any(axis=0); blind_all &= ~fin[d0:t].any(axis=0)
                if nd == 0:
                    continue
                ev = dil(seen, rad); pen = ~ev
                rows.append(dict(split=sp, rule=nm, day=int(r.date[8:10]), blind=nb, linkdays=nld, blind_in_prev_horizon=nov, penalised=int(pen.sum()), penalised_blind_all=int((pen & blind_all).sum()), links=P.n))
        load.cache_clear()
D = pd.DataFrame(rows); D["band"] = np.where(D.day <= 2, "d1-2", "d3+"); D.to_csv("C:/Users/user/tfb/work/t2/g361_blank_evidence.csv", index=False)
g = D.groupby(["split", "rule", "band"]).agg(windows=("blind", "size"), blind_share=("blind", "sum"), linkdays=("linkdays", "sum"), in_prev=("blind_in_prev_horizon", "sum"), pen=("penalised", "sum"), pen_blind=("penalised_blind_all", "sum"))
g["blind_share"] = g.blind_share / g.linkdays; g["in_prev_share_of_blind"] = g.in_prev / (g.blind_share * g.linkdays).clip(lower=1); g["pen_blind_share_of_pen"] = g.pen_blind / g.pen.clip(lower=1)
print(g[["windows", "blind_share", "in_prev_share_of_blind", "pen", "pen_blind", "pen_blind_share_of_pen"]].round(3).to_string(), flush=True)
H = pd.DataFrame(hz); print("release, official windows' horizon finite share: " + " | ".join(f"{s}: T+5..T+30 {d.h1.mean():.3f}, T+35..T+90 {d.h2.mean():.3f}, month {d.month.mean():.3f}" for s, d in H.groupby("split")), flush=True)
A = pickle.load(open("C:/Users/user/tfb/work/t2/g342_jun80.pkl", "rb")); bl = []
for p in T2_PANELS:
    P = load(p, "train")
    for r in [x for x in A["rows"] if x["panel"] == p]:
        bl.append((float(P.blank[r["t"] + 1:r["t"] + 7].mean()), float(P.blank[r["t"] + 7:r["t"] + 19].mean()), float(P.blank[r["t"] - 12:r["t"]].mean())))
    load.cache_clear()
bl = np.array(bl); print(f"training split June official windows: P.blank share history {bl[:, 2].mean():.3f}, horizon {bl[:, 0].mean():.3f}, +1 h after {bl[:, 1].mean():.3f}", flush=True)
