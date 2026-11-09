"""G198 (11/01, ledgers #158, #165, #160, the correction of WHERE_IS_THE_GAP 5): the rare layer without the blank's side effect. Rarity from the births of at least two rows (the investigator's table, leave the month out, heads +-2 links): f_slot = share of days with a head near the link in the weekend x 2-hour slot of T+30, f_site = over all hours. A window is rare when every T+30 queue link has f_slot <= 0.1.
Reads, split by that layer: onset parts (G119), the smoothed prior w 0.25 (G128, first stage of #165). Counts: the gate of the second stage (the largest rarity_table value in the 2-hour slot of T <= 0.1, known at T) among the rare windows; ongoing windows with a new site at a queue head (f_site < 0.02), M3 dev and official (first count of #160).
usage: TFB_M3VER=v3 py -B -X utf8 g198_rare_redo.py"""
import sys, pickle
R = r"C:/Users/user/tfb/review/"; sys.dont_write_bytecode = True
src = open(R + "g119_month_postproc.py", encoding="utf-8").read().split('if __name__ == "__main__":')[0]
sys.argv = [sys.argv[0], "run"]; exec(compile(src, "g119", "exec"), globals())
BF = r"C:/Users/user/AppData/Local/Temp/claude/C--Users-user-OneDrive-Desktop/1f1db98a-17cd-431b-aafa-61be2b25086b/scratchpad/incident/births_all.parquet"; B = pd.read_parquet(BF); TR = B[(B.split == "train") & (B.nrows >= 2)]; RR = 2; FC = {}; FT = PO.rarity_table()
NL = B.groupby("panel").n.first().to_dict()


def freq(p, excl, wk=None, sl=None):
    key = (p, excl, wk, sl)
    if key not in FC:
        T_ = TR[(TR.panel == p) & (TR.month != excl)]; dd = TR[TR.month != excl].date.unique()
        if wk is not None:
            T_ = T_[(T_.wkend == wk) & (T_.slot == sl)]; dd = [d for d in dd if (pd.Timestamp(d).dayofweek >= 5) == wk]
        c_ = np.zeros(NL[p])
        for d, g in T_.groupby("date"):
            m = np.zeros(NL[p], bool)
            for h in g.head0.unique():
                m[max(h - RR, 0):h + RR + 1] = True
            c_ += m
        FC[key] = c_ / max(len(dd), 1)
    return FC[key]


def clm(d, v, n_boot=1000):
    ids = pd.factorize(d.panel + "_" + d.day.astype(str))[0]; a = np.asarray(v, float); M = np.bincount(ids, a); N = np.bincount(ids); Rm = np.random.default_rng(0).multinomial(len(M), np.full(len(M), 1 / len(M)), size=n_boot); return a.mean(), float(((Rm @ M) / (Rm @ N)).std())


Gm = pd.read_csv(W + "g119_month_postproc.csv", dtype={"w": str}); M3 = Gm[(Gm.month == "2031-03") & (Gm.cond == "queue_onset")].copy(); Tr = Gm[(Gm.vis == "mask") & (Gm.cond == "queue_onset")].copy(); O = pd.read_csv(W + "g117_onset_oracle.csv").set_index("w").fmax; rows = []
for r in M3.itertuples():
    m = G.META[r.w]; P = load(m["panel"], "validation"); t30 = m["t"] + 6; ts = pd.Timestamp(P.times[t30]); To = pd.Timestamp(P.times[m["t"]]); y = (m["y"].T.astype(bool)[-1]) & (m["m"].T.astype(bool)[-1]); f = freq(m["panel"], "none", ts.dayofweek >= 5, ts.hour // 2); ft = FT.get((m["panel"], To.dayofweek >= 5, To.hour // 2), np.zeros(P.n)); rows.append(dict(w=r.w, rare2=bool(f[y].max() <= 0.1) if y.any() else False, gate=float(ft.max()) <= 0.1, gate9=float(np.quantile(ft, 0.9)) <= 0.1, tod=To.hour))
X_ = pd.DataFrame(rows).set_index("w"); M3["rare2"] = M3.w.map(X_.rare2); M3["rare1"] = M3.w.map(O) <= 0.1; M3["gate"] = M3.w.map(X_.gate); print("M3 dev onset:", len(M3), "rare old", int(M3.rare1.sum()), "rare new", int(M3.rare2.sum()), "both", int((M3.rare1 & M3.rare2).sum()), "| gate true", int(M3.gate.sum()), "of which rare new", int((M3.gate & M3.rare2).sum()), flush=True)
fx = []
for r in Tr.itertuples():
    k = int(r.w); lab, el = T.LAB[k]; ts = pd.Timestamp(str(T.META["date"][k])[:10]); t = int(T.META["t"][k]); P = load(r.panel, "train"); t3 = pd.Timestamp(P.times[t + 6]); y = (lab[5] & el[5]).astype(bool); f = freq(r.panel, r.month, t3.dayofweek >= 5, t3.hour // 2); fx.append(bool(f[y].max() <= 0.1) if y.any() else False)
Tr["rare2"] = fx; print("train onset:", len(Tr), "rare new", int(Tr.rare2.sum()), flush=True)
P8 = pd.read_csv(W + "g128_onset_smooth_prior_m3.csv", dtype={"w": str}).set_index("w"); Pt = pd.read_csv(W + "g128_onset_smooth_prior_train.csv", dtype={"w": str}).set_index(["month", "w"]); M3["pr"] = (P8["w0.25"] - P8["w0.0"]).reindex(M3.w).to_numpy(); Tr["pr"] = [Pt["w0.25"].get((r.month, r.w), np.nan) - Pt["w0.0"].get((r.month, r.w), np.nan) for r in Tr.itertuples()]
print("component | M3 rare | M3 rest | train rare | train rest")
for nm, f_ in (("evidence E-raw", lambda D: D.E - D.raw), ("physics on evidence PE-E", lambda D: D.PE - D.E), ("A-raw", lambda D: D.PE - D.raw), ("smooth prior w 0.25", lambda D: D.pr)):
    o = [nm]
    for D in (M3, Tr):
        for m_ in (D.rare2, ~D.rare2):
            d = D[m_ & f_(D).notna()]; v, s = clm(d, f_(d)); o.append(f"{v:+.4f} ({s:.4f}, n {len(d)})")
    print(" | ".join(o), flush=True)
Fg = pd.read_csv(W + "g156_m3_frame.csv"); nw = 0; nh = []
for w in Fg.w:
    m = G.META[w]; i = G.POS[w]; lq = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1], G.CI], float) == 1; f = freq(m["panel"], "none"); l = np.flatnonzero(lq); heads = [p_[-1] for p_ in np.split(l, np.flatnonzero(np.diff(l) > 2) + 1)] if len(l) else []; nh.append(any(f[h] < 0.02 for h in heads))
print("ongoing, M3 dev", len(Fg), "windows with a new-site head", int(np.sum(nh)), flush=True)
from cachelib import T2_PANELS
from common import REL
for sp in ("validation", "private"):
    n_ = 0; a_ = 0
    for p in T2_PANELS:
        P = load(p, sp); ti = {s: i for i, s in enumerate(P.times)}; f = freq(p, "none")
        for r in pd.read_csv(REL / "task2" / p / sp / "window_index.csv").itertuples():
            if r.condition != "queue_ongoing":
                continue
            t = ti[r.forecast_origin]; l = np.flatnonzero(np.isfinite(P.V[t - 1]) & (P.V[t - 1] <= P.vcut)); a_ += 1; heads = [p_[-1] for p_ in np.split(l, np.flatnonzero(np.diff(l) > 2) + 1)] if len(l) else []; n_ += any(f[h] < 0.02 for h in heads)
    print("ongoing, official", sp, a_, "windows with a new-site head", n_, flush=True)
