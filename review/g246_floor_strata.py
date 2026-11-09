"""G246 (11/04, ledger #211, the strategy's (A) from panel 6): the flow floor by stratum. Per corridor x day (06-21 h) / night:
  RMSE  : flow error per lane of the production Task 1 mix on February's hidden free-flow cells (G245's harness: G168's views,
          (1 - w) chain + w net, w = 0.7 outside the T2 blanks and 0.1 inside, net = unet_s45; free flow = observed v > 0.6 v_f)
  floor : the three-indicator noise of G243 (#206) recomputed with day = 06-21 h (584 ramp-free triples, 9 months), median of the
          middle-link sigma; and the strategy's audit_20261103/triad_noise_result.csv (39 sensor triples, June-July) as a second column
  sigma_model = sqrt(RMSE^2 - floor^2) per stratum. Line (#211): a stratum with sigma_model > 10 veh/h/lane -> the next stage (the
  flow channel's net weight w_q by corridor x day/night, chosen on odd days, read on even days).
Training split only. usage: py -B -X utf8 g246_floor_strata.py"""
import sys, numpy as np, pandas as pd
R_ = r"C:/Users/user/tfb/review/"
src = open(R_ + "g243_flow_noise_triad.py", encoding="utf-8").read()
for a, b in (("day = (slot >= 72) & (slot < 240)", "day = (slot >= 72) & (slot < 252)"), ('"C:/Users/user/tfb/work/g243_flow_noise.csv"', '"C:/Users/user/tfb/work/g246_triad_0621.csv"')):
    assert src.count(a) == 1, a; src = src.replace(a, b)
src = src.split("sq = lambda x")[0]; g3 = {"__name__": "g243"}; exec(compile(src, "g243b", "exec"), g3); TR = g3["D"]
TR = TR[TR.per.isin(["day", "night"])]; fl = TR.groupby(["panel", "per"]).s2j.apply(lambda x: float(np.median(np.sqrt(x.clip(lower=0))))).rename("floor_tri")
S = pd.read_csv(r"C:/Users/user/OneDrive/Desktop/trafficflowbench/audit_20261103/triad_noise_result.csv"); S["per"] = np.where(S.night == 1, "night", "day")
fs = S.groupby(["corridor", "per"]).sig_noise_j.median().rename("floor_strat"); fs.index.names = ["panel", "per"]
s5 = open(R_ + "g245_t1_flow_postfix.py", encoding="utf-8").read().split("XS, YS, DS, BS = [], [], [], []")[0]; g5 = {"__name__": "g245"}; exec(compile(s5, "g245b", "exec"), g5)
rows = []
for p in g5["PANELS"]:
    P0 = g5["load"](p, "train"); P, Pg, gap, wl, a0 = g5["view"](P0); z = np.load(g5["O"] + p + ".npz"); u = np.load(g5["O"] + p + "_unet_s45.npz")
    V0, Q0 = z["V0"].astype(float), z["Q0"].astype(float); gap = z["gap"]; QU = np.clip(u["QU"].astype(float), 60.0, None)
    tv = Pg.target & np.isfinite(P.V) & np.isfinite(P.Q) & P.E & np.isfinite(V0); w = np.where(gap, 0.1, 0.7); Qm = (1 - w) * Q0 + w * QU
    ff = tv & (P.V > 0.6 * P.vf[None, :]); hr = (np.arange(P.T) % 288) // 12; dmask = ((hr >= 6) & (hr < 21))[:, None]; e = (Qm - P.Q) / P.lanes[None, :]
    for per, m in (("day", ff & dmask), ("night", ff & ~dmask)):
        rows.append(dict(panel=p, per=per, cells=int(m.sum()), rmse=float(np.sqrt(np.mean(e[m] ** 2))), sse=float(np.sum(e[m] ** 2))))
    g5["load"].cache_clear()
D = pd.DataFrame(rows).set_index(["panel", "per"]).join(fl).join(fs)
for c in ("floor_tri", "floor_strat"):
    D[f"sig_model_{c[6:]}"] = np.sqrt((D.rmse ** 2 - D[c] ** 2).clip(lower=0))
D.to_csv(r"C:/Users/user/tfb/work/g246_floor_strata.csv"); print(D.round(1).to_string())
for per in ("day", "night"):
    d = D.xs(per, level="per"); r = np.sqrt(d.sse.sum() / d.cells.sum()); f_ = np.sqrt((d.floor_tri ** 2 * d.cells).sum() / d.cells.sum())
    print(f"{per}: cell-weighted RMSE {r:.1f}, floor (G243 06-21) {f_:.1f}, sigma_model {np.sqrt(max(r ** 2 - f_ ** 2, 0)):.1f}")
r = np.sqrt(D.sse.sum() / D.cells.sum()); f_ = np.sqrt((D.floor_tri ** 2 * D.cells).sum() / D.cells.sum()); print(f"all: cell-weighted RMSE {r:.1f}, floor {f_:.1f}, sigma_model {np.sqrt(max(r ** 2 - f_ ** 2, 0)):.1f}")
print("strata with sigma_model > 10 (G243 floor):", int((D.sig_model_tri > 10).sum()), "of", len(D), "| (strategy floor):", int((D.sig_model_strat > 10).sum()), "of", int(D.floor_strat.notna().sum()))
