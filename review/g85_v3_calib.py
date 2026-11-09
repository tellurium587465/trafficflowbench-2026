"""G85 (INSTRUMENT_V3_SPEC §2.5 / §6 V2, 10/23): the LB calibration gate on the v3 windows, main and sub weights.
Pairs (LB total-score differences as in m3v2_validate.py) re-scored on v3: m3v2_gatev3.csv (memory, evidence, far gate,
synthetic ongoing, B1, negative control), m3v2_partsv3.csv (s75: TTA + cone), m3v2_x1v3.csv if present (s76: X1 on
ongoing). Predicted LB difference = 0.15 x sum over the changed conditions of the March-composition (validation)
weighted M3 difference. Weights: main = v2 raking (panels, weekend x hour); sub = main + onset 'rare truth' share 0.40
(M4b) + ongoing persistence-IoU and T-1 queue-size bins at the June official windows' shares (G78).
Gate (spec §6): signs >= 8/9 incl. the negative control, slope 0.7-1.3.
usage: TFB_M3VER=v3 python g85_v3_calib.py"""
import sys, os
os.environ.setdefault("TFB_M3VER", "v3")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.argv = [sys.argv[0], "0"]
import numpy as np, pandas as pd
import t2_m3_gate as G, m3v2

W = G.W; V3 = G.VER
FR = {"gate": pd.read_csv(W + f"m3v2_gate{V3}.csv"), "parts": pd.read_csv(W + f"m3v2_parts{V3}.csv")}
if os.path.exists(W + f"m3v2_x1{V3}.csv"):
    FR["x1"] = pd.read_csv(W + f"m3v2_x1{V3}.csv")
# window extras: onset rare truth (f from the train months, G54 key), ongoing persistence / T-1 queue size
os.environ["TFB_T2CACHE"] = "2_noT_af"
from t2_eval_oof import T
key = lambda p, we, t: (str(p), bool(we), (int(t) % 288) // 24)
CNT, DEN = {}, {}
for k in range(T.NWIN):
    if T.META["is_sel"][k] and str(T.META["cond"][k]) == "queue_onset":
        lab, el = T.LAB[k]; kk = key(T.META["panel"][k], pd.Timestamp(str(T.META["date"][k])[:10]).dayofweek >= 5, T.META["t"][k])
        CNT[kk] = CNT.get(kk, 0) + (lab[5] & el[5]).astype(float); DEN[kk] = DEN.get(kk, 0) + 1
EX = {}
for w in G.WID:
    m = G.META[w]; P, Qm, _ = m3v2._panel(m["panel"]); t = m["t"]
    if m["cond"] == "queue_onset":
        y = m["y"][:, 5].astype(bool) & m["m"][:, 5].astype(bool); kk = key(m["panel"], ((m3v2.MARCH_FIRST_DOW + m["day"]) % 7) >= 5, t)
        f = CNT[kk] / DEN[kk] if kk in DEN else np.zeros(P.n); EX[w] = dict(rare=bool(y.any() and (f[y] <= 0.1).all()))
    else:
        qT, q30 = Qm[t].astype(bool), Qm[t + 6].astype(bool); u = (qT | q30).sum()
        pers = float((qT & q30).sum() / u) if u else 1.0; qs = int(Qm[t - 1].sum())
        EX[w] = dict(pbin=0 if pers < 0.3 else (1 if pers < 0.5 else 2), qbin=0 if qs < 15 else (1 if qs < 30 else 2))
# June official ongoing windows (G78), the same two statistics on the MASKED layer (P.blank applied, as the M3 windows
# see the masked release) - G78's own values were on the unmasked train data and are not comparable
from cachelib import T2_PANELS
from common import REL
pers_l, qs_l = [], []
for p_ in T2_PANELS:
    P_ = __import__("cachelib").load(p_, "train"); ti_ = {s_: i for i, s_ in enumerate(P_.times)}
    Vm = np.where(P_.blank, np.nan, P_.V); Qk = (np.nan_to_num(Vm, nan=1e9) <= P_.vcut[None, :]) & P_.E
    for x_ in pd.read_csv(REL / "task2" / p_ / "train" / "window_index.csv").itertuples():
        if x_.condition == "queue_ongoing":
            t_ = ti_[x_.forecast_origin]; qT, q30 = Qk[t_], Qk[t_ + 6]; u_ = (qT | q30).sum()
            pers_l.append(float((qT & q30).sum() / u_) if u_ else 1.0); qs_l.append(int(Qk[t_ - 1].sum()))
pers_a, qs_a = np.array(pers_l), np.array(qs_l)
pb = np.where(pers_a < 0.3, 0, np.where(pers_a < 0.5, 1, 2)); qb = np.where(qs_a < 15, 0, np.where(qs_a < 30, 1, 2))
EXTRA = {"queue_onset": [("rare", {True: 0.40})],
         "queue_ongoing": [("pbin", {v: float(np.mean(pb == v)) for v in (0, 1, 2)}), ("qbin", {v: float(np.mean(qb == v)) for v in (0, 1, 2)})]}
for D in FR.values():
    for c in ("rare", "pbin", "qbin"):
        D[c] = [EX[w].get(c, -1) for w in D.w]
mon = pd.DataFrame([dict(cond=G.META[w]["cond"], **EX[w]) for w in G.WID])
print("v3 windows: onset rare-truth share", round(float(mon[mon.cond == "queue_onset"].rare.mean()), 3), "(sub target 0.40) | ongoing persistence bins",
      mon[mon.cond == "queue_ongoing"].pbin.value_counts(normalize=True).sort_index().round(2).to_dict(), "target", {k: round(v, 2) for k, v in EXTRA["queue_ongoing"][0][1].items()},
      "| T-1 size bins", mon[mon.cond == "queue_ongoing"].qbin.value_counts(normalize=True).sort_index().round(2).to_dict(), "target", {k: round(v, 2) for k, v in EXTRA["queue_ongoing"][1][1].items()})
PAIRS = [("memory -> no memory", +0.0032, "gate", "mem", "nomem"), ("evidence rule", +0.0032, "gate", "ev_none", "ev_cur"),
         ("far gate", +0.0031, "gate", "gate_off", "gate_on"), ("sim ongoing (s60->s70)", +0.0035, "gate", "nomem", "sim"),
         ("B1 skip (s70->s72)", -0.0021, "gate", "ev_cur", "skip"), ("negative control (s49)", -0.0051, "gate", "nomem", "neg"),
         ("s75 TTA + cone", +0.00042, "parts", "cur", "tta_cone")]
if "x1" in FR:
    PAIRS.append(("s76 X1 ongoing", +0.00039, "x1", "base", "arm"))


def pred(D, a, b, extra):
    out = 0.0
    for c in ("queue_onset", "queue_ongoing"):
        gc = D[D.cond == c].dropna(subset=[a, b])
        if len(gc) == 0 or (gc[a] - gc[b]).abs().max() < 1e-12 or (b == "arm" and c == "queue_onset"):
            continue
        s = m3v2.summary(gc, a, b, "validation", n_boot=50, n_draw=50, extra=extra); out += 0.15 * s["per_cond"][c]["weighted"]
    return out


SUB_ON = {"queue_onset": EXTRA["queue_onset"]}                  # onset rare-truth share only (the new-site parts' sub)
rows = [dict(pair=n, lb=lb, main=pred(FR[f], a, b, None), sub=pred(FR[f], a, b, EXTRA), sub_on=pred(FR[f], a, b, SUB_ON)) for n, lb, f, a, b in PAIRS]
R = pd.DataFrame(rows); R.to_csv(W + f"g85_calib{V3}.csv", index=False)
print(R.round(5).to_string(index=False))
for v in ("main", "sub", "sub_on"):
    x, y = R[v].to_numpy(), R.lb.to_numpy(); rho = float(x @ y / (x @ x))
    print(f"  {v}: signs {int(np.sum(np.sign(x) == np.sign(y)))}/{len(x)}, slope {rho:.2f}, corr {np.corrcoef(x, y)[0, 1]:.2f}, residual sd {(y - rho * x).std():.4f}")
G_ = FR["gate"]
for nm, ex in (("main", None), ("sub", EXTRA), ("sub_on", SUB_ON)):
    _, ess = m3v2.weights(G_, "validation", extra=ex); print(f"  ESS {nm}: " + ", ".join(f"{c} {v:.0f}" for c, v in ess.items()))
