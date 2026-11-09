"""G337 (11/05, ledger #268, measurement only): the transfer loss by strata known at T. Per window (both conditions) of M3 v3 dev (A17;
ongoing G287 t0.8, onset G280 pc4), January / February (ongoing capB0 G287 JF t0.8, onset the OOF flow G170 L_base) and the June replica
(cvf, G321): attributes known at T = early (day of month 1-5), weekend, low history coverage (observed share of rows T-12 .. T-1 of R < 0.6,
a proxy for a preceding blank), young/small box (ongoing: G313's stratum), rare slot (onset: the largest rarity_table value of T's 2-hour slot
<= 0.1, G198's gate). Per instrument and condition: stratum shares and mean IoU; a pair's difference split into composition (sum over strata
of (w_A - w_B) IoU_B) and within (sum of w_A (IoU_A - IoU_B)). The official March / April windows' composition (window_index + nn_test_afdem R)
and the expected loss per stratum at April's composition with M3's rates. usage: TFB_M3VER=v3 py -B -X utf8 g337_transfer_strata.py"""
import os, sys, pickle
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
R0 = r"C:/Users/user/tfb/"; sys.path.insert(0, r"C:/Users/user/tfb_a8/src"); sys.path.insert(0, R0 + "src")
import t2_m3_gate as G, m3v2, t2_phys_onset as PO
from cachelib import T2_PANELS
from common import REL
W = R0 + "work/t2/"; NN = "C:/tfb_data/tfb_nn/"; RT = PO.rarity_table()


def r_attr(Rw):
    o = Rw[:, :12, 2].astype(float) > 0; v = Rw[:, :12, 0].astype(float); q = o & (v <= 1.0); rows = np.flatnonzero(o.any(axis=0)); cov = float(o.mean())
    ys = True
    if len(rows):
        ql = q[:, rows[-1]]; n = int(ql.sum()); age = 0
        for r in rows[::-1]:
            if (q[:, r] & ql).any():
                age += 1
            else:
                break
        ys = n <= 3 or age <= 3
    return cov, ys


def rare(panel, date, t):
    key = (panel, pd.Timestamp(date).dayofweek >= 5, (t % 288) // 24); f = RT.get(key); return bool(f is not None and np.max(f) <= 0.1)


rows = []
d_ = NN + "nn_test_m3v3/"; RM = np.load(d_ + "R.npy", mmap_mode="r"); om = np.load(d_ + "off.npy"); pm = {str(w): i for i, w in enumerate(np.load(d_ + "window_id.npy", allow_pickle=True))}
I7 = pd.read_csv(W + "g287_m3.csv").set_index("w")["t0.8"]; I8 = pd.read_csv(W + "g280_m3.csv").set_index("w").pc4
for w in list(I7.index) + list(I8.index):
    m = G.META[w]; cov, ys = r_attr(np.asarray(RM[om[pm[w]]:om[pm[w] + 1]])); date = f"2031-03-{m['day'] + 1:02d}"
    rows.append(dict(inst="M3", cond=m["cond"], iou=float(I7[w] if m["cond"] == "queue_ongoing" else I8[w]), dom=m["day"] + 1, early=m["day"] < 5, weekend=pd.Timestamp(date).dayofweek >= 5, lowcov=cov < 0.6, ys=ys, rare=rare(m["panel"], date, m["t"])))
A = NN + "nn_data_afdem/"; RA = np.load(A + "R.npy", mmap_mode="r"); oa = np.load(A + "off.npy"); dt = np.load(A + "date.npy", allow_pickle=True); pa = np.load(A + "panel.npy", allow_pickle=True); ta = np.load(A + "t.npy")
J7 = pd.read_csv(W + "g287_jf.csv").set_index("w")["t0.8"]; J8 = pd.read_csv(W + "g170_train.csv").set_index("w").L_base; U7 = pd.read_csv(W + "g321_jun_ongoing.csv").set_index("w")["t0.8"]; U8 = pd.read_csv(W + "g321_jun_onset.csv").set_index("w").L_base
for inst, og, on in (("JF", J7, J8), ("JUN", U7, U8)):
    for cond, S_ in (("queue_ongoing", og), ("queue_onset", on)):
        for k, v in S_.items():
            k = int(k); date = str(dt[k])[:10]; cov, ys = r_attr(np.asarray(RA[oa[k]:oa[k + 1]]))
            rows.append(dict(inst=inst, cond=cond, iou=float(v), dom=int(date[8:10]), early=int(date[8:10]) <= 5, weekend=pd.Timestamp(date).dayofweek >= 5, lowcov=cov < 0.6, ys=ys, rare=rare(str(pa[k]), date, int(ta[k]))))
d_ = NN + "nn_test_afdem/"; RO = np.load(d_ + "R.npy", mmap_mode="r"); oo = np.load(d_ + "off.npy"); po = {str(w): i for i, w in enumerate(np.load(d_ + "window_id.npy", allow_pickle=True))}
for sp, lab in (("validation", "MAR"), ("private", "APR")):
    for p in T2_PANELS:
        for r in pd.read_csv(REL / "task2" / p / sp / "window_index.csv").itertuples():
            cov, ys = r_attr(np.asarray(RO[oo[po[r.window_id]]:oo[po[r.window_id] + 1]])); t_ = pd.Timestamp(r.forecast_origin); tr = t_.hour * 12 + t_.minute // 5
            rows.append(dict(inst=lab, cond=r.condition, iou=np.nan, dom=t_.day, early=t_.day <= 5, weekend=t_.dayofweek >= 5, lowcov=cov < 0.6, ys=ys, rare=rare(p, str(r.date), tr)))
D = pd.DataFrame(rows); D.to_csv(W + "g337_strata.csv", index=False)
exec(open(R0 + "review/g337_report.py", encoding="utf-8").read())
