"""M1: what the far ongoing predictions rest on, train OOF (blend) vs test (s31 blend probs), input-only on the test side.

Far cell = predicted queued (best_set) on a link > 3 from any queue at T. For each far predicted link: precursor at T-1
(own or +-1 neighbour v/vf < 0.9 at the last observation, or v_min over the history < 0.9), the demand look-ahead
(dem30_c = forecast flow / cap at T+30, dem_shape30 = train profile ratio P(T+30)/P(T-5)), the location-time prior U
(daily queue frequency of that link in that 30-min bin over train months, same day type) and tod.
Train: precision of far predictions per group. Test: share of far predictions per group (no labels).
"""
import os, sys
os.environ["TFB_T2CACHE"] = "2_noT_af"; os.environ["TFB_T2_FAR"] = "3"
sys.path.insert(0, r"C:/Users/user/tfb/src")
import pickle, numpy as np, pandas as pd
import t2_eval_oof as EV
from t2_model import best_set
from cachelib import load, T2_PANELS
from common import REL
from t2_windows import queue_matrix

SPEC = "l127_full_af:2+q14_attn1_glob+q16_attnglob_s1+q14_attn1_wp05:2+q13_attn1:2+q12_af_r13_nopos+q17_drop1"
M, LAB = EV.T.META, EV.T.LAB
names = list(np.load("E:/tfb_nn/nn_data_afdem/names.npy", allow_pickle=True)); ix = {c: names.index(c) for c in names}
COLS = ["last_v", "v_min", "nb1_v", "nb-1_v", "last_q", "dem30_c", "dem30_over_dnmin", "dem_shape30", "dem_level", "tod", "dow"]
X = np.load("E:/tfb_nn/nn_data_afdem/X.npy", mmap_mode="r"); OFF = np.load("E:/tfb_nn/nn_data_afdem/off.npy")
lqz = np.load("E:/tfb_nn/nn_data_afdem_lastq.npz"); LQ = lqz["lq"]
dt = "E:/tfb_nn/nn_test_afdem/"; Xt = np.load(dt + "X.npy", mmap_mode="r"); offt = np.load(dt + "off.npy")
wid = list(np.load(dt + "window_id.npy", allow_pickle=True).astype(str)); condt = np.load(dt + "cond.npy", allow_pickle=True).astype(str)


def farm(lq):
    near = np.zeros(len(lq), bool)
    for j in np.flatnonzero(lq):
        near[max(0, j - 3):j + 4] = True
    return ~near


def Umaps():
    out = {}
    for p in T2_PANELS:
        P = load(p, "train"); D = P.T // 288; qb = queue_matrix(P).reshape(D, 48, 6, P.n).any(axis=2)
        days = [str(x) for x in P.day]; mon = np.array([d[:7] for d in days]); wk = np.array([pd.Timestamp(d).dayofweek >= 5 for d in days])
        for m in sorted(set(mon)) + ["none"]:
            for we in (False, True):
                out[(p, m, we)] = qb[(mon != m) & (wk == we)].mean(axis=0)
        load.cache_clear()
    return out


U = Umaps()
rows = []
oof = EV.load_oof(SPEC)
for k in np.flatnonzero(M["is_sel"] & (M["cond"] == "queue_ongoing")):
    o = oof[k]; s = best_set(o.ravel(), np.ones(o.size, bool)).reshape(o.shape)
    lq = LQ[OFF[k]:OFF[k + 1]]; far = farm(lq); lab, el = LAB[k]
    js = np.flatnonzero((s & far[None, :]).any(axis=0))
    if not len(js):
        continue
    xx = np.asarray(X[OFF[k]:OFF[k + 1]])[:, [ix[c] for c in COLS]]
    d = str(M["date"][k]); we = pd.Timestamp(d).dayofweek >= 5; Uk = U[(M["panel"][k], d[:7], we)]; b = (int(M["t"][k]) % 288) // 6
    for j in js:
        npred = int((s[:, j]).sum()); tp = int((s[:, j] & lab[:, j].astype(bool) & el[:, j].astype(bool)).sum())
        rows.append(dict(split="train", k=int(k), j=int(j), npred=npred, tp=tp, U=float(Uk[min(b + 1, 47), j]), **dict(zip(COLS, xx[j]))))
del oof
pt = pickle.load(open("C:/Users/user/tfb/work/t2/probs_s31_t1AdaptGapMLClipFd_t2v14afAttnR13Drop_t4l2.pkl", "rb"))
for i, w in enumerate(wid):
    if condt[i] != "queue_ongoing":
        continue
    o = pt[w]; s = best_set(o.ravel(), np.ones(o.size, bool)).reshape(o.shape)
    xx = np.asarray(Xt[offt[i]:offt[i + 1]]); lq = xx[:, ix["last_queue"]] == 1; far = farm(lq)
    js = np.flatnonzero((s & far[None, :]).any(axis=0))
    p = w.split("_")[1] + "_" + w.split("_")[2] + "_" + w.split("_")[3]; split = w.split("_")[4]
    wi = pd.read_csv(REL / "task2" / p / split / "window_index.csv").set_index("window_id")
    d = wi.loc[w, "date"]; we = pd.Timestamp(d).dayofweek >= 5; b = int((pd.Timestamp(wi.loc[w, "forecast_origin"]).hour * 60 + pd.Timestamp(wi.loc[w, "forecast_origin"]).minute) // 30)
    Uk = U[(p, "none", we)]
    for j in js:
        rows.append(dict(split=split, k=i, j=int(j), npred=int(s[:, j].sum()), tp=np.nan, U=float(Uk[min(b + 1, 47), j]),
                         **dict(zip(COLS, xx[j, [ix[c] for c in COLS]]))))
R = pd.DataFrame(rows); R.to_pickle(r"C:/Users/user/tfb/review/m1_farcells.pkl")
R["prec"] = (np.nanmin(R[["last_v", "v_min", "nb1_v", "nb-1_v"]].to_numpy(), axis=1) < 0.9)
R["Ub"] = pd.cut(R.U, [-0.01, 0.05, 0.2, 0.5, 1.01])
for g in ("prec", "Ub"):
    a = R.groupby(["split", g], observed=True).agg(cells=("npred", "sum"), tp=("tp", "sum")).reset_index()
    a["share"] = a.cells / a.groupby("split").cells.transform("sum"); a["precision"] = a.tp / a.cells
    print(a.round(3).to_string())
print(R.groupby("split")[["dem30_c", "dem30_over_dnmin", "dem_shape30", "dem_level", "last_q", "U", "tod"]].median().round(3).to_string())
print(R.groupby("split").agg(cells=("npred", "sum"), wins=("k", "nunique")).to_string())
