"""M1: counterfactual sensitivity of the saved ongoing LightGBM (v14af, trained on all non-selector train windows) on
selector-like train windows (held out of its training) and on the test ongoing windows (inputs only, no labels).

Counterfactuals (inputs edited, model unchanged):
  tod+2h / tod-2h        time of day shifted
  dow_swap               weekday <-> weekend (Wed <-> Sun)
  static_med             capl/lanes/length/vf/w replaced by the window's panel medians (link fingerprint removed)
  dem_neutral            demand look-ahead without the train profile: dem{10,20,30}_c = last_q (persistence),
                         dem30_over_dnmin scaled the same way, dem_shape30 = 1, dem_level = 1
Readout per group (train / validation / private): far predicted cells (best_set, > 3 links from the queue at T) that
survive the counterfactual, change in far / near predicted cells, mean |dp| at far and near cells; train also IoU.
"""
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, r"C:/Users/user/tfb/src")
import os
os.environ["TFB_T2CACHE"] = "2_noT_af"; os.environ["TFB_T2_FAR"] = "3"
import t2_eval_oof as EV
from t2_model import best_set
from t2_windows import iou

names = list(np.load("E:/tfb_nn/nn_data_afdem/names.npy", allow_pickle=True)); ix = {c: i for i, c in enumerate(names)}
B = lgb.Booster(model_file="C:/Users/user/tfb/work/t2/queue_ongoing_v14af.txt")
X = np.load("E:/tfb_nn/nn_data_afdem/X.npy", mmap_mode="r"); OFF = np.load("E:/tfb_nn/nn_data_afdem/off.npy")
dt = "E:/tfb_nn/nn_test_afdem/"; Xt = np.load(dt + "X.npy", mmap_mode="r"); offt = np.load(dt + "off.npy")
wid = np.load(dt + "window_id.npy", allow_pickle=True).astype(str); condt = np.load(dt + "cond.npy", allow_pickle=True).astype(str)
M, LAB = EV.T.META, EV.T.LAB
STAT = ["capl", "lanes", "length", "vf", "w"]


def predict(x):
    h = [np.column_stack([x, np.full(len(x), s, np.float32), x[:, ix["seg_tail"]] + np.nan_to_num(x[:, ix["s_flow"]]) * s * 5 / 60,
                          x[:, ix["seg_tail"]] + np.nan_to_num(x[:, ix["s_emp_tail"]]) * s * 5 / 60]) for s in range(1, 7)]
    return B.predict(np.vstack(h), num_threads=16).reshape(6, len(x))


def cfs(x):
    out = {}
    for nm, dh in (("tod+2h", 2.0), ("tod-2h", -2.0)):
        y = x.copy(); y[:, ix["tod"]] = np.clip(y[:, ix["tod"]] + dh, 0, 23.9); out[nm] = y
    y = x.copy(); d = y[:, ix["dow"]]; y[:, ix["dow"]] = np.where(d >= 5, 2.0, 6.0); out["dow_swap"] = y
    y = x.copy()
    for c in STAT:
        y[:, ix[c]] = np.nanmedian(x[:, ix[c]])
    out["static_med"] = y
    y = x.copy(); lq = x[:, ix["last_q"]]; r = x[:, ix["dem30_over_dnmin"]] / np.where(x[:, ix["dem30_c"]] > 0, x[:, ix["dem30_c"]], np.nan)
    for c in ("dem10_c", "dem20_c", "dem30_c"):
        y[:, ix[c]] = lq
    y[:, ix["dem30_over_dnmin"]] = lq * r; y[:, ix["dem_shape30"]] = 1.0; y[:, ix["dem_level"]] = 1.0; out["dem_neutral"] = y
    return out


def farmask(x):
    lq = x[:, ix["last_queue"]] == 1; near = np.zeros(len(x), bool)
    for j in np.flatnonzero(lq):
        near[max(0, j - 3):j + 4] = True
    return ~near


def one(x, lab=None, el=None):
    p0 = predict(x); s0 = best_set(p0.ravel(), np.ones(p0.size, bool)).reshape(p0.shape); far = farmask(x)[None, :]
    r = {"base": dict(far=int((s0 & far).sum()), near=int((s0 & ~far).sum()))}
    if lab is not None:
        r["base"]["iou"] = iou(s0, lab, el)
    for nm, y in cfs(x).items():
        p = predict(y); s = best_set(p.ravel(), np.ones(p.size, bool)).reshape(p.shape)
        r[nm] = dict(far=int((s & far).sum()), near=int((s & ~far).sum()), keep_far=int((s & s0 & far).sum()),
                     dpf=float(np.abs(p - p0)[np.broadcast_to(far, p.shape)].mean()), dpn=float(np.abs(p - p0)[np.broadcast_to(~far, p.shape)].mean()))
        if lab is not None:
            r[nm]["iou"] = iou(s, lab, el)
    return r


rng = np.random.default_rng(0)
ks = rng.choice(np.flatnonzero(M["is_sel"] & (M["cond"] == "queue_ongoing")), 500, replace=False)
if __name__ == "__main__":
    rows = []
    for k in ks:
        lab, el = LAB[k]
        for nm, d in one(np.asarray(X[OFF[k]:OFF[k + 1]], np.float32), lab.astype(bool), el.astype(bool)).items():
            rows.append(dict(split="train", k=int(k), cf=nm, **d))
    for i, w in enumerate(wid):
        if condt[i] == "queue_ongoing":
            for nm, d in one(np.asarray(Xt[offt[i]:offt[i + 1]], np.float32)).items():
                rows.append(dict(split=w.split("_")[4], k=i, cf=nm, **d))
    R = pd.DataFrame(rows); R.to_pickle(r"C:/Users/user/tfb/review/m1_cf.pkl")
    base = R[R.cf == "base"].set_index(["split", "k"])
    for (sp, cf), g in R[R.cf != "base"].groupby(["split", "cf"]):
        b = base.loc[sp].loc[g.k]
        line = (f"{sp:10s} {cf:12s} far kept {g.keep_far.sum() / max(b.far.sum(), 1):.3f}  far {b.far.sum():5d}->{g.far.sum():5d}  "
                f"near {b.near.sum():6d}->{g.near.sum():6d}  |dp| far {g.dpf.mean():.4f} near {g.dpn.mean():.4f}")
        if sp == "train":
            line += f"  IoU {b.iou.mean():.4f}->{g.iou.mean():.4f}"
        print(line, flush=True)
