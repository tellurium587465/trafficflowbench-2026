"""M1: memory-scrambled yardstick with the saved ongoing LightGBM v14af on the selector-like train windows (excluded
from its training). The truth stays; the location-time coordinates the model could have memorised are moved:
  tod_shift   tod + U(1,3) h with a random sign per window (clipped to 5..23 h)
  static_perm the static link vectors (capl, lanes, length, vf, w) circularly shifted along the corridor by a random
              offset (>= 10 links) per window: every link carries another link's fingerprint
  both        both of the above
Readout: ongoing IoU (window mean and panel mean), IoU after the far rule (D=3), their difference, per variant, with a
clustered (panel x date) bootstrap SE of the far-rule difference. Anchors: LB ongoing 0.822, far rule +0.021.
"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
import m1_cf as CF
from t2_model import best_set
from t2_windows import iou

ix = CF.ix; STAT = CF.STAT; M, LAB = CF.M, CF.LAB
rng = np.random.default_rng(7)


def variant(x, v, r):
    y = x.copy()
    if v in ("tod_shift", "both"):
        y[:, ix["tod"]] = np.clip(y[:, ix["tod"]] + r["dh"], 5.0, 23.0)
    if v in ("static_perm", "both"):
        for c in STAT:
            y[:, ix[c]] = np.roll(x[:, ix[c]], r["sh"])
    return y


ks = np.flatnonzero(M["is_sel"] & (M["cond"] == "queue_ongoing")); rows = []
for k in ks:
    x = np.asarray(CF.X[CF.OFF[k]:CF.OFF[k + 1]], np.float32); lab, el = LAB[k]; lab = lab.astype(bool); el = el.astype(bool)
    far = CF.farmask(x)[None, :]; n = len(x)
    r = dict(dh=rng.uniform(1, 3) * rng.choice([-1, 1]), sh=int(rng.integers(10, max(n - 10, 11))) if n > 20 else n // 2)
    for v in ("base", "tod_shift", "static_perm", "both"):
        p = CF.predict(x if v == "base" else variant(x, v, r)); s = best_set(p.ravel(), np.ones(p.size, bool)).reshape(p.shape)
        rows.append(dict(k=int(k), panel=M["panel"][k], cl=f"{M['panel'][k]}|{str(M['date'][k])[:10]}", v=v,
                         iou=iou(s, lab, el), far=iou(s & ~far, lab, el), nfar=int((s & far).sum()), tpfar=int((s & far & lab & el).sum())))
R = pd.DataFrame(rows); R.to_pickle(r"C:/Users/user/tfb/review/m1_cfyard.pkl"); R["d"] = R.far - R.iou
for v, g in R.groupby("v", sort=False):
    u = g.cl.unique(); idx = {c: np.flatnonzero(g.cl.to_numpy() == c) for c in u}; bs = []
    for _ in range(300):
        ii = np.concatenate([idx[c] for c in rng.choice(u, len(u))]); bs.append(g.d.to_numpy()[ii].mean())
    pm = g.groupby("panel")[["iou", "far"]].mean().mean()
    print(f"{v:12s} ongoing {g.iou.mean():.4f} far-rule {g.far.mean():.4f} d {g.d.mean():+.4f} +- {np.std(bs):.4f} | panel-mean {pm.iou:.4f} {pm.far:.4f} "
          f"| far cells {g.nfar.sum()} precision {g.tpfar.sum() / max(g.nfar.sum(), 1):.3f}", flush=True)
