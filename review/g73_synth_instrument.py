"""G73 (strategy §17, 10/22): the synthetic C world as an INSTRUMENT for newly activated sites (not training data).
nn_data_twinC24 onset windows (all: the synthetic set has no selector flag; a T+30 queue; never in any model's training: E4 took only its ongoing
windows). 'Activated' = every true T+30 link is rare in the REAL train months (G54 key: panel x weekend x 2-hour slot,
f <= 0.1) -> the synthetic demand redraw made a new section break down, the M3 kind (G72). Step 1 writes a test-layout dir
(nn_test_c24on) and the window list; step 2 (after CPU inference of the s46 onset models on it) scores:
  NN = s46 spec blend; physics = G69 model (work/t2/g69_phys_onset.txt); p = (1 - a) p_NN + a p_phys, a in {0.2, 0.3, 0.5}
  T+30 expected-IoU set (t2_model.best_set on eligible cells; no evidence rule), IoU, paired vs NN, SE clustered by
  panel x synthetic month; ranks of the best true link; physics percentile of the true link (G72 columns).
usage: python g73_synth_instrument.py build | score"""
import sys, os, json, pickle
sys.path.insert(0, r"C:/Users/user/tfb/src")
import numpy as np, pandas as pd
from common import NN

W = "C:/Users/user/tfb/work/t2/"; S = f"{NN}/nn_data_twinC24"; D = f"{NN}/nn_test_c24on"
cond = np.load(f"{S}/cond.npy", allow_pickle=True); sel = np.load(f"{S}/is_sel.npy"); off = np.load(f"{S}/off.npy")
panel = np.load(f"{S}/panel.npy", allow_pickle=True); tt = np.load(f"{S}/t.npy"); date = np.load(f"{S}/date.npy", allow_pickle=True)
Y = np.load(f"{S}/Y.npy", mmap_mode="r"); EL = np.load(f"{S}/EL.npy", mmap_mode="r")
if sys.argv[1] == "build":
    ks = [k for k in np.flatnonzero(cond == "queue_onset") if (np.asarray(Y[off[k]:off[k + 1], 5]) & np.asarray(EL[off[k]:off[k + 1], 5])).any()]
    X = np.load(f"{S}/X.npy", mmap_mode="r"); os.makedirs(D, exist_ok=True)
    np.save(f"{D}/X.npy", np.vstack([np.asarray(X[off[k]:off[k + 1]]) for k in ks]))
    np.save(f"{D}/off.npy", np.concatenate([[0], np.cumsum([off[k + 1] - off[k] for k in ks])]))
    np.save(f"{D}/window_id.npy", np.array([f"C24_{k}" for k in ks])); np.save(f"{D}/cond.npy", cond[ks])
    pickle.dump(ks, open(W + "g73_c24on_ks.pkl", "wb")); print("synthetic selector-like onset windows with a T+30 queue:", len(ks))
    sys.exit()

import lightgbm as lgb
from scipy.stats import rankdata
from t2_model import best_set
ks = pickle.load(open(W + "g73_c24on_ks.pkl", "rb"))
NAMES = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]
DROP = {"prior_on", "prior_og", "pos", "vf", "capl", "lanes", "length", "w", "tod", "dow", "arr5", "cond"}
COLS = [i for i, n in enumerate(NAMES) if n not in DROP]
PHC = ["dem30_over_dnmin", "arr6", "nb2_q"]


def pct(v):
    ok = np.isfinite(v); out = np.full(len(v), np.nan)
    if ok.sum() >= 2 and np.nanstd(v) > 0:
        out[ok] = (rankdata(v[ok]) - 0.5) / ok.sum()
    return out


def feats(Xw):
    P = np.column_stack([pct(Xw[:, c].astype(float)) for c in COLS]); return np.where(np.isfinite(P), P, 0.5)


# real train-month rarity (as G54): panel x weekend x 2-hour slot
os.environ["TFB_T2CACHE"] = "2_noT_af"; sys.argv = [sys.argv[0], "0"]
from t2_eval_oof import T
SELON = [k for k in range(T.NWIN) if T.META["is_sel"][k] and str(T.META["cond"][k]) == "queue_onset"]
tkey = lambda p, we, t: (str(p), bool(we), (int(t) % 288) // 24)
CNT, DEN = {}, {}
for k in SELON:
    lab, el = T.LAB[k]; key = tkey(T.META["panel"][k], pd.Timestamp(str(T.META["date"][k])[:10]).dayofweek >= 5, T.META["t"][k])
    CNT[key] = CNT.get(key, 0) + (lab[5] & el[5]).astype(float); DEN[key] = DEN.get(key, 0) + 1
spec = json.load(open(r"C:/Users/user/tfb/src/t2_specs.json"))["s37"]["spec"]
parts = [(x.split(":")[0], float(x.split(":")[1]) if ":" in x else 1.0) for x in spec.split("+")]
PR = {t: pickle.load(open(W + f"probs_c24on__{t}.pkl", "rb")) for t, _ in parts}
mdl = lgb.Booster(model_file=W + "g69_phys_onset.txt")
XT = np.load(f"{D}/X.npy", mmap_mode="r"); OT = np.load(f"{D}/off.npy")
AS = (0.2, 0.3, 0.5); rows = []
for i, k in enumerate(ks):
    w = f"C24_{k}"; Xw = np.asarray(XT[OT[i]:OT[i + 1]], dtype=float)
    y = np.asarray(Y[off[k]:off[k + 1], 5]).astype(bool); e = np.asarray(EL[off[k]:off[k + 1], 5]).astype(bool)
    key = tkey(panel[k], pd.Timestamp(str(date[k])[:10]).dayofweek >= 5, tt[k])
    f = CNT[key] / DEN[key] if key in DEN else np.zeros(len(y)); tl = np.flatnonzero(y & e)
    pn = sum(wt * PR[t][w][5] for t, wt in parts) / sum(wt for _, wt in parts); ph = mdl.predict(feats(Xw))
    r = dict(k=k, panel=str(panel[k]), month=str(date[k])[:7], act=bool((f[tl] <= 0.1).all()))
    for nm, sc in (("rk_nn", pn), ("rk_ph", ph)):
        o = np.argsort(-np.where(e, sc, -1.0)); rr = np.empty(len(o), int); rr[o] = np.arange(len(o)); r[nm] = int(rr[tl].min())
    for c in PHC:
        r[c] = float(np.nanmean(pct(Xw[:, NAMES.index(c)])[tl]))
    for a in (0.0,) + AS:
        s_ = best_set((1 - a) * pn + a * ph, e); u = ((s_ | y) & e).sum(); r[f"a{a}"] = 1.0 if u == 0 else float((s_ & y & e).sum() / u)
    rows.append(r)
R = pd.DataFrame(rows); R.to_csv(W + "g73_synth_instrument.csv", index=False)


def paired(g, c):
    d = g[c] - g["a0.0"]; cl = g.panel + "|" + g.month; cm = d.groupby(cl).sum(); n = len(d); nc = cl.value_counts()
    se = np.sqrt(len(cm) / max(len(cm) - 1, 1) * ((cm - cm.index.map(nc) * d.mean()) ** 2).sum()) / n
    return d.mean(), se


print(f"C24 onset windows {len(R)}, activated (all true links rare in the real months) {R.act.sum()} ({R.act.mean():.2f}); NN IoU {R['a0.0'].mean():.4f}")
for name, g in (("activated", R[R.act]), ("usual", R[~R.act])):
    print(f"{name}: n {len(g)} | physics percentile of the true link " + ", ".join(f"{c} {g[c].mean():.3f}" for c in PHC)
          + f" | top-1/3/5 NN {np.mean(g.rk_nn == 0):.2f}/{np.mean(g.rk_nn < 3):.2f}/{np.mean(g.rk_nn < 5):.2f}"
          + f" physics {np.mean(g.rk_ph == 0):.2f}/{np.mean(g.rk_ph < 3):.2f}/{np.mean(g.rk_ph < 5):.2f}")
for a in AS:
    out = []
    for name, g in (("all", R), ("activated", R[R.act]), ("usual", R[~R.act])):
        m_, s_ = paired(g, f"a{a}"); out.append(f"{name} {m_:+.4f} ({m_ / s_:+.1f} SE)" if s_ > 0 else f"{name} {m_:+.4f}")
    print(f"a={a}: " + " | ".join(out))
