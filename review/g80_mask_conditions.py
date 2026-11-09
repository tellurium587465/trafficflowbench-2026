"""G80 (INSTRUMENT_V3_SPEC §2.1, 10/23): the selector-like train windows (cache is_sel, 2982) in two history conditions,
as test-layout dirs for out-of-month inference with the fold models:
  off : official-type - the last 12 history rows unmasked (= the training cache, TFB_T2_MASKHIST=0)  -> NN/nn_test_trsel_off
  m3  : M3-type       - the day's Task 1 mask (P.blank, the day's regime) also on T-12..T-1           -> NN/nn_test_trsel_m3
Features exactly as the cache (TFB_T2_NOT=1, TFB_T2_ALLFIN=1; window_X with zero priors + arrival + demand with the
leave-one-month-out profile). Identity: "off" must equal nn_data_afdem's rows of the same windows (priors excluded).
Also writes work/t2/trsel_meta.pkl: [(k, panel, t, cond, month, regime of the day)].
usage: python g80_mask_conditions.py <off|m3>"""
import os, sys, time, pickle
kind = sys.argv[1]
os.environ.update(TFB_T2_NOT="1", TFB_T2_ALLFIN="1", TFB_T2_MASKHIST="1" if kind == "m3" else "0", TFB_T2CACHE="2_noT_af")
sys.path.insert(0, r"C:/Users/user/tfb/src")
import numpy as np
from cachelib import load
from common import NN
from t2_model import PanelInputs
from t2_cache import window_X
from t2_arrival import arrival_features, _dmin
from t2_demand import demand_features, profile

t0 = time.time()
Z = np.load("C:/Users/user/tfb/work/t2/featcache2_noT_af.npz", allow_pickle=True)
off, panel, tt, date, cond, sel = Z["off"], Z["panel"], Z["t"], Z["date"], Z["cond"], Z["is_sel"]
names = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]
I_PON, I_POG = names.index("prior_on"), names.index("prior_og")
ks = np.flatnonzero(sel); PIs = {}; rows, meta = [], []
for k in ks:
    p = str(panel[k]); t = int(tt[k])
    if p not in PIs:
        PIs[p] = PanelInputs(load(p, "train"))
    PI = PIs[p]; P = PI.P
    Xw, _ = window_X(PI, t, str(cond[k])); Xw[:, I_PON] = 0.0; Xw[:, I_POG] = 0.0
    V, Q, tl = PI.window(t); Xa, _ = arrival_features(P, V, Q, tl, PI.x)
    Xd = demand_features(P, Q, tl, t, str(date[k])[:10], profile(P, str(date[k])[:7]), _dmin(P, PI.x))
    rows.append(np.column_stack([Xw, Xa, Xd]).astype(np.float32))
    meta.append((int(k), p, t, str(cond[k]), str(date[k])[:7], int(P.regime[t])))
D = f"{NN}/nn_test_trsel_{kind}"; os.makedirs(D, exist_ok=True)
X = np.vstack(rows); o = np.concatenate([[0], np.cumsum([len(r) for r in rows])])
np.save(f"{D}/X.npy", X); np.save(f"{D}/off.npy", o)
np.save(f"{D}/window_id.npy", np.array([f"TR_{k}" for k in ks])); np.save(f"{D}/cond.npy", cond[ks])
pickle.dump(meta, open("C:/Users/user/tfb/work/t2/trsel_meta.pkl", "wb"))
cols = [i for i, n in enumerate(names) if n not in ("prior_on", "prior_og")]
XN = np.load(f"{NN}/nn_data_afdem/X.npy", mmap_mode="r"); OFFN = np.load(f"{NN}/nn_data_afdem/off.npy")
ref = np.vstack([np.asarray(XN[OFFN[k]:OFFN[k + 1]]) for k in ks])
d = np.where(np.isnan(ref[:, cols]) & np.isnan(X[:, cols]), 0, np.abs(ref[:, cols] - X[:, cols]))
print(f"{kind}: {len(ks)} windows, {len(X)} rows -> {D} ({time.time() - t0:.0f}s); vs the training cache: max |diff| {np.nanmax(d):.3g}, "
      f"cells differing {int((d > 1e-6).sum())} ({(d > 1e-6).mean():.4f}), NaN share {np.isnan(X[:, cols]).mean():.3f} (cache {np.isnan(ref[:, cols]).mean():.3f})")
