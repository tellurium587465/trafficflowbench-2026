"""M1: TreeSHAP (LightGBM pred_contrib) of the saved ongoing model v14af on far cells (> 3 links from the queue at T,
horizon step 6): what the far logit rests on, per feature group, for
  train far TP / far FP (selector-like windows, held out of v14af's training), train far cells not predicted,
  test far predicted cells (validation / private; inputs only).
"""
import sys, os, time, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, r"C:/Users/user/tfb/src")
os.environ["TFB_T2CACHE"] = "2_noT_af"; os.environ["TFB_T2_FAR"] = "3"
import t2_eval_oof as EV
from t2_model import best_set
import m1_cf as CF            # model, data handles, predict()

names = CF.names; ix = CF.ix
NM = names + ["step", "tail_sf", "tail_se"]
GRP = {"time": ["tod", "dow"], "static": ["capl", "lanes", "length", "vf", "w", "pos"], "demand": [c for c in NM if c.startswith("dem")],
       "arrival": [c for c in NM if c.startswith("arr")],
       "geometry": ["dq_up", "dq_dn", "head_rel", "tail_rel", "s_emp_head", "s_emp_tail", "s_flow", "seg_head", "seg_len", "seg_tail",
                    "tail_sf", "tail_se", "n_queued_now", "n_queued_hist", "nq_hist", "q_today", "age", "last_queue", "cond", "step"]}
used = set(sum(GRP.values(), [])); GRP["state"] = [c for c in NM if c not in used and c not in ("prior_on", "prior_og")]


def far_rows(x, s, lab=None):
    far = CF.farmask(x); j6 = np.flatnonzero(far)
    h = np.column_stack([x, np.full(len(x), 6, np.float32), x[:, ix["seg_tail"]] + np.nan_to_num(x[:, ix["s_flow"]]) * 0.5,
                         x[:, ix["seg_tail"]] + np.nan_to_num(x[:, ix["s_emp_tail"]]) * 0.5])
    out = []
    for j in j6:
        pred = bool(s[5, j]); y = bool(lab[5, j]) if lab is not None else None
        out.append((h[j], pred, y))
    return out


rng = np.random.default_rng(0); groups = {"train_TP": [], "train_FP": [], "train_FN": [], "test_val": [], "test_priv": []}
for k in CF.ks:
    x = np.asarray(CF.X[CF.OFF[k]:CF.OFF[k + 1]], np.float32); p = CF.predict(x); s = best_set(p.ravel(), np.ones(p.size, bool)).reshape(p.shape)
    lab, el = CF.LAB[k]
    for row, pred, y in far_rows(x, s, lab.astype(bool) & el.astype(bool)):
        g = "train_TP" if pred and y else ("train_FP" if pred else ("train_FN" if y else None))
        if g:
            groups[g].append(row)
for i, w in enumerate(CF.wid):
    if CF.condt[i] != "queue_ongoing":
        continue
    x = np.asarray(CF.Xt[CF.offt[i]:CF.offt[i + 1]], np.float32); p = CF.predict(x); s = best_set(p.ravel(), np.ones(p.size, bool)).reshape(p.shape)
    for row, pred, _ in far_rows(x, s):
        if pred:
            groups["test_val" if "validation" in w else "test_priv"].append(row)
rows = []
for g, L in groups.items():
    if not L:
        continue
    A = np.vstack(L); A = A[rng.choice(len(A), min(600, len(A)), replace=False)]
    t0 = time.time(); C = CF.B.predict(A, pred_contrib=True, num_threads=16); bias = C[:, -1].mean(); C = C[:, :-1]
    r = dict(group=g, n=len(L), logit=float((C.sum(1) + bias).mean()))
    for gn, cols in GRP.items():
        ii = [NM.index(c) for c in cols if c in NM]; r[gn] = float(C[:, ii].sum(1).mean())
    rows.append(r); print(g, len(L), f"{time.time() - t0:.0f}s", flush=True)
R = pd.DataFrame(rows).set_index("group"); print(R.round(3).to_string())
