"""Platt calibration of the onset probabilities before the expected-IoU decoder (10/14, review/g49_calibration.py: month-CV
onset +0.0022, SE 0.0009, +2.3 SE; ongoing already calibrated, b ~ 1.0, left unchanged).
  fit   : (a, b) of logit p' = a + b logit p on ALL month-CV OOF onset windows of the no-memory NN blend (G42 members,
          selector-like windows, onset mornings, eligible cells at T+30), penalty 1e-7 -> work/t2/onset_platt.json
  apply : reads work/t2/probs_<src>.pkl, transforms the onset windows' probabilities (every step; the onset decoder
          uses T+30), keeps ongoing windows bit-identical, writes work/t2/probs_<dst>.pkl
usage: python t2_onset_platt.py fit | python t2_onset_platt.py apply <src> <dst>"""
import sys, json, pickle, numpy as np, pandas as pd
from scipy.optimize import minimize

W = "C:/Users/user/tfb/work/t2/"
MEM = ["q25_nomem_s3", "q25_nomem_s4", "q31_nomem_wp05_s0", "q31_nomem_attn1_s0", "q31_nomem_r13_s0", "q31_nomem_drop1_s0"]
CUR = np.array([1, 1, 2, 2, 1, 1], float); CUR /= CUR.sum()
lg = lambda x: np.log(x / (1 - x))

if sys.argv[1] == "fit":
    from t2_eval_oof import T, oof_of
    O = [oof_of(m) for m in MEM]
    keys = sorted(set.intersection(*[set(o) for o in O]) & {k for k in range(T.NWIN) if T.META["is_sel"][k]})
    A, Y = [], []
    for k in keys:
        if str(T.META["cond"][k]) != "queue_onset" or (int(T.META["t"][k]) + 6) % 288 >= 144:
            continue
        lab, el = T.LAB[k]; m = el.astype(bool)[5]
        p = np.clip(np.tensordot(CUR, np.stack([o[k] for o in O]), 1), 1e-5, 1 - 1e-5)[5]
        A.append(np.column_stack([np.ones(m.sum()), lg(p[m])])); Y.append(lab.astype(bool)[5][m].astype(float))
    A, Y = np.vstack(A), np.concatenate(Y)
    f = lambda w: (np.mean(np.logaddexp(0, A @ w) - Y * (A @ w)) + 1e-7 * (w @ w), A.T @ (1 / (1 + np.exp(-(A @ w))) - Y) / len(Y) + 2e-7 * w)
    w = minimize(f, np.array([0.0, 1.0]), jac=True, method="L-BFGS-B").x
    json.dump(dict(a=float(w[0]), b=float(w[1]), cells=int(len(Y)), members=MEM, weights=CUR.tolist(),
                   source="review/g49_calibration.py protocol, all folds pooled"), open(W + "onset_platt.json", "w"), indent=1)
    print(f"onset Platt a={w[0]:.4f} b={w[1]:.4f} on {len(Y)} cells")
else:
    from common import REL
    from cachelib import T2_PANELS
    src, dst = sys.argv[2], sys.argv[3]; cfg = json.load(open(W + "onset_platt.json"))
    probs = pickle.load(open(W + f"probs_{src}.pkl", "rb")); cond = {}
    for sp in ("validation", "private"):
        for p in T2_PANELS:
            w_ = pd.read_csv(REL / "task2" / p / sp / "window_index.csv"); cond.update(dict(zip(w_.window_id, w_.condition)))
    assert set(probs) == set(cond)
    out = {}
    for k, v in probs.items():
        if cond[k] == "queue_onset":
            pc = np.clip(v, 1e-5, 1 - 1e-5); out[k] = 1 / (1 + np.exp(-(cfg["a"] + cfg["b"] * lg(pc))))
        else:
            out[k] = v
    pickle.dump(out, open(W + f"probs_{dst}.pkl", "wb"))
    n_on = sum(cond[k] == "queue_onset" for k in out)
    print(f"Platt a={cfg['a']:.4f} b={cfg['b']:.4f} applied to {n_on} onset windows; {len(out) - n_on} ongoing windows unchanged -> probs_{dst}.pkl")
