"""G90 (10/23, after LB s79e -0.0027): was E3' (ongoing) hurt by the model mismatch? On the M3 v3 ongoing windows, the
current probabilities are the PRODUCTION ongoing blend (s70 spec: q43f/q46f, seeds 0-3, weights 2,2,2,1,1) - as on the
official windows - and the earlier-day pseudo-origins' mean prediction pbar is taken either from the same production
blend (matched) or from the q24f onset blend (the mismatch shipped in s79e). Constants (a, beta, c) = (1, 1, 0) (chosen
on dev windows, 10/23 05:40); k expectation with the selector-simulated blanks hidden (as t2_m3v2_e3.stat); far gate and
decoder as the M3 v2 gate. Judged on the windows outside the hold-out (src/judge.py).
G90b (same file, mode "x1"): E3' on top of X1 - the current probabilities are the M3 v3 X1 arm of e4s_a (seeds 0,1, per k),
pbar from the e4s_a saved models at the pseudo-origins: does E3' add to X1 or double-count the same earlier-day information?
usage: TFB_M3VER=v3 python g90_e3_matched.py [x1]"""
import os, sys, pickle
os.environ.setdefault("TFB_M3VER", "v3")
if len(sys.argv) > 1 and sys.argv[1] == "x1":
    os.environ["G90_X1"] = "1"
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.argv = [sys.argv[0], "0,1,2,3"]
import numpy as np
import t2_m3_gate as G, t2_m3v2_gate as V, m3v2

W = G.W; VER = G.VER
lg = lambda x: np.log(x / (1 - x)); clip = lambda x: np.clip(x, 1e-4, 1 - 1e-4)
PM = pickle.load(open(W + f"m3{VER}prev6_meta.pkl", "rb")); BYP = {}
for e in PM:
    BYP.setdefault(e["parent"], []).append(e)
OG = [(f"{f}_{c}", w) for f in ("q43f", "q46f") for c, w in (("attnglob", 2), ("attnwp05", 2), ("attn1", 2), ("drop1", 1), ("r13", 1))]
def blend_of(prefix, spec):
    parts = [(w, pickle.load(open(W + f"probs_{prefix}__{f}_s{s}.pkl", "rb"))) for s in range(4) for f, w in spec]
    tw = sum(w for w, _ in parts); return {k: sum(w * pr[k] for w, pr in parts) / tw for k in parts[0][1]}
X1MODE = len(sys.argv) > 1 and sys.argv[1] == "x1" or os.environ.get("G90_X1", "0") == "1"
if X1MODE:
    E4 = [("e4s_a", 1)]; ld = lambda f: pickle.load(open(W + f, "rb"))
    XA = [ld(f"probs_m3{VER}v2x1__e4s_a_s{s}.pkl") for s in (0, 1)]
    CUR = {(w, k): (XA[0][(w, k)] + XA[1][(w, k)]) / 2 for (w, k) in XA[0]}
    PB = {"matched": (lambda a, b: {k: (a[k] + b[k]) / 2 for k in a})(ld(f"probs_m3{VER}prev6__e4s_a_s0.pkl"), ld(f"probs_m3{VER}prev6__e4s_a_s1.pkl"))}
else:
    CUR = blend_of(G.PRE, OG)                                   # production ongoing blend on the M3 v3 windows
    PB = {"matched": blend_of(f"m3{VER}prev6", OG), "q24f": blend_of(f"m3{VER}prev6", G.CFG5("q24f"))}
_ST = {}
def stat(w, k, src):
    key = (w, k, src)
    if key not in _ST:
        m = G.META[w]; hid = m3v2.sim_blank_rows(m["panel"], m["day"], k, m["t"]); kq = ko = sp = 0.0
        for e in BYP.get(w, []):
            if e["b"] > k:
                continue
            keep = np.array([r not in hid for r in range(e["tp"] + 1, e["tp"] + 7)])[None, :]
            mo = e["m"].astype(float) * keep; kq = kq + e["y"].astype(float) * keep; ko = ko + mo; sp = sp + PB[src][e["window_id"]].T * mo
        _ST[key] = (kq, ko, sp)
    return _ST[key]
def corrected(p_, w, k, src, a=1.0, beta=1.0, c=0.0):
    kq, ko, sp = stat(w, k, src)
    if np.isscalar(ko):
        return p_
    kq, ko, sp = kq.T, ko.T, sp.T; has = ko > 0
    pbar = clip(np.where(has, sp / np.maximum(ko, 1), 0.5)); pi = clip((kq + a * pbar) / (ko + a))
    return np.where(p_ > 0, 1 / (1 + np.exp(-(lg(clip(p_)) + np.where(has, beta * (lg(pi) - lg(pbar)) + c, 0.0)))), 0.0)
cols = {"cur": {}, "e3_matched": {}} if X1MODE else {"cur": {}, "e3_matched": {}, "e3_q24f": {}}
cur_k = (lambda w, k: CUR[(w, k)]) if X1MODE else (lambda w, k: CUR[w])
for w in G.WID:
    m = G.META[w]
    if m["cond"] != "queue_ongoing":
        continue
    kd = m3v2.kdist(m["day"]); sc = lambda p_, k: G.iou(V.ongoing_pred(p_, m, True, k), m)
    cols["cur"][w] = sum(pr * sc(cur_k(w, k), k) for k, pr in kd.items())
    for src in PB:
        cols[f"e3_{src}"][w] = sum(pr * sc(corrected(cur_k(w, k), w, k, src), k) for k, pr in kd.items())
D = V.frame(cols).dropna(subset=list(cols)); D.to_csv(W + f"g90_e3_matched{VER}" + ("_x1" if X1MODE else "") + ".csv", index=False); print("wrote", len(D), "ongoing windows")
