"""G49: calibrate the Task 2 probabilities before the expected-IoU decoder (10/14). Theory: the decoder picks the set
size from the probabilities (E|S n Y| / E|S u Y| uses sum p), so over-confident probabilities give the wrong set size;
a monotone 2-parameter map (Platt: logit p' = a + b logit p) fixes the scale without re-ranking cells within a window.
G47 found R0 (Platt alone) +0.0022 (SE 0.0010) on onset with b ~ 0.63 (over-confident). Nested by month fold: (a, b)
fitted on the other folds' windows (eligible cells; onset: step T+30 only, as the production decoder; ongoing: all six
steps), applied to the held-out fold. Members = the no-memory NN blend (G42). Penalty 1e-7 (negligible).
Reports IoU raw vs Platt per condition (cluster bootstrap, panel x day) and the fitted (a, b) per fold.
Pre-registered decision (P11): adopt for a condition if >= +2 SE on this instrument and the other condition is not
harmed by more than 2 SE; then one LB submission (s70 with the calibration), kept if LB >= s70 - 0.004.
usage: python g49_calibration.py"""
import sys, numpy as np, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, r"C:/Users/user/tfb/src")
from t2_eval_oof import T, oof_of
from t2_model import best_set

MEM = ["q25_nomem_s3", "q25_nomem_s4", "q31_nomem_wp05_s0", "q31_nomem_attn1_s0", "q31_nomem_r13_s0", "q31_nomem_drop1_s0"]
CUR = np.array([1, 1, 2, 2, 1, 1], float); CUR /= CUR.sum()
FOLD = {**{m: 0 for m in ("2030-06", "2030-07", "2030-08")}, **{m: 1 for m in ("2030-09", "2030-10", "2030-11")},
        **{m: 2 for m in ("2030-12", "2031-01", "2031-02")}}
O = [oof_of(m) for m in MEM]
keys = sorted(set.intersection(*[set(o) for o in O]) & {k for k in range(T.NWIN) if T.META["is_sel"][k]})
rec = []
for k in keys:
    cond = str(T.META["cond"][k]); t = int(T.META["t"][k])
    if cond == "queue_onset" and (t + 6) % 288 >= 144:
        continue
    lab, el = T.LAB[k]
    rec.append(dict(cond=cond, f=FOLD[str(T.META["date"][k])[:7]], cl=str(T.META["panel"][k]) + str(T.META["date"][k])[:10],
                    p=np.clip(np.tensordot(CUR, np.stack([o[k] for o in O]), 1), 1e-5, 1 - 1e-5), y=lab.astype(bool), e=el.astype(bool)))
lg = lambda x: np.log(x / (1 - x))


def fit(rs, steps):
    A, Y = [], []
    for r in rs:
        m = np.zeros_like(r["e"]); m[steps] = r["e"][steps]
        A.append(np.column_stack([np.ones(m.sum()), lg(r["p"][m])])); Y.append(r["y"][m].astype(float))
    A, Y = np.vstack(A), np.concatenate(Y)
    f = lambda w: (np.mean(np.logaddexp(0, A @ w) - Y * (A @ w)) + 1e-7 * (w @ w), A.T @ (1 / (1 + np.exp(-(A @ w))) - Y) / len(Y) + 2e-7 * w)
    return minimize(f, np.array([0.0, 1.0]), jac=True, method="L-BFGS-B").x


def iou(p, r, cond):
    q = p.copy()
    if cond == "queue_onset":
        q[:5] = 0
    s = best_set(q.ravel(), np.ones(q.size, bool)).reshape(q.shape)
    u = ((s | r["y"]) & r["e"]).sum(); return 1.0 if u == 0 else (s & r["y"] & r["e"]).sum() / u


out = []
for cond, steps in (("queue_onset", [5]), ("queue_ongoing", list(range(6)))):
    C = [r for r in rec if r["cond"] == cond]
    for f in range(3):
        w = fit([r for r in C if r["f"] != f], steps)
        print(f"{cond} fold {f}: a, b = {np.round(w, 3).tolist()}", flush=True)
        for r in (r for r in C if r["f"] == f):
            pc = 1 / (1 + np.exp(-(w[0] + w[1] * lg(r["p"]))))
            out.append(dict(cond=cond, cl=r["cl"], raw=iou(r["p"], r, cond), platt=iou(pc, r, cond)))
D = pd.DataFrame(out); rng = np.random.default_rng(0)
for cond, g in D.groupby("cond"):
    u = g.cl.unique(); idx = {x: np.flatnonzero(g.cl.to_numpy() == x) for x in u}; d = (g.platt - g.raw).to_numpy()
    se = np.std([d[np.concatenate([idx[x] for x in rng.choice(u, len(u))])].mean() for _ in range(1000)])
    print(f"{cond}: windows {len(g)}, raw {g.raw.mean():.4f}, platt {g.platt.mean():.4f}, diff {d.mean():+.4f} (SE {se:.4f}, {d.mean() / se:+.1f} SE)", flush=True)
