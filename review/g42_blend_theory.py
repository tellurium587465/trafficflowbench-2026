"""G42: blend weights from theory instead of by hand (user request 10/14). The production blends weight the five NN
configurations 2/2/2/1/1 (hand-set). Theory:
  - minimum-variance combination (Bates & Granger 1969): w ∝ Σ^-1 1, Σ = covariance of the members' errors;
  - the forecast-combination puzzle (Stock & Watson 2004; Smith & Wallis 2009): with strongly correlated members the
    estimated optimum is noisy and equal weights usually win out of sample -> shrink toward equal: w = (1-λ) w_BG + λ/K.
Members = the month-CV OOF of the no-memory configurations (the same five configs as production). Errors = p - y on
eligible cells at the scored steps (onset: T+30 only; ongoing: all six). Σ is estimated on two month folds and the
weights are scored on the third (nested), λ in {0, 0.5, 1} fixed a priori (no search). Score = mean IoU of the
expected-IoU set; onset windows restricted to mornings (the official onset windows). Cluster bootstrap by panel x day.
usage: python g42_blend_theory.py"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from t2_eval_oof import T, oof_of
from t2_model import best_set

MEM = ["q25_nomem_s3", "q25_nomem_s4", "q31_nomem_wp05_s0", "q31_nomem_attn1_s0", "q31_nomem_r13_s0", "q31_nomem_drop1_s0"]
CUR = np.array([1, 1, 2, 2, 1, 1], float)                    # the SPEC used for the no-memory blend (G23 etc.)
FOLD = {**{m: 0 for m in ("2030-06", "2030-07", "2030-08")}, **{m: 1 for m in ("2030-09", "2030-10", "2030-11")},
        **{m: 2 for m in ("2030-12", "2031-01", "2031-02")}}
O = [oof_of(m) for m in MEM]
keys = sorted(set.intersection(*[set(o) for o in O]) & {k for k in range(T.NWIN) if T.META["is_sel"][k]})
rows = []
for k in keys:
    cond = str(T.META["cond"][k]); t = int(T.META["t"][k])
    if cond == "queue_onset" and (t + 6) % 288 >= 144:
        continue
    lab, el = T.LAB[k]; y = lab.astype(bool); e = el.astype(bool)
    steps = [5] if cond == "queue_onset" else list(range(6))
    m = np.zeros_like(e); m[steps] = e[steps]
    P = np.stack([o[k] for o in O])                           # (K, 6, n)
    rows.append(dict(k=k, cond=cond, fold=FOLD[str(T.META["date"][k])[:7]], cl=str(T.META["panel"][k]) + str(T.META["date"][k])[:10],
                     R=(P[:, m] - y[m]).astype(np.float32), P=P, y=y, e=e))
print("windows", len(rows), {c: sum(r["cond"] == c for r in rows) for c in ("queue_onset", "queue_ongoing")}, flush=True)


def iou(p, y, e, cond):
    q = p.copy()
    if cond == "queue_onset":
        q[:5] = 0
    s = best_set(q.ravel(), np.ones(q.size, bool)).reshape(q.shape)
    u = ((s | y) & e).sum(); return 1.0 if u == 0 else (s & y & e).sum() / u


def bg_weights(rs):
    R = np.concatenate(rs, axis=1); S = np.cov(R) + 1e-6 * np.eye(len(MEM))
    w = np.linalg.solve(S, np.ones(len(MEM))); w = np.clip(w, 0, None)
    return w / w.sum() if w.sum() > 0 else np.ones(len(MEM)) / len(MEM)


out = []
for cond in ("queue_onset", "queue_ongoing"):
    C = [r for r in rows if r["cond"] == cond]
    for f in range(3):
        wbg = bg_weights([r["R"] for r in C if r["fold"] != f])
        schemes = {"current": CUR / CUR.sum(), "equal": np.ones(len(MEM)) / len(MEM), "bg": wbg, "bg_half": 0.5 * wbg + 0.5 / len(MEM)}
        print(cond, "fold", f, "BG weights", np.round(wbg, 3).tolist(), flush=True)
        for r in (r for r in C if r["fold"] == f):
            row = dict(cond=cond, cl=r["cl"])
            for nm, w in schemes.items():
                row[nm] = iou(np.tensordot(w, r["P"], 1), r["y"], r["e"], cond)
            out.append(row)
D = pd.DataFrame(out); rng = np.random.default_rng(0)
for cond, g in D.groupby("cond"):
    u = g.cl.unique(); idx = {x: np.flatnonzero(g.cl.to_numpy() == x) for x in u}
    msg = []
    for nm in ("equal", "bg", "bg_half"):
        d = (g[nm] - g["current"]).to_numpy()
        se = np.std([d[np.concatenate([idx[x] for x in rng.choice(u, len(u))])].mean() for _ in range(500)])
        msg.append(f"{nm} {d.mean():+.4f} (SE {se:.4f})")
    print(f"{cond}: current {g.current.mean():.4f} | vs current: " + " | ".join(msg))
