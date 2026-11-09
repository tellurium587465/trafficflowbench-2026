"""G12 (selection-aware decoding). Official ongoing windows are selected with persistence IoU <= 0.9 at T, our training
and evaluation pool also holds ~16% stabler windows, so the model's probabilities are those of the mixture. Test: a
per-group Platt correction logit p' = a_g + b_g logit p (g = link queued in the last observed row or not, x horizon
step), fitted by log-loss on the official-like windows of two month folds, scored (expected-IoU set, far gate as in
s40) on the official-like windows of the third. Control: the same correction fitted on all windows (plain
calibration, rejected before). Model: the s46-like no-memory blend OOF; SE by panel x date cluster bootstrap."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
os.environ.setdefault("TFB_T2_FAR", "3")
import t2_eval_ongoing_shift as O
from t2_eval_oof import T, load_oof, far_mask, _FARM
from scipy.optimize import minimize

SPEC = "q25_nomem_s3+q25_nomem_s4+q31_nomem_wp05_s0:2+q31_nomem_attn1_s0:2+q31_nomem_r13_s0+q31_nomem_drop1_s0"
oof = load_oof(SPEC); rng = np.random.default_rng(0)
per = np.load("E:/tfb_nn/persist_iou_afdem.npy")
FOLD = {**{m: 0 for m in ("2030-06", "2030-07", "2030-08")}, **{m: 1 for m in ("2030-09", "2030-10", "2030-11")},
        **{m: 2 for m in ("2030-12", "2031-01", "2031-02")}}


def bestset(p):
    pe = p.ravel(); o = np.argsort(-pe); cs = np.cumsum(pe[o]); kk = np.arange(1, pe.size + 1)
    kb = int(np.argmax(cs / (kk + pe.sum() - cs))) + 1; s = np.zeros(pe.size, bool); s[o[:kb]] = True
    return s.reshape(p.shape)


W = []
for k in O.SEL:
    if k not in oof:
        continue
    gate = far_mask(k)
    lq = _FARM["lq"][_FARM["off"][k]:_FARM["off"][k + 1]]
    lab, el = T.LAB[k]
    W.append(dict(base=np.clip(oof[k], 1e-6, 1 - 1e-6), lq=lq, gate=gate if gate is not None else np.zeros(len(lq), bool),
                  lab=lab.astype(bool), el=el.astype(bool), fold=FOLD[str(T.META["date"][k])[:7]], off=per[k] <= 0.9,
                  cl=f"{T.META['panel'][k]}|{str(T.META['date'][k])[:10]}"))
fold = np.array([w["fold"] for w in W]); off = np.array([w["off"] for w in W]); cl = np.array([w["cl"] for w in W])
F = W[0]["base"].shape[0]


def cells(idx):
    L, Y, G, S = [], [], [], []
    for i in idx:
        w = W[i]; m = w["el"]
        lg = np.log(w["base"] / (1 - w["base"]))
        L.append(lg[m]); Y.append(w["lab"][m]); G.append(np.broadcast_to(w["lq"][None, :], m.shape)[m])
        S.append(np.broadcast_to(np.arange(F)[:, None], m.shape)[m])
    return np.concatenate(L), np.concatenate(Y), np.concatenate(G), np.concatenate(S)


def fit(idx):
    L, Y, G, S = cells(idx); par = {}
    for g in (False, True):
        for s in range(F):
            m = (G == g) & (S == s); l, y = L[m], Y[m]
            def nll(ab):
                z = ab[0] + ab[1] * l
                return np.mean(np.logaddexp(0, z) - y * z)
            par[(g, s)] = minimize(nll, [0.0, 1.0], method="Nelder-Mead").x
    return par


def apply(w, par):
    lg = np.log(w["base"] / (1 - w["base"])); out = np.empty_like(lg)
    for s in range(F):
        a = np.where(w["lq"], par[(True, s)][0], par[(False, s)][0]); b = np.where(w["lq"], par[(True, s)][1], par[(False, s)][1])
        out[s] = 1 / (1 + np.exp(-(a + b * lg[s])))
    return out


def iou(w, p):
    s = bestset(p) & ~w["gate"][None, :]; u = ((s | w["lab"]) & w["el"]).sum()
    return 1.0 if u == 0 else (s & w["lab"] & w["el"]).sum() / u


def boot(d, idx, reps=400):
    u = np.unique(cl[idx]); g = {c: idx[cl[idx] == c] for c in u}
    return np.std([d[np.concatenate([g[c] for c in rng.choice(u, len(u))])].mean() for _ in range(reps)])


base = np.array([iou(w, w["base"]) for w in W]); sel = np.zeros(len(W)); ctl = np.zeros(len(W))
for f in range(3):
    te = np.flatnonzero((fold == f) & off)
    p_off = fit(np.flatnonzero((fold != f) & off)); p_all = fit(np.flatnonzero(fold != f))
    print(f"fold {f}: official-fit (lq, step0/5) {np.round(p_off[(True, 0)], 2)} {np.round(p_off[(True, F - 1)], 2)} "
          f"| (not lq) {np.round(p_off[(False, 0)], 2)} {np.round(p_off[(False, F - 1)], 2)}; all-fit lq step5 {np.round(p_all[(True, F - 1)], 2)}")
    for i in te:
        sel[i] = iou(W[i], apply(W[i], p_off)); ctl[i] = iou(W[i], apply(W[i], p_all))
idx = np.flatnonzero(off)
for name, v in (("selection-aware", sel), ("plain calibration", ctl)):
    d = v - base
    print(f"{name:18s} official-like n={len(idx)}: base {base[idx].mean():.4f} -> {v[idx].mean():.4f} ({d[idx].mean():+.4f}, SE {boot(d, idx):.4f})")
