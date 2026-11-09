"""G28: how noisy is a paired LB comparison? Real OOF (train months, official-like windows: onset selector set,
ongoing with exact persistence <= 0.9), far gate as in s40 for both. A month of the LB = 5 onset + 5 ongoing windows
per panel drawn from ONE train month; total-score difference = 0.30 x mean over panels of (onset + ongoing)/2 window
differences. 3000 draws -> sd = the standard error of an LB paired difference for this pair of models."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
os.environ.setdefault("TFB_T2_FAR", "3")
import t2_eval_ongoing_shift as O, t2_eval_surprise as S
from t2_eval_oof import T, load_oof, far_mask

PAIRS = [("memory blend (s31-like)", "l127_full_af:2+q14_attn1_glob+q16_attnglob_s1+q14_attn1_wp05:2+q13_attn1:2+q12_af_r13_nopos+q17_drop1",
          "no-memory blend (s46-like)", "q25_nomem_s3+q25_nomem_s4+q31_nomem_wp05_s0:2+q31_nomem_attn1_s0:2+q31_nomem_r13_s0+q31_nomem_drop1_s0"),
         ("no-memory seeds 3", "q25_nomem_s3", "no-memory seed 4", "q25_nomem_s4")]
per = np.load("E:/tfb_nn/persist_iou_afdem.npy")


def bestset(p):
    pe = p.ravel(); o = np.argsort(-pe); cs = np.cumsum(pe[o]); kk = np.arange(1, pe.size + 1)
    kb = int(np.argmax(cs / (kk + pe.sum() - cs))) + 1; s = np.zeros(pe.size, bool); s[o[:kb]] = True
    return s.reshape(p.shape)


def ious(spec):
    oof = load_oof(spec); r = {}
    for k in list(S.SEL) + [k for k in O.SEL if per[k] <= 0.9]:
        if k not in oof:
            continue
        p = oof[k].copy(); lab, el = T.LAB[k]; lab = lab.astype(bool); el = el.astype(bool)
        if T.META["cond"][k] == "queue_onset":
            p[:-1] = 0
        s = bestset(p); fm = far_mask(k)
        if fm is not None:
            s = s & ~fm[None, :]
        u = ((s | lab) & el).sum(); r[k] = 1.0 if u == 0 else (s & lab & el).sum() / u
    return r


rng = np.random.default_rng(0)
for na, a, nb, b in PAIRS:
    A, B = ious(a), ious(b); ks = [k for k in A if k in B]
    d = pd.DataFrame(dict(k=ks, d=[B[k] - A[k] for k in ks], panel=[str(T.META["panel"][k]) for k in ks],
                          cond=[str(T.META["cond"][k]) for k in ks], mon=[str(T.META["date"][k])[:7] for k in ks]))
    sims = []
    for _ in range(3000):
        tot = []
        for pn, g in d.groupby("panel"):
            g = g[g.mon == rng.choice(g.mon.unique())]
            on = g[g.cond == "queue_onset"].d.to_numpy(); og = g[g.cond == "queue_ongoing"].d.to_numpy()
            if len(on) and len(og):
                tot.append(0.5 * (rng.choice(on, min(5, len(on)), replace=False).mean() + rng.choice(og, min(5, len(og)), replace=False).mean()))
        sims.append(0.30 * np.mean(tot))
    sims = np.array(sims)
    print(f"{nb} - {na}: local mean total diff {sims.mean():+.4f}, LB-scale sd {sims.std():.4f}, P(>0) {np.mean(sims > 0):.2f}, "
          f"per-window |d|>0 share {np.mean(d.d != 0):.2f}, sd(d) {d.d.std():.3f}")
