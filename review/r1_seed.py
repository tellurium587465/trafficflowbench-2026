"""R1 audit: does the window-bootstrap SE cover training randomness?  Same config, two seeds
(q14_attn1_glob = seed 0, q16_attnglob_s1 = seed 1): per-window IoU diff on selector-like windows, SE under
three cluster levels.  Also: surprise-weighted onset diff, as t2_eval_surprise computes it."""
import sys, pickle, numpy as np
sys.path.insert(0, r"C:/Users/user/tfb/src")
import t2_eval_surprise as S
from t2_eval_oof import T

W2 = r"C:/Users/user/tfb/work/t2/"
pairs = [("q14_attn1_glob", "q16_attnglob_s1"), ("l127_full_af", "l127_fast_af")]
M = T.META

def bsiou(p, lab, el):
    pe = p.ravel(); o = np.argsort(-pe); cs = np.cumsum(pe[o]); kk = np.arange(1, pe.size + 1)
    kb = int(np.argmax(cs / (kk + pe.sum() - cs))) + 1; s = np.zeros(pe.size, bool); s[o[:kb]] = True
    s = s.reshape(p.shape); u = ((s | lab) & el).sum()
    return 1.0 if u == 0 else (s & lab & el).sum() / u

for a, b in pairs:
    A = pickle.load(open(W2 + f"oof_{a}.pkl", "rb")); Bo = pickle.load(open(W2 + f"oof_{b}.pkl", "rb"))
    for cond in ("queue_onset", "queue_ongoing"):
        ks = [k for k in range(T.NWIN) if M["is_sel"][k] and M["cond"][k] == cond and k in A and k in Bo]
        d = np.array([bsiou(A[k], *map(lambda x: x.astype(bool), T.LAB[k])) - bsiou(Bo[k], *map(lambda x: x.astype(bool), T.LAB[k])) for k in ks])
        line = f"{a} - {b} [{cond}] n={len(ks)} mean diff {d.mean():+.4f}"
        for lab_, key in (("panel|date", lambda k: f"{M['panel'][k]}|{str(M['date'][k])}"),
                          ("panel|month", lambda k: f"{M['panel'][k]}|{str(M['date'][k])[:7]}"),
                          ("month", lambda k: str(M['date'][k])[:7])):
            cl = np.array([key(k) for k in ks]); u = np.unique(cl); idx = {c: np.flatnonzero(cl == c) for c in u}
            rng = np.random.default_rng(0)
            bm = [d[np.concatenate([idx[c] for c in rng.choice(u, len(u))])].mean() for _ in range(400)]
            line += f" | SE[{lab_}] {np.std(bm):.4f} (z {d.mean()/np.std(bm):+.1f})"
        print(line, flush=True)
    # surprise-weighted onset (t2_eval_surprise's weights)
    ra = np.array([bsiou(A[k], *map(lambda x: x.astype(bool), T.LAB[k])) if k in A else np.nan for k in S.SEL])
    rb = np.array([bsiou(Bo[k], *map(lambda x: x.astype(bool), T.LAB[k])) if k in Bo else np.nan for k in S.SEL])
    d = ra - rb; m = S.stats(d); se = S.boot_se(d)
    print(f"  surprise yardstick {a}-{b}: plain/weighted/f<0.3 = " + "  ".join(f"{x:+.4f}({x/max(e,1e-9):+.1f}SE)" for x, e in zip(m, se)), flush=True)
    del A, Bo
