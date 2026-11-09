"""R1 audit: onset windows are selected so that |Y| >= 1 at T+30.  Does conditioning the marginals on that event
(per-window rescale) change the expected-IoU decision?  Variants: p/(1-prod(1-p)); min-mass logit shift so sum p >= 1."""
import sys, pickle, numpy as np
sys.path.insert(0, r"C:/Users/user/tfb/src")
import t2_eval_surprise as S
from t2_eval_oof import T, load_oof

spec = sys.argv[1] if len(sys.argv) > 1 else "l127_full_af+q14_attn1_glob+q13_attn1+q14_attn1_wp05"
O = load_oof(spec)

def dec(p):
    pe = p.ravel(); o = np.argsort(-pe); cs = np.cumsum(pe[o]); kk = np.arange(1, pe.size + 1)
    kb = int(np.argmax(cs / (kk + pe.sum() - cs))) + 1; s = np.zeros(pe.size, bool); s[o[:kb]] = True
    return s.reshape(p.shape), kb

def iou(s, lab, el):
    u = ((s | lab) & el).sum(); return 1.0 if u == 0 else (s & lab & el).sum() / u

def cond_scale(p):
    q = p.copy(); r = q[-1]; z = 1 - np.prod(1 - np.clip(r, 0, 1 - 1e-9))
    q[-1] = np.clip(r / max(z, 1e-9), 0, 1); return q

def mass_shift(p, target=1.0):
    q = p.copy(); r = np.clip(q[-1], 1e-6, 1 - 1e-6); lg = np.log(r / (1 - r))
    if r.sum() >= target: return q
    lo, hi = 0.0, 20.0
    for _ in range(40):
        m = (lo + hi) / 2; (lo, hi) = (m, hi) if (1 / (1 + np.exp(-(lg + m)))).sum() < target else (lo, m)
    q[-1] = 1 / (1 + np.exp(-(lg + hi))); return q

rows = []
for k in S.SEL:
    if k not in O: continue
    lab, el = (x.astype(bool) for x in T.LAB[k]); p = O[k]
    r = []
    for f in (lambda x: x, cond_scale, mass_shift):
        s, kb = dec(f(p)); r += [iou(s, lab, el), kb]
    rows.append([k, p[-1].sum()] + r)
R = np.array(rows, float); idx = {k: i for i, k in enumerate(S.SEL)}
sub = np.array([S.SUB[idx[int(k)]] for k in R[:, 0]]); w = np.array([S.W1[idx[int(k)]] for k in R[:, 0]])
print(f"onset sel windows {len(R)}; share with sum p < 1: {(R[:,1] < 1).mean():.3f}; f<0.3 subset {sub.sum()}")
for j, name in ((2, "current"), (4, "cond_scale"), (6, "mass_shift")):
    print(f"{name:10s} plain {R[:,j].mean():.4f}  weighted {np.sum(w*R[:,j])/w.sum():.4f}  f<0.3 {R[sub,j].mean():.4f}  mean k {R[:,j+1].mean():.2f}")
lowm = R[:, 1] < 1
print(f"windows with sum p<1 (n={lowm.sum()}): current {R[lowm,2].mean():.4f} cond_scale {R[lowm,4].mean():.4f} mass_shift {R[lowm,6].mean():.4f}")
