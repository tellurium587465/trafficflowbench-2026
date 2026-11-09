import sys, numpy as np
sys.path.insert(0, r"C:/Users/user/tfb/src")
from collections import defaultdict
from t2_eval_oof import T, load_oof
O = load_oof("l127_full_af+q14_attn1_glob+q13_attn1+q14_attn1_wp05")
def dec(p):
    pe = p.ravel(); o = np.argsort(-pe); cs = np.cumsum(pe[o]); kk = np.arange(1, pe.size + 1)
    kb = int(np.argmax(cs / (kk + pe.sum() - cs))) + 1; s = np.zeros(pe.size, bool); s[o[:kb]] = True; return s.reshape(p.shape)
r = defaultdict(list)
for k in O:
    if not T.META["is_sel"][k]: continue
    lab, el = (x.astype(bool) for x in T.LAB[k]); s = dec(O[k]); u = ((s | lab) & el).sum()
    r[(T.META["panel"][k], T.META["cond"][k][6:])].append(1.0 if u == 0 else (s & lab & el).sum() / u)
for key in sorted(r): print(key, len(r[key]), round(np.mean(r[key]), 4))
