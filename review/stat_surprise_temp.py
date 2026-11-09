"""Does softening/sharpening probabilities before best_set help on the calendar-surprise onset subset (f<0.3)?"""
import sys, os, numpy as np
sys.path.insert(0, r"C:/Users/user/tfb/review"); sys.path.insert(0, r"C:/Users/user/tfb/src")
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import stat_surprise_yardstick as S                     # reuses sel, F, iou, load_oof
from t2_model import best_set
def lg(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
for n, spec in (("LGB_v10", "l127_noT_arr"), ("BLEND_af", "l127_noT_arr+q12_af_r13_nopos"), ("NN_af", "q12_af_r13_nopos"),
                ("v8-like", "l127_noT")):
    o = S.load_oof(spec); row = []
    for a in (0.5, 0.75, 1.0, 1.5):
        r = []
        for k in S.sel:
            p = o[k][-1]; q = 1 / (1 + np.exp(-a * lg(p)))
            s = np.zeros_like(o[k], bool); s[-1] = best_set(q, np.ones(q.size, bool)); r.append(S.iou(s, *S.T.LAB[k]))
        r = np.array(r); row.append(f"a={a}: all {r.mean():.4f} surprise {r[S.F<0.3].mean():.4f} typical {r[S.F>=0.3].mean():.4f}")
    print(f"{n:9s} " + " | ".join(row))
