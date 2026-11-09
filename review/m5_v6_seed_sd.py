"""M5 audit V6: is the seed sd constant of t2_seedcmp (SIG, measured 10/08 without the far rule) still right under
TFB_T2_FAR=3? Pooled single-run sd from seed pairs (s3, s4) of the same config, virtual LB all/onset/ongoing,
with and without the far rule. Also scans the OOFs for non-finite values."""
import sys, time, pickle, numpy as np
sys.path.insert(0, "C:/Users/user/tfb/src")
import t2_eval_oof as E
from t2_model import best_set
from t2_windows import score
CFG = ["q21_base", "q21_gbal2", "q21_gbal3", "q21_tjit", "q21_wd05", "q22_ognear", "q23_base", "q23_notime", "q23_nostat",
       "q23_nomem", "q25_nomem", "q25_rstat", "q25_tok", "q25_crop"]


def vlb_both(tag):
    o = pickle.load(open(f"C:/Users/user/tfb/work/t2/oof_{tag}.pkl", "rb"))
    nonfin = sum((~np.isfinite(v)).any() for v in o.values())
    bs = {k: best_set(v.ravel(), np.ones(v.size, bool)).reshape(v.shape) for k, v in o.items()}
    E.FAR = 3
    bf = {}
    for k, s in bs.items():
        m = E.far_mask(k); bf[k] = s & ~m[None, :] if m is not None else s
    out = []
    for b in (bs, bf):
        vs = []
        for d0, ws in E.runs.items():
            ks = [E.key[x] for x in ws if x in E.key and E.key[x] in o]
            if len(ks) < 60:
                continue
            vs.append(score([E._W(k) for k in ks], [b[k] for k in ks]))
        out.append(np.mean(vs, axis=0))
    return out[0], out[1], nonfin


t0 = time.time(); R = {}
for c in CFG:
    for s in ("s3", "s4"):
        R[c + "_" + s] = vlb_both(c + "_" + s)
    a, b = R[c + "_s3"], R[c + "_s4"]
    print(f"{c:12s} FAR0 {np.round(a[0], 4)} {np.round(b[0], 4)} | FAR3 {np.round(a[1], 4)} {np.round(b[1], 4)} | non-finite windows {a[2]}/{b[2]} ({time.time()-t0:.0f}s)", flush=True)
for j, nm in ((0, "FAR=0"), (1, "FAR=3")):
    d = np.array([R[c + "_s3"][j] - R[c + "_s4"][j] for c in CFG])
    sd = np.sqrt((d ** 2).mean(0) / 2)
    print(f"pooled single-run seed sd {nm}: vlb {sd[0]:.4f} onset {sd[1]:.4f} ongoing {sd[2]:.4f}  (n pairs {len(CFG)}; SIG in t2_seedcmp: vlb 0.0023 on 0.0039 og 0.0014)")
