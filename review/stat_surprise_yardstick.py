"""Prototype of a 'calendar-surprise' importance-weighted yardstick for onset (statistician review 10/07).
Weight of an onset window = 1 / f(site | panel, daytype, 2h ToD bin)^gamma, f estimated leaving the window's month out.
Question: does up-weighting windows whose queue site is atypical for the calendar slot reproduce the LB direction
(location prior removal helps) and the val-month overconfidence signature (self E[IoU] >> realized for the prior model)?"""
import sys, os, numpy as np, pandas as pd
from collections import Counter, defaultdict
sys.path.insert(0, r"C:/Users/user/tfb/src"); os.chdir(r"C:/Users/user/tfb/src")
from t2_eval_oof import T, load_oof
sel = [k for k in range(T.NWIN) if T.META["is_sel"][k] and T.META["cond"][k] == "queue_onset"]
ev = [k for k in range(T.NWIN) if (T.META["is_eval"][k] or T.META["is_sel"][k]) and T.META["cond"][k] == "queue_onset"]
def head(k):
    y = T.LAB[k][0][-1]; return int(np.flatnonzero(y).max()) // 3 if y.any() else -1     # site = 3-link bucket
def slot(k):
    ts = pd.Timestamp(str(T.META["date"][k])); h = None
    return (T.META["panel"][k], ts.dayofweek >= 5)
tod = {}
import pickle
# time of day from t index (5-min rows; t % 288)
for k in ev:
    tod[k] = (int(T.META["t"][k]) % 288) // 24                                          # 2-hour bin
cnt = defaultdict(Counter); tot = Counter()
for k in ev:
    key_ = (*slot(k), tod[k]); m = str(T.META["date"][k])[:7]
    cnt[(key_, m)][head(k)] += 1
def f_lomo(k):
    key_ = (*slot(k), tod[k]); m = str(T.META["date"][k])[:7]; h = head(k)
    num = sum(c[h] for (kk, mm), c in cnt.items() if kk == key_ and mm != m)
    den = sum(sum(c.values()) for (kk, mm), c in cnt.items() if kk == key_ and mm != m)
    return (num + 0.5) / (den + 1.0)
F = np.array([f_lomo(k) for k in sel])
print(f"onset windows {len(sel)}; calendar-slot frequency of the true site: median {np.median(F):.2f}, share f<0.3: {np.mean(F<0.3):.3f}")

def bs(o):
    pe = o.ravel(); order = np.argsort(-pe); cs = np.cumsum(pe[order]); kk = np.arange(1, pe.size + 1)
    ei = cs / (kk + pe.sum() - cs); kb = int(np.argmax(ei)) + 1; s = np.zeros(pe.size, bool); s[order[:kb]] = True
    return s.reshape(o.shape), ei[kb - 1]
def iou(s, lab, el):
    u = ((s | lab) & el).sum(); return 1.0 if u == 0 else (s & lab & el).sum() / u
SPECS = {"v8-like l127_noT": "l127_noT", "l127_noT_mp": "l127_noT_mp", "l127_noT_mp_np": "l127_noT_mp_np",
         "LGB_v10 l127_noT_arr": "l127_noT_arr", "LGBfast_npp": "l127_fast_base", "NN_q10(pos)": "q10_raw",
         "NN_q10_nopos": "q10_raw_nopos", "NN_af": "q12_af_r13_nopos", "BLEND_af": "l127_noT_arr+q12_af_r13_nopos"}
R = {}
for n, s in SPECS.items():
    try:
        o = load_oof(s)
    except FileNotFoundError:
        continue
    r = [(iou(*(lambda x: (x[0],))(bs(o[k])), *T.LAB[k]), bs(o[k])[1]) if k in o else (np.nan, np.nan) for k in sel]
    R[n] = np.array(r)
print(f"\n{'model':22s} plain   g=0.5   g=1.0   | f<0.3 subset: realized  self-E[IoU]  gap")
for n, r in R.items():
    out = []
    for g in (0.0, 0.5, 1.0):
        w = 1 / np.maximum(F, 0.05) ** g; ok = np.isfinite(r[:, 0])
        out.append(np.sum(w[ok] * r[ok, 0]) / w[ok].sum())
    m = (F < 0.3) & np.isfinite(r[:, 0])
    print(f"{n:22s} {out[0]:.4f}  {out[1]:.4f}  {out[2]:.4f}  | {r[m,0].mean():.3f}    {r[m,1].mean():.3f}     {r[m,1].mean()-r[m,0].mean():+.3f}")
# effective sample size and TV of site distribution implied by the weights
for g in (0.5, 1.0):
    w = 1 / np.maximum(F, 0.05) ** g; w = w / w.sum()
    print(f"gamma {g}: ESS {1/np.sum(w**2):.0f} of {len(w)}")
