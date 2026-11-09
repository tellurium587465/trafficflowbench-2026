"""Weekend vs weekday windows: level and paired differences on selector-like OOF windows
(private windows are 17.5% weekend vs 35-40% in validation and in the Saturday-start virtual LBs)."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src"); os.chdir(r"C:/Users/user/tfb/src")
from t2_eval_oof import T, load_oof, runs, key
from t2_model import best_set
sel = [k for k in range(T.NWIN) if T.META["is_sel"][k]]
vlk = {key[x] for ws in runs.values() for x in ws if x in key}
def iou(s, lab, el):
    u = ((s | lab) & el).sum(); return 1.0 if u == 0 else (s & lab & el).sum() / u
I = {}
for n, s in (("LGB", "l127_noT_arr"), ("BLEND", "l127_noT_arr+q12_af_r13_nopos"), ("NN", "q12_af_r13_nopos"),
             ("NNq10", "q10_raw"), ("NNq10np", "q10_raw_nopos"), ("LGBfast", "l127_fast_base")):
    o = load_oof(s); I[n] = {k: iou(best_set(o[k].ravel(), np.ones(o[k].size, bool)).reshape(o[k].shape), *T.LAB[k]) for k in sel if k in o}
d = pd.DataFrame({"k": sel})
d["cond"] = [T.META["cond"][k][6:] for k in sel]
d["wkend"] = [pd.Timestamp(str(T.META["date"][k])).dayofweek >= 5 for k in sel]
d["in_vlb"] = [k in vlk for k in sel]
for n in I: d[n] = [I[n].get(k, np.nan) for k in sel]
print("share weekend among selector-like windows:", d.groupby("cond").wkend.mean().round(3).to_dict(),
      " | inside vLBs:", d[d.in_vlb].groupby("cond").wkend.mean().round(3).to_dict())
print(d.groupby(["cond", "wkend"])[list(I)].mean().round(4).to_string())
for a, c in (("LGB", "BLEND"), ("NN", "BLEND"), ("NNq10", "NNq10np"), ("LGB", "LGBfast")):
    x = d.assign(dd=d[c] - d[a]).groupby(["cond", "wkend"]).dd.agg(["mean", "sem", "size"])
    print(f"\n{a}->{c} paired difference by weekend\n" + x.round(4).to_string())
