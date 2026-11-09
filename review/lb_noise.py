# Read-only review script: how noisy is an 80-window LB reading of S_queue, absolute and paired?
# Uses the team's virtual-LB selector runs (train months, OOF) - no new training, no writes outside review/.
import os, sys, pickle, numpy as np
sys.path.insert(0, r"C:/Users/user/tfb/src")
os.environ.setdefault("TFB_T2CACHE", "2_noT")
import t2_eval_oof as E
from t2_model import best_set
from t2_windows import iou

pairs = [p.split(",") for p in sys.argv[1:]] or [["l127_fast_base", "l127_fast_af"]]
runs = E.runs; key = E.key; T = E.T


def per_window(spec):
    o = E.load_oof(spec)
    sel = set(key.values())
    o = {k: v for k, v in o.items() if k in sel}
    return {k: iou(best_set(v.ravel(), np.ones(v.size, bool)).reshape(v.shape), T.LAB[k][0], T.LAB[k][1]) for k, v in o.items()}


def agg(ious, ks):
    W = [dict(panel=T.META["panel"][k], cond=T.META["cond"][k]) for k in ks]
    from collections import defaultdict
    pc = defaultdict(list)
    for w, k in zip(W, ks):
        pc[(w["panel"], w["cond"])].append(ious[k])
    panels = sorted({p for p, _ in pc})
    pm = {p: np.mean([np.mean(pc[(p, c)]) for c in ("queue_onset", "queue_ongoing") if (p, c) in pc]) for p in panels}
    fam = defaultdict(list)
    for p, v in pm.items():
        fam[p.rsplit("_", 1)[0]].append(v)
    on = np.mean([np.mean(pc[(p, "queue_onset")]) for p in panels if (p, "queue_onset") in pc])
    og = np.mean([np.mean(pc[(p, "queue_ongoing")]) for p in panels if (p, "queue_ongoing") in pc])
    return np.array([np.mean([np.mean(v) for v in fam.values()]), on, og])


cache = {}
for a, b in pairs:
    for s in (a, b):
        if s not in cache:
            cache[s] = per_window(s)
    A, B = cache[a], cache[b]
    ra, rb, nwin, nchg, dsd = [], [], [], [], []
    for d0, ws in runs.items():
        ks = [key[x] for x in ws if x in key and key[x] in A and key[x] in B]
        if len(ks) < 60:
            continue
        ra.append(agg(A, ks)); rb.append(agg(B, ks)); nwin.append(len(ks))
        d = np.array([B[k] - A[k] for k in ks]); nchg.append((np.abs(d) > 1e-9).mean()); dsd.append(d.std())
    ra, rb = np.array(ra), np.array(rb); dd = rb - ra
    print(f"{a} -> {b}: runs={len(ra)} win/run={np.mean(nwin):.0f}")
    print(f"  mean S_queue A {ra[:,0].mean():.4f} B {rb[:,0].mean():.4f}  diff {dd[:,0].mean():+.4f}")
    print(f"  SD across runs: |A| {ra[:,0].std():.4f} (onset {ra[:,1].std():.4f}, ongoing {ra[:,2].std():.4f})")
    print(f"  SD across runs of paired diff: all {dd[:,0].std():.4f} onset {dd[:,1].std():.4f} ongoing {dd[:,2].std():.4f}")
    print(f"  frac runs with diff>0: {np.mean(dd[:,0]>0):.2f}; per-window changed frac {np.mean(nchg):.2f}, per-window diff SD {np.mean(dsd):.3f}")
    print(f"  => total-score SD of one 80-window paired probe = 0.30*{dd[:,0].std():.4f} = {0.3*dd[:,0].std():.4f}")
