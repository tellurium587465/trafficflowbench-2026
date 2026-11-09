"""G29: is s63's LB gain (+0.0031 = ongoing models swapped to the sim-augmented blend) outside the noise of such a swap?
(1) Real OOF, official-like ongoing windows: q36_simR (s3+s4) vs q25_nomem (s3+s4), far gate as s40 for both; an LB
month = 5 ongoing windows per panel from one train month; total diff = 0.30 x mean over panels of 0.5 x ongoing diff.
(2) Twin test worlds (shifted months): the same swap tw2ar_nomem vs tw2_nomem, ongoing windows only, same sampling.
Reports the mean (expected effect) and sd (noise) of the LB-scale difference and the z of +0.0031 under each."""
import sys, os, pickle, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
os.environ.setdefault("TFB_T2_FAR", "3")
import t2_eval_ongoing_shift as O
from t2_eval_oof import T, load_oof, far_mask
per = np.load("E:/tfb_nn/persist_iou_afdem.npy"); OBS = 0.00306


def bestset(p):
    pe = p.ravel(); o = np.argsort(-pe); cs = np.cumsum(pe[o]); kk = np.arange(1, pe.size + 1)
    kb = int(np.argmax(cs / (kk + pe.sum() - cs))) + 1; s = np.zeros(pe.size, bool); s[o[:kb]] = True
    return s.reshape(p.shape)


def sims(d, reps=3000, seed=0):
    rng = np.random.default_rng(seed); out = []
    for _ in range(reps):
        tot = []
        for pn, g in d.groupby("panel"):
            g = g[g.grp == rng.choice(g.grp.unique())]
            if len(g):
                tot.append(0.5 * rng.choice(g.d.to_numpy(), min(5, len(g)), replace=False).mean())
        out.append(0.30 * np.mean(tot))
    return np.array(out)


A = load_oof("q25_nomem_s3+q25_nomem_s4"); B = load_oof("q36_simR_s3+q36_simR_s4"); rows = []
for k in O.SEL:
    if k not in A or k not in B or per[k] > 0.9:
        continue
    lab, el = T.LAB[k]; lab = lab.astype(bool); el = el.astype(bool); fm = far_mask(k); r = []
    for o in (A, B):
        s = bestset(o[k]) & ~fm[None, :]; u = ((s | lab) & el).sum(); r.append(1.0 if u == 0 else (s & lab & el).sum() / u)
    rows.append(dict(panel=str(T.META["panel"][k]), grp=str(T.META["date"][k])[:7], d=r[1] - r[0]))
d = pd.DataFrame(rows); x = sims(d)
print(f"(1) real OOF, in-distribution: LB-scale diff mean {x.mean():+.4f}, sd {x.std():.4f}; z of +{OBS} = {(OBS - x.mean()) / x.std():.1f}")

sys.path.insert(0, r"C:/Users/user/tfb/twin")
LAB = pickle.load(open("E:/tfb_nn/twin_labels.pkl", "rb")); W = "C:/Users/user/tfb/work/t2/"
def pr(spec):
    P = [pickle.load(open(W + f"probs_{t}.pkl", "rb")) for t in spec.split("+")]
    return {k: sum(p[k] for p in P) / len(P) for k in P[0]}
PA, PB = pr("tw2_nomem_s0+tw2_nomem_s1"), pr("tw2ar_nomem_s0+tw2ar_nomem_s1"); rows = []
for wid, (y, el, panel, wd, t, cond, date) in LAB.items():
    if cond != "queue_ongoing":
        continue
    Y = y.T.astype(bool); EL = el.T.astype(bool); r = []
    for P in (PA, PB):
        s = bestset(P[wid]); u = ((s | Y) & EL).sum(); r.append(1.0 if u == 0 else (s & Y & EL).sum() / u)
    rows.append(dict(panel=panel, grp=wd, d=r[1] - r[0]))
d = pd.DataFrame(rows); x = sims(d)
print(f"(2) twin test worlds, shifted months: LB-scale diff mean {x.mean():+.4f}, sd {x.std():.4f}; z of +{OBS} vs 0 = {OBS / x.std():.1f}")
