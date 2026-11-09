"""Twin stage 3 evaluation: score models trained in world A on the twin test worlds, with the submission's post-
processing (same definitions as t2_onset_evidence.py / t2_far_gate.py: evidence from the masked layer = finite and not
a Task 1 blank, at most 3 earlier days of the same month). Ongoing windows are kept only if the exact persistence IoU
at T is <= 0.9 (the official rule). Paired comparison of two specs with a cluster bootstrap over (panel, world).

usage: python eval_twin.py <specA> [<specB>] [--rules B,OGB,GATE,CONTIG] [--days 4]
  spec = 'tag1+tag2:2' (probs_<tag>.pkl written by the FINAL runs on nn_test_twinT); rules default 1,1,0,0 = raw
"""
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load

args, opts, i = [], {}, 1
while i < len(sys.argv):
    if sys.argv[i].startswith("--"):
        opts[sys.argv[i]] = sys.argv[i + 1]; i += 2
    else:
        args.append(sys.argv[i]); i += 1
RULES = [float(x) for x in opts.get("--rules", "1,1,0,0").split(",")]; DAYS = int(opts.get("--days", "31"))
LAB = pickle.load(open(opts.get("--labels", "E:/tfb_nn/twin_labels.pkl"), "rb"))
W = "C:/Users/user/tfb/work/t2/"; PC = {}; NAMES = None


def probs(spec):
    parts = [(t.split(":")[0], float(t.split(":")[1]) if ":" in t else 1.0) for t in spec.split("+")]
    P = [(pickle.load(open(W + f"probs_{t}.pkl", "rb")), w) for t, w in parts]
    keys = set.intersection(*[set(p) for p, _ in P]); tw = sum(w for _, w in P)
    return {k: sum(w * p[k] for p, w in P) / tw for k in keys}


def world(panel, wd):
    key = (panel, wd)
    if key not in PC:
        P = load(panel, f"twin{wd}")
        q = (np.nan_to_num(P.V, nan=1e9) <= P.vcut[None, :]) & np.isfinite(P.V)
        obs = q & ~P.blank                                                          # the released masked layer
        full = (np.nan_to_num(P.V, nan=1e9) <= P.vcut[None, :]) & P.E
        PC[key] = (P, obs, full)
    return PC[key]


def bestset(p):
    pe = p.ravel(); o = np.argsort(-pe); cs = np.cumsum(pe[o]); kk = np.arange(1, pe.size + 1)
    kb = int(np.argmax(cs / (kk + pe.sum() - cs))) + 1; s = np.zeros(pe.size, bool); s[o[:kb]] = True
    return s.reshape(p.shape)


def dil(x, d):
    e = x.copy()
    for o in range(1, d + 1):
        e[o:] |= x[:-o]; e[:-o] |= x[o:]
    return e


def score(pr, rules):
    B, OGB, GATE, CONTIG = rules; out = {}
    for wid, (y, el, panel, wd, t, cond, date) in LAB.items():
        if wid not in pr or int(date[8:10]) > DAYS:
            continue
        P, obs, full = world(panel, wd)
        Y = y.T.astype(bool); EL = el.T.astype(bool); p = pr[wid].copy()
        if cond == "queue_ongoing":
            per = full[t]; fut = Y & EL; pp = np.broadcast_to(per, fut.shape) & EL; u = (pp | fut).sum()
            if u and (pp & fut).sum() / u > 0.9:
                continue                                                            # not an official ongoing window
        seen_on = np.zeros(P.n, bool); seen_og = np.zeros(P.n, bool); nd = 0
        for b in (1, 2, 3):
            dd = t - b * 288
            if dd - 18 >= 0:
                seen_on |= obs[dd - 6:dd + 12].any(axis=0); seen_og |= obs[dd - 18:dd + 12].any(axis=0); nd += 1
        if cond == "queue_onset":
            p[:-1] = 0.0
            if nd and B != 1:
                p[-1] = p[-1] * np.where(dil(seen_on, 2), 1.0, B)
            s = bestset(p)
        else:
            if nd and OGB != 1:
                p = p * np.where(dil(seen_og, 1), 1.0, OGB)[None, :]
            s = bestset(p)
            if GATE:
                d0 = t - (t % 288) + 60
                gs = seen_og | (obs[d0:t].any(axis=0) if t > d0 else False); gev = dil(gs, 2)
                lq = full[t - 1]                                                    # last history row (released unmasked)
                near = np.zeros(P.n, bool)
                for j in np.flatnonzero(lq):
                    near[max(0, j - 3):j + 4] = True
                drop = ~near & ~gev if GATE == 1 else ~near                       # GATE=2: blanket removal (s34)
                if CONTIG:
                    keep = np.zeros_like(s)
                    for r in range(s.shape[0]):
                        j = 0
                        while j < P.n:
                            if s[r, j]:
                                e = j
                                while e + 1 < P.n and s[r, e + 1]:
                                    e += 1
                                if near[j:e + 1].any():
                                    keep[r, j:e + 1] = True
                                j = e + 1
                            else:
                                j += 1
                    s = s & ~(drop[None, :] & ~keep)
                else:
                    s = s & ~drop[None, :]
        u = ((s | Y) & EL).sum()
        out[wid] = (1.0 if u == 0 else (s & Y & EL).sum() / u, panel, wd, cond)
    return out


def summary(r):
    d = pd.DataFrame([(k, *v) for k, v in r.items()], columns=["wid", "iou", "panel", "world", "cond"])
    pc = d.groupby(["panel", "cond"]).iou.mean().unstack()
    return d, pc["queue_onset"].mean(), pc["queue_ongoing"].mean()


def boot(d, reps=500, seed=0):
    rng = np.random.default_rng(seed); cl = (d.panel + "|" + d.world).to_numpy(); u = np.unique(cl)
    idx = {c: np.flatnonzero(cl == c) for c in u}; out = []
    for _ in range(reps):
        ii = np.concatenate([idx[c] for c in rng.choice(u, len(u))]); x = d.iloc[ii]
        out.append([x[x.cond == "queue_onset"].diff_.mean(), x[x.cond == "queue_ongoing"].diff_.mean()])
    return np.std(out, axis=0)


A = score(probs(args[0]), RULES); dA, onA, ogA = summary(A)
print(f"A {args[0]} rules {RULES} days<={DAYS}: onset {onA:.4f}  ongoing {ogA:.4f}  (windows {len(dA)}: "
      f"{(dA.cond == 'queue_onset').sum()} onset / {(dA.cond == 'queue_ongoing').sum()} ongoing)")
if len(args) > 1:
    rB = RULES if "--rulesB" not in opts else [float(x) for x in opts["--rulesB"].split(",")]
    Bs = score(probs(args[1]), rB); dB, onB, ogB = summary(Bs)
    m = dA.merge(dB[["wid", "iou"]], on="wid", suffixes=("", "_b")); m["diff_"] = m.iou_b - m.iou
    se = boot(m)
    print(f"B {args[1]} rules {rB}: onset {onB:.4f} ({onB - onA:+.4f}, SE {se[0]:.4f})  ongoing {ogB:.4f} ({ogB - ogA:+.4f}, SE {se[1]:.4f})")
    # LB-scale noise: one test month = 5 onset + 5 ongoing windows per panel from ONE world; total-score difference
    # = 0.30 x mean over panels of (onset + ongoing)/2 differences (the organizer's aggregation, family level ignored)
    rng = np.random.default_rng(1); sims = []
    for _ in range(2000):
        tot = []
        for pn, g in m.groupby("panel"):
            wds = g.world.unique(); g = g[g.world == rng.choice(wds)]
            on = g[g.cond == "queue_onset"].diff_.to_numpy(); og = g[g.cond == "queue_ongoing"].diff_.to_numpy()
            if len(on) and len(og):
                tot.append(0.5 * (rng.choice(on, min(5, len(on)), replace=False).mean() + rng.choice(og, min(5, len(og)), replace=False).mean()))
        sims.append(0.30 * np.mean(tot))
    print(f"   LB-scale (one month, 5+5 windows/panel): total-score diff mean {np.mean(sims):+.4f}, sd {np.std(sims):.4f}, P(diff>0) {np.mean(np.array(sims) > 0):.2f}")
