"""R1 audit (read-only): (A) persistence of score eligibility, (B) decoder with causal eligibility weights,
(C) window counts per (panel, cond), (D) bootstrap SE with panel|date vs panel|month clusters."""
import sys, pickle, numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, r"C:/Users/user/tfb/src")
from t2_windows import score

W2 = r"C:/Users/user/tfb/work/t2/"
Z = np.load(W2 + "featcache2_noT_af.npz", allow_pickle=True)
M = {k: Z[k] for k in ("panel", "t", "cond", "date", "is_eval", "is_sel")}
LAB = pickle.load(open(W2 + "labels_all2_noT_af.pkl", "rb"))
N = len(M["panel"])
tags = sys.argv[1:] or ["l127_full_af", "q14_attn1_glob", "q13_attn1", "q14_attn1_wp05"]
O = [pickle.load(open(W2 + f"oof_{t}.pkl", "rb")) for t in tags]
keys = set.intersection(*[set(o) for o in O])
B = {k: sum(o[k] for o in O) / len(O) for k in keys}
del O

# history eligibility per window: rows T-12..T-1
EH = {}
for p in sorted(set(M["panel"])):
    E = np.load(rf"C:/Users/user/tfb/work/cache/{p}_train.npz", allow_pickle=True)["E"]
    for k in np.flatnonzero(M["panel"] == p):
        t = int(M["t"][k]); EH[k] = (E[t - 12:t].mean(0), E[t - 1].astype(float))
    del E

def hb(fr, last):
    return np.where(fr == 0, 0, np.where(fr < 0.5, 1, np.where(fr < 1, 2, 3))) * 2 + last.astype(int)

# (A) table P(horizon elig | history bin, step) from training (non-sel) windows
num = np.zeros((8, 6)); den = np.zeros((8, 6))
for k in range(N):
    if M["is_sel"][k]:
        continue
    fr, last = EH[k]; b = hb(fr, last); el = LAB[k][1]
    for s in range(6):
        np.add.at(num[:, s], b, el[s]); np.add.at(den[:, s], b, 1)
TAB = num / np.maximum(den, 1)
print("overall horizon eligibility (train non-sel):", round(num.sum() / den.sum(), 4))
print("bin share:", np.round(den[:, 0] / den[:, 0].sum(), 4))
print("P(elig | bin) at T+5 / T+30:\n", np.round(TAB[:, [0, 5]], 3))

def bs(p, e):
    pe = p.ravel(); ee = e.ravel(); o = np.argsort(-pe)
    nu = np.cumsum(ee[o] * pe[o]); de = np.cumsum(ee[o] * (1 - pe[o])) + (ee * pe).sum()
    kb = int(np.argmax(nu / np.maximum(de, 1e-12))) + 1
    s = np.zeros(pe.size, bool); s[o[:kb]] = True
    return s.reshape(p.shape)

def iou(s, lab, el):
    u = ((s | lab) & el).sum()
    return 1.0 if u == 0 else (s & lab & el).sum() / u

SEL = [k for k in keys if M["is_sel"][k]]
res = defaultdict(list)
for k in SEL:
    p = B[k]; lab, el = LAB[k]; fr, last = EH[k]
    ehat = TAB[hb(fr, last)].T                      # (6, n)
    for name, e in (("current", np.ones_like(p)), ("causal_e", ehat), ("oracle_e", el.astype(float))):
        res[name].append(bs(p, e))
Wv = [dict(panel=M["panel"][k], cond=M["cond"][k], label=LAB[k][0], elig=LAB[k][1]) for k in SEL]
for name in res:
    print(f"{name:10s} S_queue(all,onset,ongoing) on {len(SEL)} sel windows:", np.round(score(Wv, res[name]), 4))

# (D) per-window diff SE: clusters panel|date vs panel|month
d = np.array([iou(a, LAB[k][0], LAB[k][1]) - iou(b, LAB[k][0], LAB[k][1]) for k, a, b in zip(SEL, res["causal_e"], res["current"])])
for lab_, key in (("panel|date", lambda k: f"{M['panel'][k]}|{M['date'][k]}"), ("panel|month", lambda k: f"{M['panel'][k]}|{M['date'][k][:7]}"),
                  ("month", lambda k: M['date'][k][:7])):
    cl = np.array([key(k) for k in SEL]); u = np.unique(cl); idx = {c: np.flatnonzero(cl == c) for c in u}
    rng = np.random.default_rng(0)
    bm = [d[np.concatenate([idx[c] for c in rng.choice(u, len(u))])].mean() for _ in range(400)]
    print(f"diff causal_e - current: mean {d.mean():+.5f}  SE[{lab_}] {np.std(bm):.5f}  clusters {len(u)}")

# (C) window counts per (panel, cond): training pool vs selector-like
tr = Counter((M["panel"][k], M["cond"][k]) for k in range(N) if not M["is_sel"][k])
se = Counter((M["panel"][k], M["cond"][k]) for k in range(N) if M["is_sel"][k])
print("train windows per (panel,cond):", sorted(tr.items()))
print("sel windows per (panel,cond):", sorted(se.items()))
on = sum(v for (p, c), v in tr.items() if c == "queue_onset"); og = sum(v for (p, c), v in tr.items() if c == "queue_ongoing")
print("train onset/ongoing windows:", on, og)
