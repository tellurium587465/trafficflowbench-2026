"""Nested choice of the onset evidence strength B on the twin test worlds (P8: no winner's curse). Worlds split in two
halves (T0-T3 / T4-T7); B chosen on one half by mean onset IoU, scored on the other; reported against B = 0.5 (s52)
with a cluster bootstrap over (panel, world). Same evidence definition as t2_onset_evidence.py (masked layer, <= 3
earlier days, slot T-30..T+60 min, +-2 links).
usage: python nested_B.py <spec>"""
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load

SPEC = sys.argv[1]; BS = (1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.05)
LAB = pickle.load(open("E:/tfb_nn/twin_labels.pkl", "rb")); W = "C:/Users/user/tfb/work/t2/"
parts = [(t.split(":")[0], float(t.split(":")[1]) if ":" in t else 1.0) for t in SPEC.split("+")]
PR = [(pickle.load(open(W + f"probs_{t}.pkl", "rb")), w) for t, w in parts]; tw = sum(w for _, w in PR); PC = {}


def bestset(p):
    pe = p.ravel(); o = np.argsort(-pe); cs = np.cumsum(pe[o]); kk = np.arange(1, pe.size + 1)
    kb = int(np.argmax(cs / (kk + pe.sum() - cs))) + 1; s = np.zeros(pe.size, bool); s[o[:kb]] = True
    return s.reshape(p.shape)


rows = []
for wid, (y, el, panel, wd, t, cond, date) in LAB.items():
    if cond != "queue_onset":
        continue
    if (panel, wd) not in PC:
        P = load(panel, f"twin{wd}"); q = (np.nan_to_num(P.V, nan=1e9) <= P.vcut[None, :]) & np.isfinite(P.V); PC[(panel, wd)] = (P, q & ~P.blank)
    P, obs = PC[(panel, wd)]
    p0 = sum(w * p[wid] for p, w in PR) / tw; p0 = p0.copy(); p0[:-1] = 0
    seen = np.zeros(P.n, bool); nd = 0
    for b in (1, 2, 3):
        dd = t - b * 288
        if dd - 6 >= 0:
            seen |= obs[dd - 6:dd + 12].any(axis=0); nd += 1
    ev = seen.copy()
    for o in (1, 2):
        ev[o:] |= seen[:-o]; ev[:-o] |= seen[o:]
    Y = y.T.astype(bool); EL = el.T.astype(bool); r = dict(panel=panel, world=wd, day=int(date[8:10]))
    for B in BS:
        p = p0.copy()
        if nd:
            p[-1] = p[-1] * np.where(ev, 1.0, B)
        s = bestset(p); u = ((s | Y) & EL).sum(); r[B] = 1.0 if u == 0 else (s & Y & EL).sum() / u
    rows.append(r)
D = pd.DataFrame(rows); D["half"] = D.world.str[1:].astype(int) // 4
print("in-sample mean onset IoU by B:", {B: round(D[B].mean(), 4) for B in BS})
held = np.zeros(len(D)); chosen = []
for h in (0, 1):
    tr, te = D.half != h, D.half == h
    b = max(BS, key=lambda B: D.loc[tr, B].mean()); chosen.append(b); held[te.to_numpy()] = D.loc[te, b].to_numpy()
d = held - D[0.5].to_numpy(); cl = (D.panel + "|" + D.world).to_numpy(); u = np.unique(cl); idx = {c: np.flatnonzero(cl == c) for c in u}
rng = np.random.default_rng(0); se = np.std([d[np.concatenate([idx[c] for c in rng.choice(u, len(u))])].mean() for _ in range(1000)])
print(f"nested: chosen {chosen}, held-out - B0.5 {d.mean():+.4f} (SE {se:.4f}); days<=4 {d[D.day.to_numpy() <= 4].mean():+.4f}")
