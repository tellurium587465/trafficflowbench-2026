"""D2: the boundary specialist inside the validation-like simulation (audit/sim npz from c1: perturbed Feb, val-like
blanks, final = ADAPT + gapml + clip + FD0.75).  Selected cells (targets outside the blanks with a mixed released
neighbourhood) get w * specialist + (1-w) * final, optionally followed by FD 0.75 on those cells; density clip last.
Scored like the LB; LB-scaled total = 0.35 dS_state + 0.15 * 0.55 * dS_phys (physics proxy over-reads ~1.8x)."""
import sys, glob, os, copy, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/audit")
from cachelib import load, FAMILY
import t1_features2 as F2, t1_features3 as F3m
from t1_features import kernel_mean
from t1_features3 import grid_features3, rows3, base_of3
from t1_bnd import released, selector, build
from c3_post import P_of, score, fd_v

MD = "C:/Users/user/tfb/work/t1/"; TAG = sys.argv[1] if len(sys.argv) > 1 else "final"
M = {ch: lgb.Booster(model_file=MD + (f"{TAG}_{ch}_final.txt" if TAG.startswith("bnd") else f"bnd_{ch}_{TAG}.txt")) for ch in ("speed", "flow")}
ORIG = F2.link_statics


def own_statics(P):
    V = np.where(P.blank, np.nan, P.V); Qv = np.where(P.blank, np.nan, P.Q)
    ok = np.isfinite(V) & P.E & (V > 0.9 * P.vf[None, :])
    vr = np.where(ok, V / P.vf[None, :], np.nan); vmed = np.nanmedian(vr[::3], axis=0)
    x = np.concatenate([[0.0], np.cumsum(P.length[:-1])]); sub = slice(0, None, 7)
    Q = np.where(ok, Qv, np.nan)[sub]; km, _ = kernel_mean(Q, ok[sub], x, 1.0, [0], 1e9, 1.0)
    return np.nan_to_num(vmed, nan=1.0).astype(np.float32), np.nan_to_num(np.nanmedian(Q / km, axis=0), nan=1.0).astype(np.float32)


WS = (0.25, 0.5, 0.75, 1.0)
rows = []
for f in sorted(glob.glob("C:/Users/user/tfb/audit/sim/*.npz")):
    p = os.path.basename(f)[:-4]; Z = dict(np.load(f)); Ps = P_of(Z)
    P0 = load(p, "train"); feb = np.flatnonzero([s.startswith("2031-02") for s in P0.times]); a, b = feb[0], feb[-1] + 1
    P = copy.copy(P0)
    for k in ("times", "E", "regime", "P"):
        setattr(P, k, getattr(P0, k)[a:b].copy())
    P.V, P.Q, P.blank, P.target, P.T, P._sim = Z["TV"].astype(np.float32), Z["TQ"].astype(np.float32), Z["blank"], Z["target"], b - a, True
    load.cache_clear(); del P0
    st = own_statics(P); fn = lambda panel, split="train": st
    F2.link_statics = fn; F3m.link_statics = fn
    try:
        F, _, _, _ = grid_features3(P)
    finally:
        F2.link_statics = ORIG; F3m.link_statics = ORIG
    bv, bq = base_of3(F)
    V, Q = released(P, P.V, P.Q); sel, cc, ff = selector(P, V)
    cells = P.target & ~Z["gap"] & sel
    r, c = np.nonzero(cells)
    X, _ = build(P, F, V, Q, r, c, cc, ff, rows3); del F
    vb = np.clip((bv[r, c] + M["speed"].predict(X, num_threads=8)) * P.vf[c], 3.0, 1.15 * P.vf[c])
    qb = np.clip((bq[r, c] + M["flow"].predict(X, num_threads=8)) * P.lanes[c], 60.0, None)
    ev = Z["E"][r, c] & np.isfinite(Z["TV"][r, c])
    rm = lambda A, T: float(np.sqrt(np.mean((A[ev] - T[ev]) ** 2)))
    lanes = P.lanes[c]
    rows.append(dict(panel=p, w="rmse", S=rm(Z["V"][r, c], Z["TV"][r, c]), Ph=rm(Z["Q"][r, c] / lanes, Z["TQ"][r, c] / lanes),
                     Sb=rm(vb, Z["TV"][r, c]), Phb=rm(qb / lanes, Z["TQ"][r, c] / lanes), n=int(ev.sum())))
    s0, p0 = score(Z, Ps, Z["V"], Z["Q"]); rows.append(dict(panel=p, w="base", S=s0, Ph=p0))
    for w in WS:
        for post in ("", "+fd"):
            V2, Q2 = Z["V"].copy(), Z["Q"].copy()
            V2[r, c] = w * vb + (1 - w) * Z["V"][r, c]; Q2[r, c] = w * qb + (1 - w) * Z["Q"][r, c]
            if post:
                Vf, Qf = fd_v(Ps, V2, Q2, 0.75); V2[r, c] = Vf[r, c]
            V2 = np.maximum(V2, Q2 / Ps.kj[None, :])
            s, ph = score(Z, Ps, V2, Q2); rows.append(dict(panel=p, w=f"{w}{post}", S=s, Ph=ph))
    print(p, "cells", len(r), {k: round(v, 3) for k, v in rows[-2 * len(WS) - 2].items() if k in ("S", "Ph", "Sb", "Phb")}, flush=True)
d = pd.DataFrame(rows)
rm = d[d.w == "rmse"].set_index("panel")
print("selected cells: RMSE v final -> specialist, q/lane final -> specialist (per panel)")
print(rm[["n", "S", "Sb", "Ph", "Phb"]].rename(columns={"S": "v_final", "Sb": "v_bnd", "Ph": "q_final", "Phb": "q_bnd"}).round(2).to_string())
base = d[d.w == "base"].set_index("panel"); pans = list(base.index); fams = sorted({FAMILY[p] for p in pans})
out = []
for w, g in d[~d.w.isin(["rmse", "base"])].groupby("w"):
    g = g.set_index("panel")
    dS = np.mean([np.mean([g.S[p] - base.S[p] for p in pans if FAMILY[p] == fm]) for fm in fams]); dP = np.mean([g.Ph[p] - base.Ph[p] for p in pans])
    out.append(dict(w=w, dS_state=dS, dS_phys=dP, sim_dTotal=0.35 * dS + 0.15 * dP, LB_scaled=0.35 * dS + 0.15 * 0.55 * dP,
                    panels_better=int(sum((0.35 * (g.S[p] - base.S[p]) + 0.15 * 0.55 * (g.Ph[p] - base.Ph[p])) > 0 for p in pans))))
print(pd.DataFrame(out).round(5).to_string(index=False))
d.to_csv("C:/Users/user/tfb/audit/sim/bnd_sim.csv", index=False)
