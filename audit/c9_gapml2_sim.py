"""C9: gapml v1 vs v2 inside the simulated months (c1 npz: perturbed Feb, val-like blanks), both followed by the
density clip and FD 0.75 on the replaced cells, scored like the LB."""
import sys, glob, os, copy, types, numpy as np, lightgbm as lgb, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/audit")
from cachelib import load, T2_PANELS, FAMILY
from t2_demand import profile
import t2_demand
from t2_arrival import _dmin
from t1_gapml import view_train, gap_features, G
from c3_post import P_of, score, fd_v

MD = "C:/Users/user/tfb/work/t1/"
M = {v: {ch: lgb.Booster(model_file=MD + f"{v}_{ch}_cv.txt") for ch in ("speed", "flow")} for v in ("gapml", "gapml2")}
Zc = np.load("C:/Users/user/tfb/work/t2/featcache2_noT.npz", allow_pickle=True)
meta = {k: Zc[k] for k in ("panel", "t", "cond")}; del Zc
rows = []
for f in sorted(glob.glob("C:/Users/user/tfb/audit/sim/*.npz")):
    p = os.path.basename(f)[:-4]; Z = dict(np.load(f)); Ps = P_of(Z)
    if p not in T2_PANELS:
        s, ph = score(Z, Ps, Z["V"], Z["Q"]); rows += [dict(panel=p, v=v, S=s, Ph=ph) for v in ("gapml", "gapml2")]; continue
    P0 = load(p, "train"); feb = np.flatnonzero([s.startswith("2031-02") for s in P0.times]); a, b = feb[0], feb[-1] + 1
    P = copy.copy(P0)
    for k in ("times", "E", "regime"):
        setattr(P, k, getattr(P0, k)[a:b].copy())
    P.V, P.Q, P.blank, P.target, P.T = Z["TV"], Z["TQ"], Z["blank"], Z["target"], b - a
    x = np.concatenate([[0.0], np.cumsum(P.length[:-1])]); dm = _dmin(P, x); prof = profile(P0, "2031-02")
    cond = {int(t) - a: str(c) for pp, t, c in zip(meta["panel"], meta["t"], meta["cond"]) if pp == p}
    for v in ("gapml", "gapml2"):
        V2, Q2 = Z["V"].copy(), Z["Q"].copy()
        for t in Z["wl"]:
            t = int(t); Vw, Qw = view_train(P, t)
            X, _, bv, bq, kk, jj = gap_features(P, Vw, Qw, t, cond[t], prof, dm, x)
            rr = t + kk; tg = P.target[rr, jj] & np.isfinite(bv) & np.isfinite(bq)
            V2[rr[tg], jj[tg]] = np.clip(bv + M[v]["speed"].predict(X, num_threads=8), 3.0, 1.15 * P.vf[jj])[tg]
            Q2[rr[tg], jj[tg]] = np.clip((bq + M[v]["flow"].predict(X, num_threads=8)) * P.lanes[jj], 60.0, None)[tg]
        g = Z["gap"] & Z["target"]
        Vf, Qf = fd_v(Ps, np.maximum(V2, Q2 / Ps.kj[None, :]), Q2, 0.75)
        Vo, Qo = np.where(g, Vf, Z["V"]), np.where(g, Qf, Z["Q"])
        s, ph = score(Z, Ps, Vo, Qo); rows.append(dict(panel=p, v=v, S=s, Ph=ph))
    print(p, [r for r in rows if r["panel"] == p], flush=True)
    load.cache_clear(); t2_demand._PROF.clear()
d = pd.DataFrame(rows).pivot(index="panel", columns="v", values=["S", "Ph"])
pans = list(d.index); fams = sorted({FAMILY[p] for p in pans})
dS = np.mean([np.mean([d.S.gapml2[p] - d.S.gapml[p] for p in pans if FAMILY[p] == f]) for f in fams]); dP = np.mean([d.Ph.gapml2[p] - d.Ph.gapml[p] for p in pans])
print(f"gapml2 vs gapml (sim): dS_state {dS:+.5f}  dS_phys {dP:+.5f}  dTotal {0.35*dS + 0.15*dP:+.5f}")
