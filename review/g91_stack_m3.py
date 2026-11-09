"""G91 (10/23, lesson of s79e): the parts measured IN THE PRODUCTION STACK, not alone. M3 v3, base = e4s_a (seeds 0,1; the
family whose X1 exists on M3). Cumulative stack in the production order (scored with the k expectation, simulated blanks,
evidence rule / far gate / decoder of the M3 v2 gate):
  S0 base -> S1 + X1 (ongoing, per k) -> S2 + cone V3 (ongoing) -> S3 + no-memory LightGBM a=0.2 (ongoing, before the cone
  as in production) -> S4 + physics onset (month-adaptive a)
and each part alone on the base (base + cone, base + LightGBM, base + physics, base + TTA) to compare marginal effects.
Order inside the ongoing probability (as t2_build_variant): X1 -> LightGBM mix -> cone factor -> far gate / decoder.
Writes work/t2/g91_stack<ver>.csv; judge the pairs with src/judge.py (outside the hold-out).
usage: TFB_M3VER=v3 python g91_stack_m3.py"""
import os, sys, pickle
os.environ.setdefault("TFB_M3VER", "v3")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.argv = [sys.argv[0], "0,1,2,3"]
import numpy as np, lightgbm as lgb
import t2_m3_gate as G, t2_m3v2_gate as V, m3v2
import t2_phys_onset as PO
from common import NN

W = G.W; VER = G.VER; PRE = G.PRE
ld = lambda f: pickle.load(open(W + f, "rb")); av = lambda a, b: {k: (a[k] + b[k]) / 2 for k in a}
BASE = av(ld(f"probs_{PRE}__e4s_a_s0.pkl"), ld(f"probs_{PRE}__e4s_a_s1.pkl"))
X1 = av(ld(f"probs_m3{VER}v2x1__e4s_a_s0.pkl"), ld(f"probs_m3{VER}v2x1__e4s_a_s1.pkl"))
lgt = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
TD = [av(ld(f"probs_{PRE}tta{k}__e4s_a_s0.pkl"), ld(f"probs_{PRE}tta{k}__e4s_a_s1.pkl")) for k in range(1, 9)]
TTA = {w: np.where((np.stack([d[w] for d in TD]) == 0).all(axis=0), 0.0, 1 / (1 + np.exp(-lgt(np.stack([d[w] for d in TD])).mean(axis=0)))) for w in G.WID}
X = np.load(f"{NN}/nn_test_m3{VER}/X.npy", mmap_mode="r"); OFF = np.load(f"{NN}/nn_test_m3{VER}/off.npy")
phys = lgb.Booster(model_file=W + "physon_base.txt"); ev = PO.month_events("validation", PO.rarity_table())
os.environ.update(TFB_T2_NOT="1", TFB_T2CACHE="2_noT_af", TFB_T2_ARR="1", TFB_T2_DEM="1", TFB_T2_DROP="prior_on,prior_og,pos,vf,capl,lanes,length,w,tod,dow")
import t2_fast as TF
LGO = lgb.Booster(model_file=W + "lgbnm_queue_ongoing.txt")
src = open(r"C:/Users/user/tfb/review/g55_m3_x7_shock.py", encoding="utf-8").read().split("\nrows, st = [], []")[0]
g55 = type(sys)("g55"); exec(compile(src, "g55", "exec"), g55.__dict__)


def cone_f(m, k):
    _, cone, _ = g55.parts(m["window_id"])
    P, _, q = m3v2._panel(m["panel"]); t = m["t"]; d0 = t - (t % 288) + 60; hid = V._hidden_today(m, k)
    rr = [r for r in range(d0, t) if r not in hid]
    ev_ = q[rr].any(axis=0) if rr else np.zeros(P.n, bool)
    s2, _ = m3v2.seen(m["panel"], t, k, -18, 12)
    return np.where(G.dil(ev_ | s2, 2)[None, :], 1.0, cone)


VARS = {"S0": dict(), "S1": dict(x1=1), "S2": dict(x1=1, cone=1), "S3": dict(x1=1, lgb=1, cone=1), "S4": dict(x1=1, lgb=1, cone=1, phys=1),
        "b_cone": dict(cone=1), "b_lgb": dict(lgb=1), "b_phys": dict(phys=1), "b_tta": dict(tta=1)}
cols = {v: {} for v in VARS}
for i, w in enumerate(G.WID):
    m = G.META[w]; Xw = np.asarray(X[OFF[i]:OFF[i + 1]]); og = m["cond"] == "queue_ongoing"
    if og:
        Xd = Xw.copy(); Xd[:, TF.I_PON] = 0.0; Xd[:, TF.I_POG] = 0.0; Xd = TF.apply_drop(Xd)
        pl = LGO.predict(np.vstack([TF.step_block(Xd, s + 1) for s in range(6)]), num_threads=8).reshape(6, len(Xd))
    else:
        ph = phys.predict(PO.feats(Xw)); a_ph = PO.adaptive_a(ev, m["day"])
    for v, o in VARS.items():
        def one(k):
            if og:
                p = X1[(w, k)] if o.get("x1") else (TTA[w] if o.get("tta") else BASE[w])
                if o.get("lgb"):
                    p = 0.8 * p + 0.2 * pl
                if o.get("cone"):
                    p = p * cone_f(m, k)
                return G.iou(V.ongoing_pred(p, m, True, k), m)
            p = BASE[w].copy()
            if o.get("phys"):
                p[5] = (1 - a_ph) * p[5] + a_ph * ph
            return G.iou(V.onset_pred(p, m, "cur", k), m)
        cols[v][w] = sum(pr * one(k) for k, pr in m3v2.kdist(m["day"]).items())
D = V.frame(cols); D.to_csv(W + f"g91_stack{VER}.csv", index=False); print("wrote", len(D))
