"""G93 (10/23, METAGAME_REPLY §23 / §24): the final A's April parts measured in the PRODUCTION ORDER ON B's BASE (M3 v3,
the k expectation with the simulated blanks, the evidence rule / far gate / decoder of the M3 v2 gate). B's base:
onset = q24f 8 seeds (CFG5 weights), ongoing = c24f attnglob seeds 0, 1 + X1 (M3 v3 X1 per k; the production uses all 20
c24f models). A's parts in the production order of t2_final_spec.window_probs:
  ongoing: X1 -> LightGBM 0.2 (no-memory, before the cone) -> cone V3 -> far gate / decoder
  onset  : onset shift (c = 0.5, beta = 0: logit + 0.5 where an earlier-day pseudo-origin b <= k observed the cell, the
           simulated blanks of (day, k) hidden) -> physics onset (month-adaptive a, physon_base) -> evidence rule / decoder
Columns: B, lgb, cone, lgb_cone (= A on ongoing), shift, phys (= A on onset), shift_phys, A (per window the final A).
A's base E4 cannot be measured on M3 (it learns March). Writes work/t2/g93_stack_b<ver>_<ongoing family>.csv; judge with src/judge.py.
usage: TFB_M3VER=v3 python g93_stack_b.py"""
import os, sys, pickle
os.environ.setdefault("TFB_M3VER", "v3")
sys.path.insert(0, r"C:/Users/user/tfb/src"); argv0 = sys.argv[0]
import numpy as np, lightgbm as lgb
import t2_phys_onset as PO, t2_final_spec as FS, t2_e3_official as E3
sys.argv = [argv0, "0,1,2,3"]
import t2_m3_gate as G, t2_m3v2_gate as V, m3v2
from common import NN

W = G.W; VER = G.VER; PRE = G.PRE
ld = lambda f: pickle.load(open(W + f, "rb"))
CF5 = (("attnglob", 2.0), ("attnwp05", 2.0), ("attn1", 2.0), ("drop1", 1.0), ("r13", 1.0))
ON = [(f"q24f_{c}_s{s}", w) for s in range(8) for c, w in CF5]; PRO = {t: ld(f"probs_{PRE}__{t}.pkl") for t, _ in ON}
OGF = os.environ.get("G93_OG", "c24f_attnglob")                # e4s_a = the G91 base (code check before c24f exists)
XA = [ld(f"probs_m3{VER}v2x1__{OGF}_s{s}.pkl") for s in (0, 1)]
X1 = {key: (XA[0][key] + XA[1][key]) / 2 for key in XA[0]}
X = np.load(f"{NN}/nn_test_m3{VER}/X.npy", mmap_mode="r"); OFF = np.load(f"{NN}/nn_test_m3{VER}/off.npy")
phys = lgb.Booster(model_file=W + "physon_base.txt"); ev = PO.month_events("validation", PO.rarity_table())
PM = ld(f"m3{VER}prev6_meta.pkl"); BYP = {}
for e in PM:
    BYP.setdefault(e["parent"], []).append(e)
src = open(r"C:/Users/user/tfb/review/g55_m3_x7_shock.py", encoding="utf-8").read().split("\nrows, st = [], []")[0]
g55 = type(sys)("g55"); exec(compile(src, "g55", "exec"), g55.__dict__)


def cone_f(m, k):                                               # as G91
    _, cone, _ = g55.parts(m["window_id"])
    P, _, q = m3v2._panel(m["panel"]); t = m["t"]; d0 = t - (t % 288) + 60; hid = V._hidden_today(m, k)
    rr = [r for r in range(d0, t) if r not in hid]
    ev_ = q[rr].any(axis=0) if rr else np.zeros(P.n, bool)
    s2, _ = m3v2.seen(m["panel"], t, k, -18, 12)
    return np.where(G.dil(ev_ | s2, 2)[None, :], 1.0, cone)


def shift(p, w, k):
    """the onset shift of t2_e3_official (a, beta, c) = (1, 0, 0.5) with the pseudo-origins b <= k, blanks of (day, k) hidden"""
    m = G.META[w]; hid = m3v2.sim_blank_rows(m["panel"], m["day"], k, m["t"]); ko = None
    for e in BYP.get(w, []):
        if e["b"] > k:
            continue
        keep = np.array([r not in hid for r in range(e["tp"] + 1, e["tp"] + 7)])[None, :]
        mo = (e["m"].astype(float) * keep).T; ko = mo if ko is None else ko + mo
    return p if ko is None else E3.correct(p, np.zeros_like(ko), ko, np.zeros_like(ko), *E3.CONST["queue_onset"])


VARS = {"B": (), "lgb": ("lgb",), "cone": ("cone",), "lgb_cone": ("lgb", "cone"), "shift": ("shift",), "phys": ("phys",), "shift_phys": ("shift", "phys")}
cols = {v: {} for v in list(VARS) + ["A"]}
for i, w in enumerate(G.WID):
    m = G.META[w]; Xw = np.asarray(X[OFF[i]:OFF[i + 1]]); og = m["cond"] == "queue_ongoing"
    if og:
        pl = FS.lgb_ongoing(Xw)
    else:
        pn = sum(wt * PRO[t][w] for t, wt in ON) / sum(wt for _, wt in ON); ph = phys.predict(PO.feats(Xw)); a_ph = PO.adaptive_a(ev, m["day"])
    for v, o in VARS.items():
        if og != (v in ("B", "lgb", "cone", "lgb_cone")) and v != "B":
            continue
        def one(k):
            if og:
                p = X1[(w, k)]
                if "lgb" in o:
                    p = 0.8 * p + 0.2 * pl
                if "cone" in o:
                    p = p * cone_f(m, k)
                return G.iou(V.ongoing_pred(p, m, True, k), m)
            p = shift(pn, w, k) if "shift" in o else pn
            if "phys" in o:
                p = p.copy(); p[5] = (1 - a_ph) * p[5] + a_ph * ph
            return G.iou(V.onset_pred(p, m, "cur", k), m)
        cols[v][w] = sum(pr * one(k) for k, pr in m3v2.kdist(m["day"]).items())
    cols["A"][w] = cols["lgb_cone" if og else "phys"][w]       # the final A (10/23 07:00: no onset shift, ledger #28)
D = V.frame(cols); D.to_csv(W + f"g93_stack_b{VER}_{OGF}.csv", index=False); print("wrote", len(D), "windows")
