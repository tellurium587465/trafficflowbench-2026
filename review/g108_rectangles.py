"""G108 (10/29, strategy lane 11, ledger #49): the generator's jams are per-site rectangles (audit/b13, b13b, b13c: head
fixed, birth tail = the site's modal tail in 96% of fully seen births, whole-run fates). Two post-processes on the final A
flow of G93 (M3 v3, k expectation with the simulated blanks):
  O1 (onset)  : each run decoded at T+30 whose head is within +-1 of a known site head (work/b13_sites.parquet, training
                months only) -> [modal tail, site head].
  G1 / G2     : ongoing, after X1 -> LightGBM 0.2 -> cone -> evidence weight and before best_set, each contiguous run of
                last_queue (the X column far_gate uses, rows <= T) gets per horizon row the run's mean probability (G1) or
                its head link's probability (G2). Outside the runs and the far gate: unchanged.
Identity: the unchanged columns reproduce G93's lgb_cone (ongoing) and phys (onset) per window.
Diagnostics: share of (visible run, horizon row) pairs the base decode selects partially (k-weighted), and the label fates
of the same pairs at T+30 (dev windows).
Writes work/t2/g108_rectangles.csv for judge.py --cond.
usage: TFB_M3VER=v3 py g108_rectangles.py"""
import os, sys, pickle
os.environ.setdefault("TFB_M3VER", "v3")
sys.path.insert(0, r"C:/Users/user/tfb/src"); argv0 = sys.argv[0]
import numpy as np, pandas as pd, lightgbm as lgb
import t2_phys_onset as PO, t2_final_spec as FS
sys.argv = [argv0, "0,1,2,3"]
import t2_m3_gate as G, t2_m3v2_gate as V, m3v2
from t2_model import best_set
from common import NN, WORK

W = G.W; VER = G.VER; PRE = G.PRE
ld = lambda f: pickle.load(open(W + f, "rb"))
CF5 = (("attnglob", 2.0), ("attnwp05", 2.0), ("attn1", 2.0), ("drop1", 1.0), ("r13", 1.0))
ON = [(f"q24f_{c}_s{s}", w) for s in range(8) for c, w in CF5]; PRO = {t: ld(f"probs_{PRE}__{t}.pkl") for t, _ in ON}
XA = [ld(f"probs_m3{VER}v2x1__c24f_attnglob_s{s}.pkl") for s in (0, 1)]
X1 = {key: (XA[0][key] + XA[1][key]) / 2 for key in XA[0]}
phys = lgb.Booster(model_file=W + "physon_base.txt"); ev = PO.month_events("validation", PO.rarity_table())
src = open(r"C:/Users/user/tfb/review/g55_m3_x7_shock.py", encoding="utf-8").read().split("\nrows, st = [], []")[0]
g55 = type(sys)("g55"); exec(compile(src, "g55", "exec"), g55.__dict__)
S = pd.read_parquet(WORK + "/b13_sites.parquet"); SITE = {(r.panel, int(r.head)): int(r.tail) for r in S.itertuples()}
REF = pd.read_csv(W + f"g93_stack_b{VER}_c24f_attnglob.csv").set_index("w")


def cone_f(m, k):                                               # as G93
    _, cone, _ = g55.parts(m["window_id"])
    P, _, q = m3v2._panel(m["panel"]); t = m["t"]; d0 = t - (t % 288) + 60; hid = V._hidden_today(m, k)
    rr = [r for r in range(d0, t) if r not in hid]
    ev_ = q[rr].any(axis=0) if rr else np.zeros(P.n, bool)
    s2, _ = m3v2.seen(m["panel"], t, k, -18, 12)
    return np.where(G.dil(ev_ | s2, 2)[None, :], 1.0, cone)


def runs(v):
    dd = np.diff(np.r_[0, np.asarray(v, int), 0]); return list(zip(np.flatnonzero(dd == 1), np.flatnonzero(dd == -1) - 1))


def og_pred(p_, m, k, mode, R):                                 # V.ongoing_pred with the run step before best_set
    s, nd = m3v2.seen(m["panel"], m["t"], k, -18, 12)
    pb = p_ * np.where(G.dil(s, 1), 1.0, 0.85)[None, :] if nd else p_.copy()
    if mode:
        pb = pb.copy()
        for a, b in R:
            pb[:, a:b + 1] = pb[:, a:b + 1].mean(1, keepdims=True) if mode == "mean" else pb[:, b:b + 1]
    sel = best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape)
    return V.far_gate(sel, m, k)


def snap(sel, panel):
    v = sel[-1]; out = v.copy()
    for a, b in runs(v):
        h = next((hh for hh in (b, b - 1, b + 1) if (panel, hh) in SITE), None)
        if h is not None:
            out[a:b + 1] = False; out[SITE[(panel, h)]:h + 1] = True
    s2 = sel.copy(); s2[-1] = out; return s2


cols = {c: {} for c in ("A", "O1", "G1", "G2")}; dmax = 0.0
diag = np.zeros(3); fate = np.zeros(3); nsnap = 0
for i, w in enumerate(G.WID):
    m = G.META[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); og = m["cond"] == "queue_ongoing"
    dev = not m3v2.is_holdout(np.array([m["day"]]))[0]
    if og:
        pl = FS.lgb_ongoing(Xw); R = runs(np.asarray(Xw[:, G.CI], float) == 1)
        sc = {c: 0.0 for c in ("A", "G1", "G2")}
        for k, pr in m3v2.kdist(m["day"]).items():
            p = (0.8 * X1[(w, k)] + 0.2 * pl) * cone_f(m, k)
            for c, mode in (("A", None), ("G1", "mean"), ("G2", "head")):
                sel = og_pred(p, m, k, mode, R); sc[c] += pr * G.iou(sel, m)
                if c == "A" and dev:
                    for a, b in R:
                        f = sel[:, a:b + 1].mean(1); diag += pr * np.array([(f == 0).sum(), ((f > 0) & (f < 1)).sum(), (f == 1).sum()])
        for c in sc:
            cols[c][w] = sc[c]
        if dev:
            y, mk = m["y"].astype(bool), m["m"].astype(bool)
            for a, b in R:
                o = mk[a:b + 1, 5]
                if o.sum() >= 2:
                    f = y[a:b + 1, 5][o].mean(); fate += [f == 0, 0 < f < 1, f == 1]
    else:
        pn = sum(wt * PRO[t][w] for t, wt in ON) / sum(wt for _, wt in ON); ph = phys.predict(PO.feats(Xw)); a_ph = PO.adaptive_a(ev, m["day"])
        sa = so = 0.0
        for k, pr in m3v2.kdist(m["day"]).items():
            p = pn.copy(); p[5] = (1 - a_ph) * p[5] + a_ph * ph
            sel = V.onset_pred(p, m, "cur", k); s2 = snap(sel, m["panel"]); nsnap += int((s2 != sel).any())
            sa += pr * G.iou(sel, m); so += pr * G.iou(s2, m)
        cols["A"][w] = sa; cols["O1"][w] = so
    ref = REF.loc[w, "lgb_cone" if og else "phys"]; dmax = max(dmax, abs(cols["A"][w] - ref))
print(f"identity: unchanged columns vs G93 (lgb_cone / phys): max |d| {dmax:.2e}")
print("dev ongoing, base decode on (visible run, horizon row) pairs, k-weighted: none %.3f | partial %.3f | all %.3f" % tuple(diag / diag.sum()))
print("dev ongoing, label fate of visible runs at T+30 (>= 2 observed links): all gone %.3f | partial %.3f | all left %.3f" % tuple(fate / fate.sum()))
print("onset (window, k) decodes changed by the snap:", nsnap)
D = V.frame(cols); D.to_csv(W + "g108_rectangles.csv", index=False)
dv = ~m3v2.is_holdout(D.day)
for cond, cs in (("queue_onset", ("A", "O1")), ("queue_ongoing", ("A", "G1", "G2"))):
    d = D[dv & (D.cond == cond)]
    print(cond, len(d), " | ".join(f"{c} {d[c].mean():.4f}" for c in cs))
