"""G315 (11/05, ledger #250, ongoing): a scenario-joint decode against best_set. Each candidate is one scenario of the queue: a contiguous
link range [a, b] held from the first horizon row through row E (E = 1 .. 6; 6 = it does not end in the window); its expected IoU under the
marginals is approximated as best_set does, A / (|S| + P - A) with A = the probability mass inside the set and P = the total mass; the
empty set is a candidate with score 0. The argmax replaces best_set; the far gate and everything else unchanged. M3 v3 dev ongoing (G284's
dump = A17's March mix, temperature 0.8, G132's gate, m3v2.summary) and January / February (G285's capB0 flow, temperature 0.8 as in
G287, G119's far gate, cluster bootstrap). Line (#247-#252): both points > 0 and merged z >= +1.
usage: TFB_M3VER=v3 py -B -X utf8 g315_scenario_decode.py"""
import os, sys, pickle, importlib.util
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
R = r"C:/Users/user/tfb/"; sys.path.insert(0, R + "src")
import t2_m3_gate as G, m3v2
from t2_model import best_set
W = R + "work/t2/"
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))); tmp = lambda p, T: 1 / (1 + np.exp(-lg(p) / T))
bs = lambda pb: best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape)


def scen(p, th=0.05):
    """p (6, n) -> one scenario per box: the boxes are the maximal runs of links whose largest horizon probability exceeds th; each box
    gets the rectangle rows 0 .. E-1 x links a .. b inside it (or nothing) that maximises the window's A / (|S| + P - A), by coordinate
    ascent over the boxes (two sweeps), the other boxes held"""
    F, n = p.shape; P = float(p.sum()); hot = p.max(axis=0) > th; boxes, j = [], 0
    while j < n:
        if hot[j]:
            e = j
            while e + 1 < n and hot[e + 1]:
                e += 1
            boxes.append((j, e)); j = e + 1
        else:
            j += 1
    cs = np.cumsum(p, axis=0); cand = []
    for a0, b0 in boxes:                                            # every rectangle inside the box: (A, size, E, a, b)
        cb = [(0.0, 0, 0, a0, a0)]
        for E in range(1, F + 1):
            col = cs[E - 1, a0:b0 + 1]; pre = np.concatenate([[0.0], np.cumsum(col)]); m = len(col)
            for a in range(m):
                for b in range(a, m):
                    cb.append((float(pre[b + 1] - pre[a]), E * (b - a + 1), E, a0 + a, a0 + b))
        cand.append(cb)
    ch = [0] * len(boxes)
    for _ in range(2):
        for i, cb in enumerate(cand):
            A0 = sum(cand[j][ch[j]][0] for j in range(len(cand)) if j != i); S0 = sum(cand[j][ch[j]][1] for j in range(len(cand)) if j != i)
            sc = [(A0 + c[0]) / (S0 + c[1] + P - A0 - c[0]) if S0 + c[1] > 0 else 0.0 for c in cb]; ch[i] = int(np.argmax(sc))
    out = np.zeros_like(p, dtype=bool)
    for i, cb in enumerate(cand):
        _, sz, E, a, b = cb[ch[i]]
        if sz:
            out[:E, a:b + 1] = True
    return out


def module(nm, path, argv):
    a = sys.argv; sys.argv = argv; spec = importlib.util.spec_from_file_location(nm, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); sys.argv = a
    return mod


g132 = module("g132", R + "review/g132_p3_grid.py", ["g132", "run"]); D4 = pickle.load(open(W + "g284_og_pretemp_dev.pkl", "rb")); rows = []
for w, v in D4.items():
    m = G.META[w]; r = dict(w=w, cond=m["cond"], panel=m["panel"], day=m["day"], hour=(m["t"] % 288) // 12, base=0.0, sc=0.0)
    for pr, p, ex, nr, _ in v["k"].values():
        q = tmp(p, 0.8); r["base"] += pr * G.iou(g132.gate(bs(q), ex, nr), m); r["sc"] += pr * G.iou(g132.gate(scen(q), ex, nr), m)
    rows.append(r)
DM = pd.DataFrame(rows); sm = m3v2.summary(DM, "base", "sc", "private"); R7 = pd.read_csv(W + "g287_m3.csv").set_index("w")["t0.8"]
print(f"identity base against g287 t0.8: {float(np.abs(DM.set_index('w').base - R7.reindex(DM.w).to_numpy()).max()):.1e} | M3 dev ongoing (A17) scenario - best_set: {sm['t2_weighted']:+.5f} (SE {sm['se']:.5f}, z {sm['t2_weighted'] / max(sm['se'], 1e-12):+.2f}), windows changed {int((~np.isclose(DM.sc, DM.base)).sum())}", flush=True)
h = module("g119", R + "review/g119_month_postproc.py", ["g119", "run"]); T_ = h.T; D5 = pickle.load(open(W + "g285_jf_pretemp.pkl", "rb")); rows = []
for mo in h.MONTHS:
    for p_ in sorted({v["panel"] for v in D5.values() if v["month"] == mo}):
        P = h.load(p_, "train"); key = (mo, p_, "mask"); h.CTX.clear(); h.sim_blank.cache_clear(); h.CTX[key] = h.Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = h.CTX[key]
        for k in [k for k, v in D5.items() if v["month"] == mo and v["panel"] == p_]:
            t = int(T_.META["t"][k]); y, e = D5[k]["y"], D5[k]["e"]; lq = np.asarray(h.XN[h.OFFN[k]:h.OFFN[k + 1]][:, h.G.CI], float) == 1; r = dict(w=k, month=mo, panel=p_, day=D5[k]["day"], base=0.0, sc=0.0)
            for kk, (pr, pb) in D5[k]["k"].items():
                q = tmp(pb, 0.8); r["base"] += pr * h.iou(h.far_gate(c, h.decode(q), t, kk, lq), y, e, "queue_ongoing"); r["sc"] += pr * h.iou(h.far_gate(c, scen(q), t, kk, lq), y, e, "queue_ongoing")
            rows.append(r)
        h.load.cache_clear()
DJ = pd.DataFrame(rows); R8 = pd.read_csv(W + "g287_jf.csv").set_index("w")["t0.8"]; cl = pd.factorize(DJ.month + "_" + DJ.panel + "_" + DJ.day.astype(str))[0]
Rm = np.random.default_rng(0).multinomial(cl.max() + 1, np.full(cl.max() + 1, 1 / (cl.max() + 1)), size=2000); d = (DJ.sc - DJ.base).to_numpy(); se = float(((Rm @ np.bincount(cl, d)) / (Rm @ np.bincount(cl))).std())
print(f"identity JF base against g287 t0.8: {float(np.abs(DJ.set_index('w').base - R8.reindex(DJ.w).to_numpy()).max()):.1e} | JF ongoing (capB0) scenario - best_set: {d.mean():+.5f} (SE {se:.5f}, z {d.mean() / max(se, 1e-12):+.2f}), windows changed {int((np.abs(d) > 1e-12).sum())}", flush=True)
w1, w2 = 1 / max(sm["se"], 1e-9) ** 2, 1 / max(se, 1e-9) ** 2; mg = (w1 * sm["t2_weighted"] + w2 * d.mean()) / (w1 + w2); smg = (w1 + w2) ** -0.5
print(f"merged {mg:+.5f} (z {mg / smg:+.2f}) | line: {sm['t2_weighted'] > 0 and d.mean() > 0 and mg / smg >= 1}", flush=True); DM.to_csv(W + "g315_m3.csv", index=False); DJ.to_csv(W + "g315_jf.csv", index=False)
