"""G326 (11/05, ledger #259): lift the satellite band by calibration. Band = the links 4-8 upstream of each tail of the last-queue runs (lq: each
link's latest finite state at T-60 .. T-5; tails as G269 = the upstream ends; links ordered upstream -> downstream), every horizon row.
c = (truth rate of the band cells) / (mean predicted probability of the band cells), from the training months only: the OOF of q25 nomem s3 /
s4 (oof_q25_nomem_s*) on the selector ongoing windows of 2030-06 .. 2030-12 (January / February left out, they are the read). Then the band
cells' probability (before the temperature) times c, capped at 0.99, and the usual decode: M3 v3 dev ongoing (G284's A17 mix, temperature
0.8, best_set, G132's gate) and January / February (G285's capB0 flow, temperature 0.8, G119's gate). Line: both > 0, merged z >= +1.
usage: TFB_M3VER=v3 py -B -X utf8 g326_band_calib.py"""
import os, sys, pickle, importlib.util
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
R0 = r"C:/Users/user/tfb/"; sys.path.insert(0, R0 + "src")
import t2_m3_gate as G, m3v2
from t2_model import best_set
W = R0 + "work/t2/"
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))); tmp = lambda p, T: 1 / (1 + np.exp(-lg(p) / T))
bs = lambda pb: best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape)


def band(lq):
    n = len(lq); m = np.zeros(n, bool)
    for j in [j for j in range(n) if lq[j] and (j == 0 or not lq[j - 1])]:
        m[max(j - 8, 0):max(j - 3, 0)] = True
    return m & ~lq


def module(nm, path, argv):
    a = sys.argv; sys.argv = argv; spec = importlib.util.spec_from_file_location(nm, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); sys.argv = a
    return mod


h = module("g119", R0 + "review/g119_month_postproc.py", ["g119", "run"]); T_ = h.T; MON = h.MON
O = [pickle.load(open(W + f"oof_q25_nomem_s{s}.pkl", "rb")) for s in (3, 4)]; yt = pt = n = 0.0
KS = [k for k in np.flatnonzero(T_.META["is_sel"] & np.isin(MON, ["2030-06", "2030-07", "2030-08", "2030-09", "2030-10", "2030-11", "2030-12"])) if str(T_.META["cond"][k]) == "queue_ongoing" and k in O[0] and k in O[1]]
for k in KS:
    lab, el = T_.LAB[k]; y, e = lab.astype(bool), el.astype(bool); lq = np.asarray(h.XN[h.OFFN[k]:h.OFFN[k + 1]][:, h.G.CI], float) == 1; b = band(lq)
    p = (O[0][k] + O[1][k]) / 2; msk = e & b[None, :]; yt += (y & msk).sum(); pt += p[msk].sum(); n += msk.sum()
C = float(yt / max(pt, 1e-9)); print(f"training OOF (2030-06 .. 12, {len(KS)} ongoing windows): band cells {int(n)}, truth rate {yt / n:.4f}, mean p {pt / n:.4f} -> c {C:.3f}", flush=True)


def lift(p, lq):
    b = band(lq); q = p.copy(); q[:, b] = np.minimum(q[:, b] * C, 0.99); return q


g132 = module("g132", R0 + "review/g132_p3_grid.py", ["g132", "run"]); D4 = pickle.load(open(W + "g284_og_pretemp_dev.pkl", "rb")); rows = []
for w, v in D4.items():
    m = G.META[w]; i_ = G.POS[w]; lq = np.asarray(G.X[G.OFF[i_]:G.OFF[i_ + 1], G.CI], float) == 1; r = dict(w=w, cond=m["cond"], panel=m["panel"], day=m["day"], hour=(m["t"] % 288) // 12, base=0.0, bd=0.0)
    for pr, p, ex, nr, _ in v["k"].values():
        r["base"] += pr * G.iou(g132.gate(bs(tmp(p, 0.8)), ex, nr), m); r["bd"] += pr * G.iou(g132.gate(bs(tmp(lift(p, lq), 0.8)), ex, nr), m)
    rows.append(r)
DM = pd.DataFrame(rows); sm = m3v2.summary(DM, "base", "bd", "private"); R7 = pd.read_csv(W + "g287_m3.csv").set_index("w")["t0.8"]
print(f"identity base against g287: {float(np.abs(DM.set_index('w').base - R7.reindex(DM.w).to_numpy()).max()):.1e} | M3 dev ongoing (A17): band x c - base {sm['t2_weighted']:+.5f} (SE {sm['se']:.5f}, z {sm['t2_weighted'] / max(sm['se'], 1e-12):+.2f}), changed {int((~np.isclose(DM.bd, DM.base)).sum())}", flush=True)
D5 = pickle.load(open(W + "g285_jf_pretemp.pkl", "rb")); rows = []
for mo in h.MONTHS:
    for pnl in sorted({v["panel"] for v in D5.values() if v["month"] == mo}):
        P = h.load(pnl, "train"); key = (mo, pnl, "mask"); h.CTX.clear(); h.sim_blank.cache_clear(); h.CTX[key] = h.Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = h.CTX[key]
        for k in [k for k, v in D5.items() if v["month"] == mo and v["panel"] == pnl]:
            t = int(T_.META["t"][k]); y, e = D5[k]["y"], D5[k]["e"]; lq = np.asarray(h.XN[h.OFFN[k]:h.OFFN[k + 1]][:, h.G.CI], float) == 1; r = dict(w=k, month=mo, panel=pnl, day=D5[k]["day"], base=0.0, bd=0.0)
            for kk, (pr, pb) in D5[k]["k"].items():
                r["base"] += pr * h.iou(h.far_gate(c, h.decode(tmp(pb, 0.8)), t, kk, lq), y, e, "queue_ongoing"); r["bd"] += pr * h.iou(h.far_gate(c, h.decode(tmp(lift(pb, lq), 0.8)), t, kk, lq), y, e, "queue_ongoing")
            rows.append(r)
        h.load.cache_clear()
DJ = pd.DataFrame(rows); cl = pd.factorize(DJ.month + "_" + DJ.panel + "_" + DJ.day.astype(str))[0]; Rm = np.random.default_rng(0).multinomial(cl.max() + 1, np.full(cl.max() + 1, 1 / (cl.max() + 1)), size=2000)
dd = (DJ.bd - DJ.base).to_numpy(); se = float(((Rm @ np.bincount(cl, dd)) / (Rm @ np.bincount(cl))).std()); R8 = pd.read_csv(W + "g287_jf.csv").set_index("w")["t0.8"]
print(f"identity JF base against g287: {float(np.abs(DJ.set_index('w').base - R8.reindex(DJ.w).to_numpy()).max()):.1e} | Jan / Feb ongoing (capB0): band x c - base {dd.mean():+.5f} (SE {se:.5f}, z {dd.mean() / max(se, 1e-12):+.2f}), changed {int((np.abs(dd) > 1e-12).sum())}", flush=True)
w1, w2 = 1 / max(sm["se"], 1e-9) ** 2, 1 / max(se, 1e-9) ** 2; mg = (w1 * sm["t2_weighted"] + w2 * dd.mean()) / (w1 + w2); smg = (w1 + w2) ** -0.5
print(f"merged {mg:+.5f} (z {mg / smg:+.2f}) | line: {sm['t2_weighted'] > 0 and dd.mean() > 0 and mg / smg >= 1}", flush=True)
