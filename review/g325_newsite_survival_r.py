"""G325 (11/05, ledger #258r): #258 measured again with the declared definitions. (i) Survival = the box lifetime of WHERE_IS_THE_GAP
5 (the investigator's births table births_all.parquet, training births of >= 2 rows: dur in rows; new sites f_site < 0.02 median 70 min, all
280 min); (ii) new site = G198's / #160's f_site (share of training days with a head within +-2 links; heads = the ends of the last_queue
runs with gaps <= 2 merged; M3 = all training months, January / February = without the window's month) < 0.02 at a head. Otherwise as G323.
From G323: Head of a box = the most downstream queued link of a run on the last visible row
of R (higher index = downstream, as G119's cone). Site head frequency = share of visible rows where the link is a head (queued, its downstream
neighbour not): M3 windows = training months + March days before the window's day; January / February = the other training months. New site:
the box's head frequency < 0.02 (112 M3 dev / 114 January / February ongoing windows). Survival from the training months: per link, the runs
of consecutive visible queued rows (runs cut by a hidden row are dropped as censored), S_all(a) = P(run > a) over all links, S_new over the
links whose training head frequency < 0.02. A new-site box seen for a rows (consecutive rows back from the last visible row with a queued
link of the run) gets, on its links and horizon row h = 1 .. 6, the factor [S_new(a + h) / S_new(a)] / [S_all(a + h) / S_all(a)] on the
probability before the temperature; then A17's temperature 0.8, best_set, far gate (M3: G284's dump, G132's gate; January / February: G285's
capB0 flow, G119's gate). No constant fitted. Line: both > 0, merged z >= +1. usage: TFB_M3VER=v3 py -B -X utf8 g325_newsite_survival_r.py"""
import os, sys, pickle, importlib.util
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
R0 = r"C:/Users/user/tfb/"; sys.path.insert(0, r"C:/Users/user/tfb_a8/src"); sys.path.insert(0, R0 + "src")
from cachelib import load, T2_PANELS
import t2_m3_gate as G, m3v2
from t2_model import best_set
W = R0 + "work/t2/"; AMAX = 400
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))); tmp = lambda p, T: 1 / (1 + np.exp(-lg(p) / T))
bs = lambda pb: best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape)


def heads(V, vcut):
    fin = np.isfinite(V); q = fin & (V <= vcut[None, :]); h = q & ~np.concatenate([q[:, 1:], np.zeros((len(q), 1), bool)], axis=1); return h.sum(axis=0), fin.sum(axis=0), q, fin


BF = r"C:/Users/user/AppData/Local/Temp/claude/C--Users-user-OneDrive-Desktop/1f1db98a-17cd-431b-aafa-61be2b25086b/scratchpad/incident/births_all.parquet"
B = pd.read_parquet(BF); TR = B[(B.split == "train") & (B.nrows >= 2)]; NL = B.groupby("panel").n.first().to_dict(); FC = {}
DUR_ALL = TR.dur.to_numpy(); DUR_NEW = TR[TR.f_site < 0.02].dur.to_numpy()


def fsite(p, excl):
    key = (p, excl)
    if key not in FC:
        T_ = TR[(TR.panel == p) & (TR.month != excl)]; dd = TR[TR.month != excl].date.unique(); c_ = np.zeros(NL[p])
        for d, g in T_.groupby("date"):
            m = np.zeros(NL[p], bool)
            for h in g.head0.unique():
                m[max(h - 2, 0):h + 3] = True
            c_ += m
        FC[key] = c_ / max(len(dd), 1)
    return FC[key]


surv = lambda d: np.array([(np.asarray(d) > a).mean() for a in range(AMAX + 8)])
SA, SN = surv(DUR_ALL), surv(DUR_NEW)
print(f"box lifetimes (training births >= 2 rows): all {len(DUR_ALL)} (median {np.median(DUR_ALL) * 5:.0f} min), new sites {len(DUR_NEW)} (median {np.median(DUR_NEW) * 5:.0f} min); S(6): all {SA[6]:.3f}, new {SN[6]:.3f}", flush=True)


def factor(a):
    a = min(a, AMAX); out = np.ones(6)
    for h in range(1, 7):
        if SA[a] > 0 and SN[a] > 0 and SA[a + h] > 0:
            out[h - 1] = (SN[a + h] / SN[a]) / (SA[a + h] / SA[a])
    return out


def adjust(p, lq, Rw, freq):
    """the last_queue runs (gaps <= 2 merged) whose head (run end) has f_site < 0.02 get the survival factor on their links"""
    out = p.copy(); hit = False; l = np.flatnonzero(lq)
    if not len(l):
        return out, hit
    v, o = Rw[:, :12, 0].astype(float), Rw[:, :12, 2].astype(float) > 0; q = o & (v <= 1.0); rows = np.flatnonzero(o.any(axis=0))
    for run in np.split(l, np.flatnonzero(np.diff(l) > 2) + 1):
        j, e = int(run[0]), int(run[-1])
        if freq[e] < 0.02:
            a = 0
            for r in rows[::-1]:
                if q[j:e + 1, r].any():
                    a += 1
                else:
                    break
            out[:, j:e + 1] = np.clip(out[:, j:e + 1] * factor(max(a, 1))[:, None], 0, 1); hit = True
    return out, hit


def module(nm, path, argv):
    a = sys.argv; sys.argv = argv; spec = importlib.util.spec_from_file_location(nm, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); sys.argv = a
    return mod


g132 = module("g132", R0 + "review/g132_p3_grid.py", ["g132", "run"]); D4 = pickle.load(open(W + "g284_og_pretemp_dev.pkl", "rb"))
d = "C:/tfb_data/tfb_nn/nn_test_m3v3/"; RM = np.load(d + "R.npy", mmap_mode="r"); off = np.load(d + "off.npy"); pos = {str(w): i for i, w in enumerate(np.load(d + "window_id.npy", allow_pickle=True))}; rows = []
for w, v in D4.items():
    m = G.META[w]; pnl = m["panel"]; freq = fsite(pnl, "none"); i_ = G.POS[w]; lq = np.asarray(G.X[G.OFF[i_]:G.OFF[i_ + 1], G.CI], float) == 1; Rw = np.asarray(RM[off[pos[w]]:off[pos[w] + 1]]); r = dict(w=w, cond=m["cond"], panel=pnl, day=m["day"], hour=(m["t"] % 288) // 12, base=0.0, ns=0.0, hit=False)
    for pr, p, ex, nr, _ in v["k"].values():
        p2, hit = adjust(p, lq, Rw, freq); r["hit"] |= hit; r["base"] += pr * G.iou(g132.gate(bs(tmp(p, 0.8)), ex, nr), m); r["ns"] += pr * G.iou(g132.gate(bs(tmp(p2, 0.8)), ex, nr), m)
    rows.append(r)
DM = pd.DataFrame(rows); sm = m3v2.summary(DM, "base", "ns", "private"); R7 = pd.read_csv(W + "g287_m3.csv").set_index("w")["t0.8"]
print(f"identity base against g287: {float(np.abs(DM.set_index('w').base - R7.reindex(DM.w).to_numpy()).max()):.1e} | M3 dev ongoing (A17): new-site windows {int(DM.hit.sum())}, ns - base {sm['t2_weighted']:+.5f} (SE {sm['se']:.5f}, z {sm['t2_weighted'] / max(sm['se'], 1e-12):+.2f}), changed {int((~np.isclose(DM.ns, DM.base)).sum())}", flush=True)
h = module("g119", R0 + "review/g119_month_postproc.py", ["g119", "run"]); T_ = h.T; D5 = pickle.load(open(W + "g285_jf_pretemp.pkl", "rb")); A = "C:/tfb_data/tfb_nn/nn_data_afdem/"; RA = np.load(A + "R.npy", mmap_mode="r"); oa = np.load(A + "off.npy"); rows = []
for mo in h.MONTHS:
    for pnl in sorted({v["panel"] for v in D5.values() if v["month"] == mo}):
        P = h.load(pnl, "train"); key = (mo, pnl, "mask"); h.CTX.clear(); h.sim_blank.cache_clear(); h.CTX[key] = h.Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = h.CTX[key]
        freq = fsite(pnl, mo)
        for k in [k for k, v in D5.items() if v["month"] == mo and v["panel"] == pnl]:
            t = int(T_.META["t"][k]); y, e = D5[k]["y"], D5[k]["e"]; lq = np.asarray(h.XN[h.OFFN[k]:h.OFFN[k + 1]][:, h.G.CI], float) == 1; Rw = np.asarray(RA[oa[k]:oa[k + 1]])
            r = dict(w=k, month=mo, panel=pnl, day=D5[k]["day"], base=0.0, ns=0.0, hit=False)
            for kk, (pr, pb) in D5[k]["k"].items():
                p2, hit = adjust(pb, lq, Rw, freq); r["hit"] |= hit
                r["base"] += pr * h.iou(h.far_gate(c, h.decode(tmp(pb, 0.8)), t, kk, lq), y, e, "queue_ongoing"); r["ns"] += pr * h.iou(h.far_gate(c, h.decode(tmp(p2, 0.8)), t, kk, lq), y, e, "queue_ongoing")
            rows.append(r)
        h.load.cache_clear()
DJ = pd.DataFrame(rows); cl = pd.factorize(DJ.month + "_" + DJ.panel + "_" + DJ.day.astype(str))[0]; Rm = np.random.default_rng(0).multinomial(cl.max() + 1, np.full(cl.max() + 1, 1 / (cl.max() + 1)), size=2000)
dd = (DJ.ns - DJ.base).to_numpy(); se = float(((Rm @ np.bincount(cl, dd)) / (Rm @ np.bincount(cl))).std()); R8 = pd.read_csv(W + "g287_jf.csv").set_index("w")["t0.8"]
print(f"identity JF base against g287: {float(np.abs(DJ.set_index('w').base - R8.reindex(DJ.w).to_numpy()).max()):.1e} | Jan / Feb ongoing (capB0): new-site windows {int(DJ.hit.sum())}, ns - base {dd.mean():+.5f} (SE {se:.5f}, z {dd.mean() / max(se, 1e-12):+.2f}), changed {int((np.abs(dd) > 1e-12).sum())}", flush=True)
w1, w2 = 1 / max(sm["se"], 1e-9) ** 2, 1 / max(se, 1e-9) ** 2; mg = (w1 * sm["t2_weighted"] + w2 * dd.mean()) / (w1 + w2); smg = (w1 + w2) ** -0.5
print(f"merged {mg:+.5f} (z {mg / smg:+.2f}) | line: {sm['t2_weighted'] > 0 and dd.mean() > 0 and mg / smg >= 1}", flush=True)
