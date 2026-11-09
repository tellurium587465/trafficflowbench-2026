"""G273 (11/04, ledger #228 stage 2'): the January / February instrument on the first-passage windows (G272, 442) against the old
random-order windows (T.META is_sel, 316), one flow for both: capB0 fold 2 seeds 0-3 (old: probs_jfmh0, new: probs_jfnew, the same
build path G220), G119's SCG flow with the row T as the last queue (G229's trow, the A-side post-processing; B's January / February
X3a / u2d probabilities do not exist for the new windows). Per window the IoU (eligible cells, k expectation); per horizon row the
calibration sum(p) / sum(y) on eligible cells, split by the cell's link being queued at the last observation (lq2) or not; window
traits (hour, queue length at T, persistence). Identity: windows in both sets have the same probabilities. Training split only.
usage: TFB_M3VER=v3 py -B -X utf8 g273_firstpass_read.py"""
import os, sys, pickle, importlib.util
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
R = r"C:/Users/user/tfb/"; a_ = sys.argv; sys.argv = ["g119", "run"]
spec = importlib.util.spec_from_file_location("g119", R + "review/g119_month_postproc.py"); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g); sys.argv = a_
T, MON, W = g.T, g.MON, g.W; M = T.META; NN = "C:/tfb_data/tfb_nn/"
PO = [pickle.load(open(W + f"probs_jfmh0__capB0_s{s}_fold2030-12.pkl", "rb")) for s in range(4)]; PN = [pickle.load(open(W + f"probs_jfnew__capB0_s{s}_fold2030-12.pkl", "rb")) for s in range(4)]
XN_ = np.load(NN + "nn_test_jfnew/X.npy", mmap_mode="r"); ON_ = np.load(NN + "nn_test_jfnew/off.npy"); WN_ = [str(x) for x in np.load(NN + "nn_test_jfnew/window_id.npy", allow_pickle=True)]; POS = {w: i for i, w in enumerate(WN_)}
NEW = pd.read_csv(W + "g272_windows.csv"); OLD = [int(k) for k in np.flatnonzero(M["is_sel"] & np.isin(MON, g.MONTHS)) if str(M["cond"][k]) == "queue_ongoing"]
items = [("old", str(M["panel"][k]), int(M["t"][k]), k) for k in OLD] + [("new", r.panel, int(r.t), None) for r in NEW.itertuples()]
rows = []; CAL = {}; ident = []
for p in sorted({x[1] for x in items}):
    P = g.load(p, "train"); key = ("2031", p, "mask"); g.CTX.clear(); g.sim_blank.cache_clear(); g.CTX[key] = g.Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = g.CTX[key]
    Qm = (np.nan_to_num(P.V, nan=1e9) <= P.vcut[None, :]) & P.E
    for src, _, t, k in [x for x in items if x[1] == p]:
        if src == "old":
            p_ = sum(d[str(k)] for d in PO) / 4; Xw = np.asarray(g.XN[g.OFFN[k]:g.OFFN[k + 1]])
        else:
            w = f"{p}|{t}"; p_ = sum(d[w] for d in PN) / 4; i = POS[w]; Xw = np.asarray(XN_[ON_[i]:ON_[i + 1]])
            kk_ = [k2 for k2 in OLD if str(M["panel"][k2]) == p and int(M["t"][k2]) == t]
            if kk_:
                ident.append(float(np.abs(p_ - sum(d[str(kk_[0])] for d in PO) / 4).max()))
        y = Qm[t + 1:t + 7]; e = P.E[t + 1:t + 7]; dom = int(P.times[t][8:10]) - 1
        lq = np.asarray(Xw[:, g.G.CI], float) == 1; ot = c.obs[t].astype(bool); lq2 = np.where(ot, c.q[t].astype(bool), lq); cone = g.cone_parts(c, t, lq2, P.length)
        iou = sum(pr * g.iou(g.ongoing_pred(c, p_ * g.cone_f(c, cone, t, kk), t, kk, True, lq2), y, e, "queue_ongoing") for kk, pr in g.m3v2.kdist(dom, cap=3).items())
        for h in range(6):
            for lab, mk in (("queued at T", lq2), ("free at T", ~lq2)):
                m_ = e[h] & mk; a = CAL.setdefault((src, h, lab), [0.0, 0.0]); a[0] += float(p_[h][m_].sum()); a[1] += float(y[h][m_].sum())
        rows.append(dict(src=src, panel=p, t=t, month=P.times[t][:7], day=dom, hour=int(P.times[t][11:13]), iou=iou, qlen=int(lq2.sum()), ycells=int(y.sum())))
    g.load.cache_clear()
D = pd.DataFrame(rows); D.to_csv(W + "g273_windows.csv", index=False)
print("identity (windows in both sets):", len(ident), "max |dp|", max(ident) if ident else None, flush=True)


def boot(d, cl, n=2000):
    ids = pd.factorize(cl)[0]; M_ = np.bincount(ids, d); N_ = np.bincount(ids); Rm = np.random.default_rng(0).multinomial(len(M_), np.full(len(M_), 1 / len(M_)), size=n); return float(((Rm @ M_) / (Rm @ N_)).std())


for src, d in D.groupby("src"):
    cl = (d.month + "_" + d.panel + "_" + d.day.astype(str)).to_numpy()
    print(f"{src}: {len(d)} windows | IoU {d.iou.mean():.4f} (SE {boot(d.iou.to_numpy(), cl):.4f}) | hour median {d.hour.median():.0f} | queued links at T median {d.qlen.median():.0f} | true horizon cells median {d.ycells.median():.0f}", flush=True)
print("calibration sum(p) / sum(y) by horizon row (old | new):")
for lab in ("queued at T", "free at T"):
    print(f"  {lab}: " + " ".join(f"T+{5 * (h + 1)} {CAL[('old', h, lab)][0] / max(CAL[('old', h, lab)][1], 1):.2f}|{CAL[('new', h, lab)][0] / max(CAL[('new', h, lab)][1], 1):.2f}" for h in range(6)), flush=True)
