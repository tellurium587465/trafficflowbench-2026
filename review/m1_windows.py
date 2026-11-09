"""M1: per-window table for the ongoing yardstick (train OOF) + the same history-only descriptors on the test windows.

History-only descriptors (rows T-12..T-1; the same code runs on train and on the released test-window inputs):
  ncomp, f_vis (rarity of the visible queue head: daily queue frequency at head+-1 in the 30-min bin, other months,
  same day type; for test windows all 9 train months), f_vis_min (min over components), r_vis (min over components of
  flow at head+1 / typical flow there in that 5-min slot; < 1 = throughput below normal = capacity-loss signature),
  hist_rows (history rows holding a queue), nq_T1 (queued links at the last row).
Train only (labels, for the mechanism): IoU of the blend, IoU after the far rule (D=3), persistence IoU, far label/pred
cells, incident-episode overlap (m1_episodes r_typ < 0.9).
usage: python m1_windows.py -> m1_windows.pkl, m1_testwin.pkl
"""
import os, sys
os.environ["TFB_T2CACHE"] = "2_noT_af"; os.environ["TFB_T2_FAR"] = "3"
sys.path.insert(0, r"C:/Users/user/tfb/src")
import numpy as np, pandas as pd
import t2_eval_oof as EV
from t2_model import best_set
from cachelib import load, T2_PANELS
from common import REL
from t2_windows import queue_matrix, iou

SPEC = os.environ.get("M1_SPEC", "l127_full_af:2+q14_attn1_glob+q16_attnglob_s1+q14_attn1_wp05:2+q13_attn1:2+q12_af_r13_nopos+q17_drop1")
M, LAB = EV.T.META, EV.T.LAB
EV.far_mask(int(np.flatnonzero(M["cond"] == "queue_ongoing")[0]))      # loads the last-queue array
LQ, OFF = EV._FARM["lq"], EV._FARM["off"]
EP = pd.read_pickle(r"C:/Users/user/tfb/review/m1_episodes.pkl")
INC = EP[(EP.dur >= 6) & (EP.r_typ < 0.9)]


def comps(lq):
    idx = np.flatnonzero(lq); out = []
    for j in idx:
        if out and j - out[-1][-1] <= 2:
            out[-1].append(j)
        else:
            out.append([j])
    return out


def tables(P, mon, wk, months):
    D = P.T // 288; n = P.n; Qm = queue_matrix(P)
    Q3 = np.where(np.isfinite(P.Q), P.Q, np.nan).reshape(D, 288, n); qb = Qm.reshape(D, 48, 6, n).any(axis=2)
    typ, frq = {}, {}
    for m in months:
        for we in (False, True):
            s = (mon != m) & (wk == we)
            typ[(m, we)] = np.nanmedian(Q3[s], axis=0); frq[(m, we)] = qb[s].mean(axis=0)
    return Qm, typ, frq


def describe(P, Qm, t, lq, ty, fr):
    cs = comps(lq); slot = t % 288; out = dict(ncomp=len(cs), nq_T1=int(lq.sum()), hist_rows=int(Qm[t - 12:t].any(axis=1).sum()))
    fs, rs, sz = [], [], []
    for c in cs:
        h = c[-1]; fs.append(fr[slot // 6, max(0, h - 1):h + 2].max()); sz.append(c[-1] - c[0] + 1)
        j = h + 1; r = []
        if j < P.n:
            for row in range(t - 6, t):
                if not Qm[row, j] and np.isfinite(P.Q[row, j]) and np.isfinite(ty[row % 288, j]) and ty[row % 288, j] > 0:
                    r.append(P.Q[row, j] / ty[row % 288, j])
        rs.append(np.median(r) if r else np.nan)
    if cs:
        mi = int(np.argmax(sz)); out.update(f_vis=fs[mi], f_vis_min=min(fs), r_vis=np.nanmin(rs) if np.isfinite(rs).any() else np.nan,
                                            r_main=rs[mi], head=cs[mi][-1], size_main=sz[mi])
    else:
        out.update(f_vis=np.nan, f_vis_min=np.nan, r_vis=np.nan, r_main=np.nan, head=-1, size_main=0)
    return out


if __name__ == "__main__":
    oof = EV.load_oof(SPEC)
    rows = []
    for p in T2_PANELS:
        P = load(p, "train"); days = [str(x) for x in P.day]; mon = np.array([d[:7] for d in days])
        wk = np.array([pd.Timestamp(d).dayofweek >= 5 for d in days]); months = sorted(set(mon))
        Qm, typ, frq = tables(P, mon, wk, months)
        inc = INC[INC.panel == p]
        for k in np.flatnonzero((M["panel"] == p) & (M["cond"] == "queue_ongoing") & (M["is_eval"] | M["is_sel"])):
            if k not in oof:
                continue
            t = int(M["t"][k]); d = t // 288; m, we = mon[d], wk[d]
            lq = LQ[OFF[k]:OFF[k + 1]]; lab, el = LAB[k]; lab = lab.astype(bool); el = el.astype(bool)
            r = describe(P, Qm, t, lq, typ[(m, we)], frq[(m, we)])
            o = oof[k]; s = best_set(o.ravel(), np.ones(o.size, bool)).reshape(o.shape); far = EV.far_mask(k)
            sf = s & ~far[None, :]
            pers = np.broadcast_to(Qm[t - 1], lab.shape)
            labq = lab.any(axis=0)
            ov = inc[(inc.t0 <= t + 6) & (inc.t1 >= t - 12)]
            hit = any(((labq | lq)[max(0, h - 3):h + 4]).any() for h in ov.head0)
            rows.append(dict(k=int(k), panel=p, t=t, date=days[d], month=m, weekend=bool(we), tod=(t % 288) / 12,
                             is_sel=bool(M["is_sel"][k]), is_eval=bool(M["is_eval"][k]), iou=iou(s, lab, el), iou_far=iou(sf, lab, el),
                             iou_pers=iou(pers, lab, el), n_lab=int((lab & el).sum()), far_lab=int((lab & el & far[None, :]).sum()),
                             far_pred=int((s & far[None, :]).sum()), n_pred=int(s.sum()), far_tp=int((s & lab & el & far[None, :]).sum()),
                             inc=bool(hit), **r))
        print(p, sum(1 for r in rows if r["panel"] == p), flush=True); load.cache_clear()
    W = pd.DataFrame(rows); W.to_pickle(r"C:/Users/user/tfb/review/m1_windows.pkl")
    # test windows: history-only descriptors against all 9 train months
    names = list(np.load("E:/tfb_nn/nn_data_afdem/names.npy", allow_pickle=True)); ci = names.index("last_queue")
    dt = "E:/tfb_nn/nn_test_afdem/"; Xt = np.load(dt + "X.npy", mmap_mode="r"); offt = np.load(dt + "off.npy")
    wid = np.load(dt + "window_id.npy", allow_pickle=True).astype(str); widx = {w: i for i, w in enumerate(wid)}
    trows = []
    for p in T2_PANELS:
        P = load(p, "train"); days = [str(x) for x in P.day]; mon = np.array([d[:7] for d in days])
        wk = np.array([pd.Timestamp(d).dayofweek >= 5 for d in days])
        _, typ, frq = tables(P, mon, wk, ["none"])
        for split in ("validation", "private"):
            Pt = load(p, split); Qt = queue_matrix(Pt); ti = {s: i for i, s in enumerate(Pt.times)}
            wi = pd.read_csv(REL / "task2" / p / split / "window_index.csv")
            for r in wi.itertuples():
                t = ti[r.forecast_origin]; i = widx[r.window_id]
                lq = np.asarray(Xt[offt[i]:offt[i + 1], ci]) == 1
                we = pd.Timestamp(r.date).dayofweek >= 5
                dsc = describe(Pt, Qt, t, lq, typ[("none", we)], frq[("none", we)]) if r.condition == "queue_ongoing" else {}
                trows.append(dict(window_id=r.window_id, panel=p, split=split, cond=r.condition, date=r.date, tod=(t % 288) / 12,
                                  order=int(r.window_id.rsplit("_", 1)[1]), **dsc))
            load.cache_clear()
    pd.DataFrame(trows).to_pickle(r"C:/Users/user/tfb/review/m1_testwin.pkl")
    print("test windows", len(trows))
