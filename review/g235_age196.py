"""G235 (11/03, ledger #196): the ongoing training pool's jam-age weights, the gate on fold 2.
Age (the strategy's learnrev/persist.py): rows since the corridor last had no queued link, over the 72 rows before T (cap 72).
  weights : the age of every nn_data_afdem ongoing window, the shares of the training pool (~is_sel) and of the selector windows (is_sel),
            the ratio per age bin (<= 6, 7-24, 25-71, >= 72 rows) from fold 2's training months only (2030-06..2030-11, the read months
            not used) -> C:/tfb_data/tfb_nn/age196/{age,aw}.npy (aw = the ratio of the window's bin, 1 for onset); first an alignment
            check of t against the labels (T+5..T+30 == queue rows t+1..t+6)
  read    : fold-2 OOF on the selector windows of 2030-12 / 2031-01 / 2031-02 (history not masked), ongoing, raw best_set IoU of the
            2-seed mean, new (awL s0, s1) - base (a0L s0, s1: AW=0, the same script trained now), paired; cluster (panel x day) bootstrap;
            per month, per age bin. ref = b0L s0, s1 (10/31 02:01, before t2_nn3.py changed at 04:15): printed, not compared
  m3      : M3 v3 dev ongoing from g130's m3 mode (G130_CFGS=B0,AW G130_LOCAL=B0=a0L,AW=awL G130_OUT=g235_m3.csv, the fold models'
            CPU probabilities probs_m3v3cap__<tag>_s<s>_fold2030-12.pkl), AW - B0 on all windows and on the top third of history coverage
            (nn_test_m3v3 R[:, :12, 2], as G212)
usage: py -B -X utf8 g235_age196.py weights | read | m3"""
import os, sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.dont_write_bytecode = True
from cachelib import load
from t2_windows import queue_matrix
D = "C:/tfb_data/tfb_nn/nn_data_afdem/"; O = "C:/tfb_data/tfb_nn/age196/"; MODE = sys.argv[1]
cond = np.load(D + "cond.npy", allow_pickle=True).astype(str); date = np.load(D + "date.npy", allow_pickle=True).astype(str)
panel = np.load(D + "panel.npy", allow_pickle=True).astype(str); tt = np.load(D + "t.npy"); is_sel = np.load(D + "is_sel.npy")
mon = np.array([d[:7] for d in date]); og = cond == "queue_ongoing"
EDGES = (6, 24, 71); NAMES = ("<=6", "7-24", "25-71", ">=72")
binof = lambda a: np.searchsorted(np.array(EDGES), a, side="left")   # 0: <= 6, 1: 7-24, 2: 25-71, 3: >= 72

if MODE == "weights":
    os.makedirs(O, exist_ok=True); age = np.full(len(cond), -1, int)
    from t2_eval_oof import T
    nchk = bad = 0
    for p in sorted(set(panel)):
        P = load(p, "train"); Qm = queue_matrix(P)
        for k in np.flatnonzero(og & (panel == p)):
            t = int(tt[k]); hist = Qm[max(0, t - 72):t].any(axis=1)[::-1]
            age[k] = int(np.argmin(hist)) if not hist.all() else 72
            if nchk < 400 and k % 37 == 0:                       # alignment: the labels are the queue rows t+1..t+6 on eligible cells
                lab, el = T.LAB[k]; e = el.astype(bool); nchk += 1; bad += int((lab.astype(bool)[e] != Qm[t + 1:t + 7][e]).any())
    print(f"alignment: {nchk - bad} / {nchk} windows match", flush=True); assert bad == 0
    b = binof(age); tr6 = np.isin(mon, [f"2030-{m:02d}" for m in range(6, 12)])
    for nm, m in (("all months", np.ones(len(cond), bool)), ("fold 2 training months 2030-06..11", tr6), ("read months 2030-12..2031-02", ~tr6)):
        sp = [np.mean(b[og & ~is_sel & m] == i) for i in range(4)]; ss = [np.mean(b[og & is_sel & m] == i) for i in range(4)]
        print(f"{nm}: pool n {int((og & ~is_sel & m).sum())} {dict(zip(NAMES, np.round(sp, 3)))} | selector n {int((og & is_sel & m).sum())} {dict(zip(NAMES, np.round(ss, 3)))}", flush=True)
    sp = np.array([np.mean(b[og & ~is_sel & tr6] == i) for i in range(4)]); ss = np.array([np.mean(b[og & is_sel & tr6] == i) for i in range(4)])
    r = ss / sp; print("ratio per bin (fold 2 training months):", dict(zip(NAMES, np.round(r, 3))), flush=True)
    aw = np.ones(len(cond), np.float32); aw[og] = r[b[og]]
    np.save(O + "age.npy", age); np.save(O + "aw.npy", aw); print("written", O, flush=True)

if MODE == "read":
    from t2_eval_oof import T
    from t2_model import best_set
    age = np.load(O + "age.npy"); ld = lambda tag: [pickle.load(open(f"C:/tfb_data/tfb_nn/{d_}/oof_{tag}_s{s}.pkl", "rb")) for d_, s in (("b0L" if tag == "b0L" else "c15", 0), ("b0L" if tag == "b0L" else "c15", 1))]
    A, B, RF = ld("a0L"), ld("awL"), ld("b0L"); rows = []
    aw = np.load(O + "aw.npy")
    for k in np.flatnonzero(og & np.isin(mon, ["2030-12", "2031-01", "2031-02"])):   # 09:13 addendum: the pool's evaluation windows too (raw / raked)
        if k not in A[0] or k not in B[0]:
            continue
        lab, el = T.LAB[k]; y = lab.astype(bool); e = el.astype(bool); r = dict(k=k, sel=bool(is_sel[k]), w=float(aw[k]), panel=panel[k], day=date[k], mon=mon[k], age=age[k], bin=NAMES[binof(age[k])])
        for nm, X in (("base", A), ("new", B), ("ref", RF)):
            pb = (X[0][k] + X[1][k]) / 2; s = best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape)
            u = ((s | y) & e).sum(); r[nm] = 1.0 if u == 0 else float((s & y & e).sum() / u)
        rows.append(r)
    R = pd.DataFrame(rows); R["d"] = R.new - R.base; R.to_csv(O + "read.csv", index=False); rng = np.random.default_rng(0)

    def bs(d, wt=None):
        wt = np.ones(len(d)) if wt is None else wt; cl = pd.factorize(d.panel + "_" + d.day)[0]; n = cl.max() + 1
        M = np.bincount(cl, wt * d.d.to_numpy(), n); N = np.bincount(cl, wt, n)
        Rm = rng.multinomial(n, np.full(n, 1 / n), size=2000); return M.sum() / N.sum(), ((Rm @ M) / (Rm @ N)).std()
    S, P = R[R.sel], R[~R.sel]
    print("selector windows (the official composition by construction)", flush=True)
    for nm, d in [("all", S)] + [(m, S[S.mon == m]) for m in ("2030-12", "2031-01", "2031-02")] + [("2031-01 + 02", S[S.mon >= "2031-01"])] + [(f"age {b_}", S[S.bin == b_]) for b_ in NAMES]:
        if len(d):
            m_, se = bs(d); print(f"{nm:14s} n {len(d):5d} ref {d.ref.mean():.4f} base {d.base.mean():.4f} new - base {m_:+.4f} (SE {se:.4f}, z {m_ / se:+.2f}, /1.25 {m_ / se / 1.25:+.2f})", flush=True)
    print("pool evaluation windows: raw | raked to the selector age composition (weights = the bin ratio)", flush=True)
    for nm, d in [("all", P), ("2031-01 + 02", P[P.mon >= "2031-01"])] + [(f"age {b_}", P[P.bin == b_]) for b_ in NAMES]:
        if len(d):
            m_, se = bs(d); mr, ser = bs(d, d.w.to_numpy())
            print(f"{nm:14s} n {len(d):5d} raw {m_:+.4f} (SE {se:.4f}, z {m_ / se:+.2f}) | raked {mr:+.4f} (SE {ser:.4f}, z {mr / ser:+.2f}, /1.25 {mr / ser / 1.25:+.2f})", flush=True)

if MODE == "m3":
    d_ = "C:/tfb_data/tfb_nn/nn_test_m3v3/"; R_ = np.load(d_ + "R.npy", mmap_mode="r"); o_ = np.load(d_ + "off.npy"); w_ = [str(x) for x in np.load(d_ + "window_id.npy", allow_pickle=True)]
    cov = {x: float(np.asarray(R_[o_[i]:o_[i + 1], :12, 2], float).mean()) for i, x in enumerate(w_)}
    M = pd.read_csv("C:/Users/user/tfb/work/t2/g235_m3.csv"); M = M[M.cond == "queue_ongoing"].dropna(subset=["B0", "AW"]).reset_index(drop=True)
    M["cov"] = M.w.astype(str).map(cov); M["d"] = M.AW - M.B0; M["day"] = M.day.astype(str); M["panel"] = M.panel.astype(str); rng = np.random.default_rng(0)
    qe = M["cov"].quantile(2 / 3)
    for nm, d in (("M3 dev ongoing, all", M), (f"top coverage third (> {qe:.3f})", M[M["cov"] > qe])):
        cl = pd.factorize(d.panel + "_" + d.day)[0]; n = cl.max() + 1; Ms = np.bincount(cl, d.d.to_numpy(), n); N = np.bincount(cl, minlength=n)
        Rm = rng.multinomial(n, np.full(n, 1 / n), size=2000); se = ((Rm @ Ms) / (Rm @ N)).std()
        print(f"{nm:34s} n {len(d):4d} base {d.B0.mean():.4f} AW - base {d.d.mean():+.4f} (SE {se:.4f}, z {d.d.mean() / se:+.2f}, /1.25 {d.d.mean() / se / 1.25:+.2f})", flush=True)
