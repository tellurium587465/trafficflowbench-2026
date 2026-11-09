"""G341b (#272 control): the March / April l1 is read on data with Task 1 blanks (Q finite ~0.57) while the training months are ~0.83. Control:
a training month of the same length with the test month's own NaN pattern laid on row by row (January <- March mask, November <- April mask),
its shape against the pool of the other training months. usage: py -B -X utf8 g341b_mask_control.py"""
import sys, importlib.util, numpy as np
sys.argv = ["x"]; sp = importlib.util.spec_from_file_location("g341", r"C:/Users/user/tfb/review/g341_demand_shape.py"); src = open(sp.origin, encoding="utf-8").read()
src = src[:src.index("for p in T2_PANELS:")]; g = {}; exec(compile(src, "g341", "exec"), g)
load, T2_PANELS, shape, cmp = g["load"], g["T2_PANELS"], g["shape"], g["cmp"]; out = {}
for p in T2_PANELS:
    P = load(p, "train"); mon = np.array([s[:7] for s in P.times]); MS = sorted(set(mon)); SH = {m: shape(P.Q[mon == m], [s for s in P.times if s.startswith(m)]) for m in MS}
    for tm, sp_ in (("2031-01", "validation"), ("2030-11", "private")):
        Pv = load(p, sp_); rr = np.flatnonzero(mon == tm); n = min(len(rr), len(Pv.times)); Qm = P.Q[rr[:n]].copy(); Qm[~np.isfinite(Pv.Q[:n])] = np.nan
        S = shape(Qm, [P.times[r] for r in rr[:n]]); pl = np.nanmean(np.stack([SH[m] for m in MS if m != tm]), axis=0)
        out.setdefault(tm, []).extend([cmp(S[:, j], pl[:, j])[1] for j in range(P.n)]); out.setdefault(tm + " unmasked", []).extend([cmp(SH[tm][:, j], pl[:, j])[1] for j in range(P.n)])
    load.cache_clear()
for k, v in out.items():
    print(f"{k}: l1 {np.nanmean(v):.4f}", flush=True)
