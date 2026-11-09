"""G180 (11/01, ledger #129, stage 0): does the Task 2 network's probability of the jam going on explain the speed error of the blank model inside the T2 blanks? Feb harness (g168 base cache: f3big + gapml_cv), cells of links queued at the last visible row, target cells inside the simulated
blanks with a truth. Residual r = (truth - base speed) / vf. Features: the OOF probabilities of q25_nomem seeds 3, 4 at the link for the horizon rows (p1 .. p6, rows after T+30 take p6), the rows after T, base speed / vf. LightGBM, 5 folds by window, CV R2 against the mean, and the same without the probabilities.
Line: R2 with the probabilities >= 0.05 and above the one without them. Otherwise closed.
usage: TFB_M3VER=v3 py -B -X utf8 g180_vanish_gate.py"""
import sys, pickle
R = r"C:/Users/user/tfb/review/"; sys.dont_write_bytecode = True
src = open(R + "g119_month_postproc.py", encoding="utf-8").read().split('if __name__ == "__main__":')[0]
sys.argv = [sys.argv[0], "run"]; exec(compile(src, "g119", "exec"), globals())
OOF = [pickle.load(open(W + f"oof_q25_nomem_s{s}.pkl", "rb")) for s in (3, 4)]; O = "C:/tfb_data/tfb_nn/unet_h/"; Z = np.load(W + "gapml_windows.npz", allow_pickle=True); WM = Z["is_sel"]; GG = 18
key = {(str(T.META["panel"][k]), int(T.META["t"][k])): k for k in np.flatnonzero(MON == "2031-02") if k in OOF[0] and k in OOF[1]}; rows = []; miss = 0
for p in sorted({str(x) for x in Z["panel"]}):
    P0 = load(p, "train"); f = np.flatnonzero([s.startswith("2031-02") for s in P0.times]); a, b = f[0], f[-1] + 1; z = np.load(O + p + ".npz"); V0 = z["V0"]; gap = z["gap"]; V = P0.V[a:b]; Tg = P0.target[a:b] & P0.E[a:b] & np.isfinite(V)
    wl = sorted({int(t) - a for pp, t in zip(Z["panel"][WM], Z["t"][WM]) if str(pp) == p and 30 <= int(t) - a < (b - a) - GG - 31})[:10]; vis = np.where(P0.blank[a:b], np.nan, V)
    for t in wl:
        k = key.get((p, t + a))
        if k is None:
            miss += 1; continue
        pr = (OOF[0][k] + OOF[1][k]) / 2; h = np.where(np.isfinite(V[t - 12:t]), V[t - 12:t], np.nan); idx = np.where(np.isfinite(h), np.arange(12)[:, None], -1).max(axis=0); last = np.where(idx >= 0, h[np.maximum(idx, 0), np.arange(P0.n)], np.nan); lq = np.nan_to_num(last, nan=1e9) <= P0.vcut
        for dt in range(1, GG + 1):
            r_ = t + dt; j = np.flatnonzero(lq & Tg[r_] & np.isfinite(V0[r_]))
            if len(j):
                rows.append(pd.DataFrame(dict(w=f"{p}_{t}", dt=dt, res=(V[r_, j] - V0[r_, j]) / P0.vf[j], base=V0[r_, j] / P0.vf[j], p1=pr[0, j], p2=pr[1, j], p3=pr[2, j], p4=pr[3, j], p5=pr[4, j], p6=pr[5, j])))
    load.cache_clear(); print(p, len(wl), flush=True)
D = pd.concat(rows, ignore_index=True); print("cells", len(D), "windows", D.w.nunique(), "missing windows", miss, "res sd", round(D.res.std(), 4), "rmse km/h ~", round(float(D.res.std() * 115), 2)); PRM = dict(objective="regression", learning_rate=0.05, num_leaves=15, min_data_in_leaf=50, verbose=-1, num_threads=4, feature_fraction=0.8)
ws = D.w.unique(); fold = dict(zip(ws, np.random.default_rng(0).integers(5, size=len(ws)))); f = D.w.map(fold).to_numpy(); y = D.res.to_numpy()
for nm, cols in (("without p", ["dt", "base"]), ("with p", ["dt", "base", "p1", "p2", "p3", "p4", "p5", "p6"])):
    pr = np.zeros(len(D))
    for k in range(5):
        m = lgb.train(PRM, lgb.Dataset(D.loc[f != k, cols].to_numpy(), y[f != k]), 150); pr[f == k] = m.predict(D.loc[f == k, cols].to_numpy())
    print(f"{nm}: CV R2 {1 - ((y - pr) ** 2).sum() / ((y - y.mean()) ** 2).sum():+.4f}", flush=True)
