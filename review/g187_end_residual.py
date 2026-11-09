"""G187 (11/01, ledger #141 stage 0, strategy T2_DEEP_REVIEW 18): the end of a jam on top of the ongoing probabilities. Links of the runs queued at the last visible row, every horizon. Features of the run: flow / capacity at the link past the head, its 30-minute slope, flow into the tail / flow past the head, time of day, run length, position in the run, the horizon.
LightGBM on logit(y) with init score logit(p), 5 folds by window, log loss per cell, window bootstrap. p: final A's mix on M3 dev, the OOF on January and February. Line: improvement >= 2 SE with the same sign on M3 dev and on January + February.
usage: TFB_M3VER=v3 py -B -X utf8 g187_end_residual.py"""
import os, sys, pickle
R = r"C:/Users/user/tfb/review/"; sys.dont_write_bytecode = True
src = open(R + "g119_month_postproc.py", encoding="utf-8").read().split('if __name__ == "__main__":')[0]
sys.argv = [sys.argv[0], "run"]; exec(compile(src, "g119", "exec"), globals())
import t2_final_spec as FS
XA = [pickle.load(open(W + f"probs_m3{G.VER}v2x1__c24f_attnglob_s{s}.pkl", "rb")) for s in (0, 1)]; X1 = {key: (XA[0][key] + XA[1][key]) / 2 for key in XA[0]}; PL = [pickle.load(open(W + f"probs_m3v3__tfm4_s{s}.pkl", "rb")) for s in range(5)]; names = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]; I = {c: names.index(c) for c in ("last_q", "q_trend", "tod")}
lg_ = lambda x: np.log(np.clip(x, 1e-5, 1 - 1e-5) / (1 - np.clip(x, 1e-5, 1 - 1e-5)))


def runs(v):
    dd = np.diff(np.r_[0, np.asarray(v, int), 0]); return list(zip(np.flatnonzero(dd == 1), np.flatnonzero(dd == -1) - 1))


def cells(w, inst, p, y, e, Xw, lq, rows):
    n = len(lq); q = np.nan_to_num(Xw[:, I["last_q"]]); tr = np.nan_to_num(Xw[:, I["q_trend"]]); tod = float(np.nanmedian(Xw[:, I["tod"]]))
    for a, b in runs(lq):
        hd = min(b + 1, n - 1); tl = max(a - 1, 0); ii, jj = np.nonzero(e[:, a:b + 1]); jj = jj + a
        if len(ii):
            rows.append(pd.DataFrame(dict(inst=inst, w=str(w), h=ii.astype(np.int8), lp=lg_(p[ii, jj]).astype(np.float32), y=y[ii, jj].astype(np.int8), qh=q[hd], sl=tr[hd], ra=q[tl] / max(q[hd], 1e-3), tod=tod, ln=b - a + 1, pos=(b - jj) / max(b - a, 1))))


rows = []; DEV = [w for w in G.WID if G.META[w]["cond"] == "queue_ongoing" and not m3v2.is_holdout(np.array([G.META[w]["day"]]))[0]]
for w in DEV:
    m = G.META[w]; i = G.POS[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]], float); lq = Xw[:, G.CI] == 1; pl = FS.lgb_ongoing(np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]])); kd = m3v2.kdist(m["day"]); pt = sum(d[w] for d in PL) / 5; p = sum(pr * (0.8 * (0.8 * X1[(w, k)] + 0.2 * pt) + 0.2 * pl) for k, pr in kd.items()); cells(w, "2031-03", p, m["y"].T.astype(bool), m["m"].T.astype(bool), Xw, lq, rows)
OOF = [pickle.load(open(W + f"oof_q25_nomem_s{s}.pkl", "rb")) for s in (3, 4)]
for mo in MONTHS:
    for k in [k for k in np.flatnonzero(T.META["is_sel"] & (MON == mo)) if str(T.META["cond"][k]) == "queue_ongoing"]:
        lab, el = T.LAB[k]; Xw = np.asarray(XN[OFFN[k]:OFFN[k + 1]], float); cells(k, mo, (OOF[0][k] + OOF[1][k]) / 2, lab.astype(bool), el.astype(bool), Xw, Xw[:, G.CI] == 1, rows)
D = pd.concat(rows, ignore_index=True); print(D.groupby("inst").agg(cells=("y", "size"), windows=("w", "nunique"), y=("y", "mean")).round(3).to_string(), flush=True); FE = ["h", "qh", "sl", "ra", "tod", "ln", "pos"]; PRM = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=200, lambda_l2=5.0, feature_fraction=0.8, verbose=-1, num_threads=4)


def test(g_, name):
    ws = g_.w.unique(); fold = dict(zip(ws, np.random.default_rng(0).integers(5, size=len(ws)))); f = g_.w.map(fold).to_numpy(); y = g_.y.to_numpy(float); lp = g_.lp.to_numpy(float); X = g_[FE].to_numpy(float); o = np.zeros(len(g_))
    for k in range(5):
        a = f != k; d = lgb.Dataset(X[a], y[a], init_score=lp[a], free_raw_data=False); m = lgb.train(PRM, d, 200); o[~a] = m.predict(X[~a], raw_score=True)
    ll = lambda s: np.logaddexp(0, -(2 * y - 1) * s); d_ = ll(lp) - ll(lp + o); ids = pd.factorize(g_.w)[0]; M = np.bincount(ids, d_); N = np.bincount(ids); Rm = np.random.default_rng(0).multinomial(len(M), np.full(len(M), 1 / len(M)), size=2000); s = ((Rm @ M) / (Rm @ N)).std(); print(f"{name}: {len(ws)} windows, {len(g_)} cells | log loss {ll(lp).mean():.5f} -> {ll(lp + o).mean():.5f} | improvement {d_.mean():+.6f} (SE {s:.6f}, z {d_.mean() / s:+.2f})", flush=True)


test(D[D.inst == "2031-03"], "March dev"); test(D[D.inst.isin(MONTHS)], "January + February")
