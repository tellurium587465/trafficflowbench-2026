"""G125 (10/30, ledger #79, strategy J19): predict the onset label size at T+30 per window and decode the top round(N)
links instead of best_set's set. Window features (<= T, from the NN inputs): panel, time of day, weekday, link count, and
max / mean over links of dem30_over_dnmin, dem_level, dq_up, prior_on, q_today, n_queued_hist, nq_hist, q_trend, v_trend.
LightGBM poisson (300 rounds, lr 0.05, 15 leaves; fixed). Trained on the training months' selector-like onset windows
(January / February evaluated with their month left out; March with all months).
  fit : train the three models, evaluate 2031-01 / 2031-02 on the raw OOF (q25_nomem s3/s4) set
  m3  : M3 v3 DEV onset windows, final A's flow (G120 path), base = G108 column A
Also recorded: the oracle (top N_true) and how N-hat tracks the truth. The label size is counted on the decode's
scale (span_count: blanks inside a true run count; 10/30 fix after the first run counted observed links only).
usage: py g125_onset_count.py [fit | m3]"""
import os, sys, time, pickle, importlib.util
from collections import defaultdict
MODE = (sys.argv[1:] or ["fit"])[0]
import numpy as np, pandas as pd, lightgbm as lgb
TM = time.perf_counter; T00 = TM(); R = r"C:/Users/user/tfb/"; WT = R + "work/t2/"
PRM = dict(objective="poisson", learning_rate=0.05, num_leaves=15, min_data_in_leaf=20, seed=0, verbose=-1)
AGG = ("dem30_over_dnmin", "dem_level", "dq_up", "prior_on", "q_today", "n_queued_hist", "nq_hist", "q_trend", "v_trend")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def wfeat(Xw, panel, names):
    ix = {n: i for i, n in enumerate(names)}; f = [PANELS.index(panel), float(Xw[0, ix["tod"]]), float(Xw[0, ix["dow"]]), float(len(Xw))]
    for a in AGG:
        v = np.asarray(Xw[:, ix[a]], float)
        f += [np.nanmax(v), np.nanmean(v)] if np.isfinite(v).any() else [np.nan, np.nan]
    return f


def span_count(y5, e5):
    """the label size on the scale the decode picks from (all links): observed true links, plus the masked links between two
    observed true links with nothing observed-false between them (a run is not broken by a blank)"""
    ix = np.flatnonzero(y5 & e5)
    if not ix.size:
        return 0
    n, a = 0, ix[0]
    for b0, b1 in zip(ix[:-1], ix[1:]):
        if e5[b0 + 1:b1].any():                                 # an observed free link between: a new run
            n += b0 - a + 1; a = b1
    return n + ix[-1] - a + 1


def clean(y5, e5):
    """every link within 3 of an observed true link is observed (the label size is then exact on the decode's scale)"""
    ix = np.flatnonzero(y5 & e5); return bool(ix.size) and all(e5[max(0, j - 3):j + 4].all() for j in ix)


def topn(p5, n):
    s = np.zeros(p5.size, bool); s[np.argsort(-p5, kind="stable")[:max(1, int(n))]] = True; return s


def boot(d, cl, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed); ids = pd.factorize(cl)[0]; M = np.bincount(ids, d); N = np.bincount(ids)
    Rm = rng.multinomial(len(M), np.full(len(M), 1.0 / len(M)), size=n_boot); return float(((Rm @ M) / (Rm @ N)).std())


def iou5(sel5, y5, e5):
    u = ((sel5 | y5) & e5).sum(); return 1.0 if u == 0 else float((sel5 & y5 & e5).sum() / u)


PANELS = sorted(["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S", "D12_I5_N", "D12_I5_S"])   # cachelib.T2_PANELS


def report(D, nm, base, new, orc):
    cl = D.panel.astype(str) + "_" + D.day.astype(str); x = (D[new] - D[base]).to_numpy(); se = boot(x, cl)
    print(f"{nm}: windows {len(D)} | base {D[base].mean():.4f} | top N-hat {D[new].mean():.4f} gain {x.mean():+.4f} (SE {se:.4f}, z {x.mean() / se:+.2f}) "
          f"| oracle top N_true {D[orc].mean():.4f} ({(D[orc] - D[base]).mean():+.4f}) | N-hat mean {D.nhat.mean():.2f} vs truth {D.ntrue.mean():.2f} "
          f"vs best_set {D.nbs.mean():.2f} | corr(N-hat, truth) {np.corrcoef(D.nhat, D.ntrue)[0, 1]:.2f}, corr(best_set, truth) {np.corrcoef(D.nbs, D.ntrue)[0, 1]:.2f}")


if MODE == "fit":
    g = module("g119", R + "review/g119_month_postproc.py"); T = g.T
    names = [str(x) for x in np.load(f"{g.NN}/nn_data_afdem/names.npy", allow_pickle=True)]
    ks = [k for k in np.flatnonzero(T.META["is_sel"]) if str(T.META["cond"][k]) == "queue_onset" and str(T.META["panel"][k]) in PANELS]
    F = np.array([wfeat(np.asarray(g.XN[g.OFFN[k]:g.OFFN[k + 1]]), str(T.META["panel"][k]), names) for k in ks])
    Y = np.array([span_count(T.LAB[k][0][5].astype(bool), T.LAB[k][1][5].astype(bool)) for k in ks]); MONk = g.MON[ks]
    print(f"training onset windows {len(ks)} | months {pd.Series(MONk).value_counts().sort_index().to_dict()} | features {F.shape[1]} | {TM() - T00:.0f}s", flush=True)
    for tag, keep in (("all", np.ones(len(ks), bool)), ("no2031-01", MONk != "2031-01"), ("no2031-02", MONk != "2031-02")):
        lgb.train(PRM, lgb.Dataset(F[keep], Y[keep]), 300).save_model(WT + f"g125_count_{tag}.txt")
    OOF = [pickle.load(open(g.W + f"oof_q25_nomem_s{s}.pkl", "rb")) for s in (3, 4)]; rows = []
    for mo in ("2031-01", "2031-02"):
        mdl = lgb.Booster(model_file=WT + f"g125_count_no{mo}.txt")
        for j in np.flatnonzero(MONk == mo):
            k = ks[j]; lab, el = T.LAB[k]; y5, e5 = lab[5].astype(bool), el[5].astype(bool); p_ = (OOF[0][k] + OOF[1][k]) / 2
            sel = g.decode(p_)[5]; nh = float(mdl.predict(F[j:j + 1])[0]); nt = span_count(y5, e5)
            rows.append(dict(w=int(k), month=mo, panel=str(T.META["panel"][k]), day=str(T.META["date"][k])[:10], base=iou5(sel, y5, e5),
                             new=iou5(topn(p_[5], round(nh)), y5, e5), orc=iou5(topn(p_[5], nt), y5, e5), nhat=nh, ntrue=nt, nbs=int(sel.sum()), clean=clean(y5, e5)))
    D = pd.DataFrame(rows); D.to_csv(WT + "g125_onset_count_fit.csv", index=False); print(f"fit + evaluation {TM() - T00:.0f}s")
    for mo in ("2031-01", "2031-02"):
        report(D[D.month == mo], f"{mo} raw (month left out)", "base", "new", "orc")
        report(D[(D.month == mo) & D.clean], f"{mo} raw, clean label rows only", "base", "new", "orc")
else:
    g = module("g120", R + "review/g120_onset_gate.py"); G, m3v2, PO, best_set = g.G, g.m3v2, g.PO, g.best_set
    names = g.NAMES; mdl = lgb.Booster(model_file=WT + "g125_count_all.txt")
    PN = g.nn_blend(); phys = lgb.Booster(model_file=g.W + "physon_base.txt"); ev = PO.month_events("validation", PO.rarity_table()); rows = []
    for w in g.DEV:
        m = G.META[w]; i = G.POS[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); ph = phys.predict(PO.feats(Xw)); a = PO.adaptive_a(ev, m["day"])
        nh = float(mdl.predict(np.array([wfeat(Xw, m["panel"], names)]))[0]); y5, e5 = m["y"].T.astype(bool)[5], m["m"].T.astype(bool)[5]
        r = defaultdict(float, w=w, panel=m["panel"], day=m["day"], nhat=nh, ntrue=span_count(y5, e5), clean=clean(y5, e5))
        for k, pr in m3v2.kdist(m["day"]).items():
            pb = PN[w].copy(); pb[5] = (1 - a) * pb[5] + a * ph; s, nd = m3v2.seen(m["panel"], m["t"], k, -6, 12)
            if nd:
                pb[-1] = pb[-1] * np.where(G.dil(s, 2), 1.0, 0.25)
            sel = best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape)
            r["base"] += pr * G.iou(sel, m); r["nbs"] += pr * sel[5].sum()
            r["new"] += pr * iou5(topn(pb[5], round(nh)), y5, e5); r["orc"] += pr * iou5(topn(pb[5], r["ntrue"]), y5, e5)
        rows.append(dict(r))
    D = pd.DataFrame(rows); ref = pd.read_csv(g.W + "g108_rectangles.csv").set_index("w")["A"]
    print(f"m3: {len(D)} windows in {TM() - T00:.0f}s | identity base vs G108 column A max |d| {float(np.max(np.abs(D.base - ref.loc[D.w].to_numpy()))):.2e}")
    D.to_csv(WT + "g125_onset_count_m3.csv", index=False); report(D, "M3 dev onset (final A)", "base", "new", "orc")
    report(D[D.clean], "M3 dev onset, clean label rows only", "base", "new", "orc")
