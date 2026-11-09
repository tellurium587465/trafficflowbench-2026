"""G120 (10/30, ledger #73, strategy J14): an oracle test of a "metacognition gate" on the onset. M3 v3 DEV onset windows
(the hold-out days are not read), final A's onset flow as G108 column A (q24f 8 seeds CF5 + physics onset with the
adaptive a, evidence x0.25, best_set; k expectation, simulated blanks). Per (window, k) the gate looks at the links S
final A selects at T+30 with the precursor tests of G117 (J11), read on S instead of the label set:
  (a) speed: min over rows T-6..T-1 of the mean over S of V / vff >= 0.98 (vff as G117; all-NaN -> not met)
  (b) upstream: flow over S + the 3 links upstream / the training months' weekday x time-of-day profile in [0.85, 1.15]
  (c) site: the rarity table's T+30 frequency <= 0.1 on every link of S
  gate "ab" (primary: no dynamic precursor at the chosen place = the type-A failure, a confident usual-site pick with
  nothing happening there) and "abc" (J11's literal definition). Empty S -> closed.
Open -> the T+30 set is replaced by
  "67" (primary): the top |S| links of the G67 lambdarank model (work/t2/g67_phys_ranker.txt, train months)
  "ph"          : best_set on the physics onset probabilities alone (physon_base), no evidence rule
Also U03 = a uniform a = 0.3 mix (G70) instead of the adaptive a. Reported: gain vs A, windows opened (k-weighted and
any k), rare-site truths among them (G117 fmax <= 0.1), the loss on false openings (mean over the windows of min(0, d)).
Identity (first): every gate closed reproduces G108's column A per window (max |d| <= 1e-12, dev mean 0.8641).
usage: TFB_M3VER=v3 py g120_onset_gate.py [time | identity | run]"""
import os, sys, time, pickle
os.environ.setdefault("TFB_M3VER", "v3")
sys.path.insert(0, r"C:/Users/user/tfb/src"); MODE = (sys.argv[1:] or ["run"])[0]; sys.argv = [sys.argv[0], "0"]
TM = time.perf_counter; CUR = {}; T00 = TM()
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import rankdata
import t2_phys_onset as PO, t2_m3_gate as G, t2_m3v2_gate as V, m3v2
from t2_model import best_set
from cachelib import load
from t2_demand import profile


def tick(name, t0):
    CUR[name] = CUR.get(name, 0.0) + TM() - t0


tick("import (M3 meta, X)", T00)
W = G.W; ld = lambda f: pickle.load(open(W + f, "rb"))
CF5 = (("attnglob", 2.0), ("attnwp05", 2.0), ("attn1", 2.0), ("drop1", 1.0), ("r13", 1.0))
ON = [(f"q24f_{c}_s{s}", w) for s in range(8) for c, w in CF5]
DEV = [w for w in G.WID if G.META[w]["cond"] == "queue_onset" and not m3v2.is_holdout(np.array([G.META[w]["day"]]))[0]]
PN_CACHE = W + "g120_pn_dev_onset.pkl"


def nn_blend():
    """the final A onset NN blend (as G108) on the dev onset windows; cached once (the 40 probability files are the load)"""
    if os.path.exists(PN_CACHE):
        return pickle.load(open(PN_CACHE, "rb"))
    PRO = {t: ld(f"probs_{G.PRE}__{t}.pkl") for t, _ in ON}
    out = {w: sum(wt * PRO[t][w] for t, wt in ON) / sum(wt for _, wt in ON) for w in DEV}
    pickle.dump(out, open(PN_CACHE, "wb")); return out


ST_CACHE = W + "g120_panel_stats.pkl"


def panel_stats():
    """vff, weekday x time-of-day flow profile and upstream direction per panel from the training months (as G117); cached"""
    if os.path.exists(ST_CACHE):
        return pickle.load(open(ST_CACHE, "rb"))
    VFF, PROF, UP = {}, {}, {}
    for p in sorted({G.META[w]["panel"] for w in DEV}):
        PT = load(p, "train"); ff = np.where(PT.E & (PT.V >= 0.85 * PT.vf[None, :]), PT.V, np.nan)
        VFF[p] = np.nanmedian(ff, axis=0); PROF[p] = profile(PT)
        li = {l: i for i, l in enumerate(PT.links)}; sg = []
        for j, s in enumerate(PT.topo.incoming_link_ids.fillna("").astype(str)):
            sg += [np.sign(li[x] - j) for x in s.split(";") if x in li]
        UP[p] = int(np.sign(np.median(sg))); load.cache_clear()
    out = (VFF, PROF, UP); pickle.dump(out, open(ST_CACHE, "wb")); return out


NAMES = [str(x) for x in np.load(f"{G.NN}/nn_data_afdem/names.npy", allow_pickle=True)]
DROP = {"prior_on", "prior_og", "pos", "vf", "capl", "lanes", "length", "w", "tod", "dow", "arr5", "cond"}
COLS = [i for i, n in enumerate(NAMES) if n not in DROP]


def feats67(Xw):                                                 # g67.feats
    X = Xw[:, COLS].astype(float); P = np.full_like(X, 0.5)
    for c in range(X.shape[1]):
        v = X[:, c]; ok = np.isfinite(v)
        if ok.sum() >= 2 and np.nanstd(v) > 0:
            P[ok, c] = (rankdata(v[ok]) - 0.5) / ok.sum()
    return np.column_stack([P, X])


def precursor(S, m, P, VFF, PROF, UP, F_):
    """(a, b, c) on the selected set S (bool n) of window m: True = the test of 'no precursor' is met"""
    t = m["t"]; p = m["panel"]; Y = np.flatnonzero(S)
    with np.errstate(all="ignore"):
        a = bool(np.nanmin(np.nanmean(P.V[t - 6:t][:, Y] / VFF[p][Y][None, :], axis=1)) >= 0.98)
        lo, hi = Y.min(), Y.max(); R = np.r_[Y, np.arange(lo - 3, lo) if UP[p] < 0 else np.arange(hi + 1, hi + 4)]
        R = R[(R >= 0) & (R < P.n)]
        dow = pd.Timestamp(P.times[t][:10]).dayofweek; pr = PROF[p][dow][(np.arange(t - 6, t) % 288)][:, R]
        b = bool(0.85 <= np.nanmean(P.Q[t - 6:t][:, R]) / np.nanmean(pr) <= 1.15)
    key = (p, dow >= 5, (t % 288) // 24); f = F_.get(key, np.zeros(P.n)); c = bool(f[Y].max() <= 0.1)
    return a, b, c


def run(limit=None):
    t0 = TM(); PN = nn_blend(); tick("load NN probs (40 files -> cache)", t0)
    t0 = TM(); VFF, PROF, UP = panel_stats(); tick("load panel stats (train months -> cache)", t0)
    t0 = TM(); phys = lgb.Booster(model_file=W + "physon_base.txt"); r67 = lgb.Booster(model_file=W + "g67_phys_ranker.txt")
    F_ = PO.rarity_table(); ev = PO.month_events("validation", F_); tick("load models, rarity, March events", t0)
    O117 = pd.read_csv(W + "g117_onset_oracle.csv").set_index("w"); rows = []
    for w in DEV[:limit]:
        m = G.META[w]; i = G.POS[w]; t0 = TM(); Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); P, _, _ = m3v2._panel(m["panel"]); tick("window inputs", t0)
        t0 = TM(); ph = phys.predict(PO.feats(Xw)); a_ph = PO.adaptive_a(ev, m["day"]); tick("physics onset", t0)
        t0 = TM(); s67 = r67.predict(feats67(Xw)); tick("G67 candidates", t0)
        pp = np.zeros((6, P.n)); pp[5] = ph; t0 = TM(); sel_ph = best_set(pp.ravel(), np.ones(pp.size, bool)).reshape(pp.shape); tick("decode", t0)
        r = dict(w=w, panel=m["panel"], day=m["day"], a=a_ph, A=0.0, U03=0.0, open_ab=0.0, open_abc=0.0, any_ab=False, any_abc=False)
        for g in ("ab", "abc"):
            for rp in ("67", "ph"):
                r[f"{g}_{rp}"] = 0.0
        for k, pr in m3v2.kdist(m["day"]).items():
            p = PN[w].copy(); p[5] = (1 - a_ph) * p[5] + a_ph * ph
            t0 = TM(); sel = V.onset_pred(p, m, "cur", k); tick("decode", t0)
            pu = PN[w].copy(); pu[5] = 0.7 * pu[5] + 0.3 * ph
            t0 = TM(); sel_u = V.onset_pred(pu, m, "cur", k); tick("decode", t0)
            S = sel[-1]; t0 = TM(); gt = precursor(S, m, P, VFF, PROF, UP, F_) if S.any() else (False, False, False); tick("precursor", t0)
            op = {"ab": gt[0] and gt[1], "abc": gt[0] and gt[1] and gt[2]}
            top = np.zeros_like(sel); top[-1, np.argsort(-s67)[:int(S.sum())]] = True
            t0 = TM(); base = G.iou(sel, m); r["A"] += pr * base; r["U03"] += pr * G.iou(sel_u, m)
            alt = {"67": G.iou(top, m), "ph": G.iou(sel_ph, m)}
            for g, o in op.items():
                r[f"open_{g}"] += pr * o; r[f"any_{g}"] |= o
                for rp in ("67", "ph"):
                    r[f"{g}_{rp}"] += pr * (alt[rp] if o else base)
            tick("scoring", t0)
        r["rare_truth"] = bool(O117.loc[w, "fmax"] <= 0.1) if w in O117.index and np.isfinite(O117.loc[w, "fmax"]) else False
        rows.append(r)
    return pd.DataFrame(rows)


def boot(d, cl, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed); ids = pd.factorize(cl)[0]; M = np.bincount(ids, d); N = np.bincount(ids)
    R = rng.multinomial(len(M), np.full(len(M), 1.0 / len(M)), size=n_boot); return float(((R @ M) / (R @ N)).std())


if __name__ == "__main__":
    lim = 1 if MODE == "time" else None
    D = run(lim); ref = pd.read_csv(W + "g108_rectangles.csv").set_index("w")["A"]
    dmax = float(np.max(np.abs(D.A.to_numpy() - ref.loc[D.w].to_numpy())))
    print(f"identity (all gates closed = G108 column A): max |d| {dmax:.2e} over {len(D)} windows | mean A {D.A.mean():.4f}")
    print("time per stage (s):", {k: round(v, 2) for k, v in CUR.items()}, f"| total {TM() - T00:.1f}s")
    if MODE == "time":
        per = sum(v for k, v in CUR.items() if not k.startswith(("import", "load")))
        print(f"per window {per:.2f}s -> 100 windows {per * 100 / 60:.1f} min + loads"); sys.exit()
    if dmax > 1e-12:
        print("IDENTITY FAILED: stop"); sys.exit(1)
    if MODE == "identity":
        sys.exit()
    D.to_csv(W + "g120_onset_gate.csv", index=False); cl = D.panel + "_" + D.day.astype(str)
    print(f"dev onset windows {len(D)} | rare-site truths {int(D.rare_truth.sum())}")
    for g in ("ab", "abc"):
        for rp in ("67", "ph"):
            d = (D[f"{g}_{rp}"] - D.A).to_numpy(); op = D[f"any_{g}"]; loss = np.minimum(d, 0).mean(); gain = d.mean()
            print(f"gate {g:3s} -> {rp}: gain {gain:+.4f} (SE {boot(d, cl):.4f}) | opened: k-weighted {D[f'open_{g}'].sum():.1f}, any k {int(op.sum())} "
                  f"(rare truth {int((op & D.rare_truth).sum())}) | wins {int((d > 1e-12).sum())} losses {int((d < -1e-12).sum())} | "
                  f"false-open loss {loss:+.4f} | gain on rare-truth windows {d[D.rare_truth.to_numpy()].sum() / len(D):+.4f}")
    d = (D.U03 - D.A).to_numpy()
    print(f"uniform a = 0.3 (G70) vs adaptive a: {d.mean():+.4f} (SE {boot(d, cl):.4f}) | windows changed {int((np.abs(d) > 1e-12).sum())}")
