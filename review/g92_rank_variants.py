"""G92 (10/23, strategy §16-4, the rare-site onset push): improve the location-agnostic onset model (physon_base: binary
LightGBM on within-window percentiles of the non-memory inputs, train-month onset windows with a T+30 queue).
Variants (all trained on the same rows as physon_base, nn_data_afdem):
  base   : physon_base (the production model file)
  rank   : LightGBM lambdarank (one group per window), its score turned into a probability by a logistic fit on the
           training rows (in-sample; only the monotone map is fitted)
  marg   : base features + the link's own speeds against ITS fixed queue threshold vcut (v - vcut in km/h and v / vcut, raw
           and within-window percentile; v = v_T, v_T1, v_T2, v_min, last_v, v_m3, v_m12): how close each link is to
           breaking down, a physical quantity that does not name the link
  rankm  : rank + marg
Mixed into the production onset blend at T+30 in the production order (NN -> physics -> evidence rule -> decoder), two
ways: lin p = (1 - a) p_NN + a p_ph; logit: logit p = (1 - a) logit p_NN + a logit p_ph; a = the month-adaptive a (causal,
PO.adaptive_a on the month's completed earlier days).
Yardsticks: (1) M3 v3 dev onset windows (hold-out excluded; onset blend = q24f 8 seeds, the final March / B onset;
t2_m3v2_gate.score: evidence rule, decoder, k expectation); (2) M5 = the synthetic C24 onset windows (G73: activated =
every true link rare in the real train months; NN = the s46 blend on them, decoder on eligible cells, no evidence rule,
fixed a = 0.3 as G73), paired vs base, SE clustered by panel x synthetic month.
usage: python g92_rank_variants.py"""
import sys, os, json, pickle
os.environ.setdefault("TFB_M3VER", "v3"); os.environ["TFB_T2CACHE"] = "2_noT_af"
sys.path.insert(0, r"C:/Users/user/tfb/src"); argv0 = sys.argv[0]
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import rankdata
import t2_phys_onset as PO
sys.argv = [argv0, "0"]
import t2_m3_gate as G, t2_m3v2_gate as V, m3v2
from cachelib import load
from common import NN
from t2_model import best_set

W = G.W; NAMES = PO.NAMES; VC = {}
MV = [NAMES.index(c) for c in ("v_T", "v_T1", "v_T2", "v_min", "last_v", "v_m3", "v_m12")]


def vcut(panel):
    if panel not in VC:
        VC[panel] = load(panel, "train").vcut.astype(float)
    return VC[panel]


def pct(v):
    ok = np.isfinite(v); out = np.full(len(v), 0.5)
    if ok.sum() >= 2 and np.nanstd(v) > 0:
        out[ok] = (rankdata(v[ok]) - 0.5) / ok.sum()
    return out


def feats_m(Xw, vc):
    v = Xw[:, MV].astype(float); d = v - vc[:, None]; r = v / vc[:, None]
    return np.column_stack([PO.feats(Xw), d, r] + [pct(d[:, j]) for j in range(d.shape[1])])


# training rows (as PO.rows("nn_data_afdem"), plus the margins and the window groups)
d_ = f"{NN}/nn_data_afdem"
X = np.load(f"{d_}/X.npy", mmap_mode="r"); Y = np.load(f"{d_}/Y.npy", mmap_mode="r"); E = np.load(f"{d_}/EL.npy", mmap_mode="r")
off = np.load(f"{d_}/off.npy"); cond = np.load(f"{d_}/cond.npy", allow_pickle=True); pan = np.load(f"{d_}/panel.npy", allow_pickle=True)
Fb, Fm, ys, grp = [], [], [], []
for k in np.flatnonzero(cond == "queue_onset"):
    a, b = off[k], off[k + 1]; y = np.asarray(Y[a:b, 5]).astype(bool); e = np.asarray(E[a:b, 5]).astype(bool)
    if (y & e).any():
        Xw = np.asarray(X[a:b]); fm = feats_m(Xw, vcut(str(pan[k])))
        Fm.append(fm[e]); Fb.append(fm[e][:, :len(PO.COLS)]); ys.append(y[e].astype(int)); grp.append(int(e.sum()))
print("train onset windows", len(ys), "rows", sum(grp), flush=True)
yy = np.concatenate(ys)
RPRM = dict(PO.PRM, objective="lambdarank", lambdarank_truncation_level=10, eval_at=[1, 3, 5])
MODELS = {"base": lgb.Booster(model_file=W + "physon_base.txt")}
MODELS["marg"] = lgb.train(PO.PRM, lgb.Dataset(np.vstack(Fm), yy), 400)
CAL = {}
for name, F in (("rank", Fb), ("rankm", Fm)):
    MODELS[name] = lgb.train(RPRM, lgb.Dataset(np.vstack(F), yy, group=grp), 400)
    s = MODELS[name].predict(np.vstack(F)); from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6).fit(s[:, None], yy); CAL[name] = (float(lr.coef_[0, 0]), float(lr.intercept_[0]))
    print(name, "logistic map of the score:", CAL[name], flush=True)
for n_, m_ in MODELS.items():
    if n_ != "base":
        m_.save_model(W + f"g92_{n_}.txt")
pickle.dump(CAL, open(W + "g92_cal.pkl", "wb"))


def predict(name, Xw, vc):
    F = feats_m(Xw, vc); F = F if name in ("marg", "rankm") else F[:, :len(PO.COLS)]
    s = MODELS[name].predict(F)
    return 1 / (1 + np.exp(-(CAL[name][0] * s + CAL[name][1]))) if name in CAL else s


lgt = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
mix = {"lin": lambda pn, ph, a: (1 - a) * pn + a * ph, "logit": lambda pn, ph, a: 1 / (1 + np.exp(-((1 - a) * lgt(pn) + a * lgt(ph))))}

# (1) M3 v3 dev onset windows
CF5 = (("attnglob", 2.0), ("attnwp05", 2.0), ("attn1", 2.0), ("drop1", 1.0), ("r13", 1.0))
parts = [(f"q24f_{c}_s{s}", w) for s in range(8) for c, w in CF5]
PR = {t: pickle.load(open(W + f"probs_{G.PRE}__{t}.pkl", "rb")) for t, _ in parts}
XM = np.load(f"{NN}/nn_test_m3{G.VER}/X.npy", mmap_mode="r"); OFFM = np.load(f"{NN}/nn_test_m3{G.VER}/off.npy")
ev = PO.month_events("validation", PO.rarity_table())
WON = [w for w in G.WID if G.META[w]["cond"] == "queue_onset" and not m3v2.is_holdout(G.META[w]["day"])]
cols = {"nn": {}} | {f"{n_}_{mx}": {} for n_ in MODELS for mx in mix}
for w in WON:
    m = G.META[w]; i = G.WID.index(w); Xw = np.asarray(XM[OFFM[i]:OFFM[i + 1]]); vc = vcut(m["panel"])
    p = sum(wt * PR[t][w] for t, wt in parts) / sum(wt for _, wt in parts); a = PO.adaptive_a(ev, m["day"])
    cols["nn"][w] = V.score(p, m)
    for n_ in MODELS:
        ph = predict(n_, Xw, vc)
        for mx, f in mix.items():
            q = p.copy(); q[5] = f(p[5], ph, a); cols[f"{n_}_{mx}"][w] = V.score(q, m)
D = V.frame(cols).dropna(subset=["nn"])
G64 = pd.read_csv(W + "g64_onset_rare_diag.csv").set_index("w"); D["rare"] = D.w.map(G64.rare_true)
D.to_csv(W + f"g92_rank_m3{G.VER}.csv", index=False)
print(f"M3 {G.VER} dev onset windows {len(D)} (rare truth {int((D.rare == True).sum())})")
for c in [c for c in cols if c != "nn"]:
    s = m3v2.summary(D, "nn", c, "private"); d = D[c] - D.base_lin
    print(m3v2.fmt(f"NN -> {c}", s), f"| vs base_lin {d.mean():+.4f} (rare {d[D.rare == True].mean():+.4f}, common {d[D.rare == False].mean():+.4f})")

# (2) M5: synthetic C24 onset windows (G73)
S = f"{NN}/nn_data_twinC24"; DT = f"{NN}/nn_test_c24on"; ks = pickle.load(open(W + "g73_c24on_ks.pkl", "rb"))
sY = np.load(f"{S}/Y.npy", mmap_mode="r"); sE = np.load(f"{S}/EL.npy", mmap_mode="r"); soff = np.load(f"{S}/off.npy")
span = np.load(f"{S}/panel.npy", allow_pickle=True); stt = np.load(f"{S}/t.npy"); sdate = np.load(f"{S}/date.npy", allow_pickle=True)
F_ = PO.rarity_table(); spec = json.load(open(r"C:/Users/user/tfb/src/t2_specs.json"))["s37"]["spec"]
sparts = [(x.split(":")[0], float(x.split(":")[1]) if ":" in x else 1.0) for x in spec.split("+")]
SPR = {t: pickle.load(open(W + f"probs_c24on__{t}.pkl", "rb")) for t, _ in sparts}
XT = np.load(f"{DT}/X.npy", mmap_mode="r"); OT = np.load(f"{DT}/off.npy"); rows = []
for i, k in enumerate(ks):
    w = f"C24_{k}"; Xw = np.asarray(XT[OT[i]:OT[i + 1]]); pnl = str(span[k])
    y = np.asarray(sY[soff[k]:soff[k + 1], 5]).astype(bool); e = np.asarray(sE[soff[k]:soff[k + 1], 5]).astype(bool)
    key = (pnl, pd.Timestamp(str(sdate[k])[:10]).dayofweek >= 5, (int(stt[k]) % 288) // 24)
    f = F_.get(key, np.zeros(len(y))); tl = np.flatnonzero(y & e)
    pn = sum(wt * SPR[t][w][5] for t, wt in sparts) / sum(wt for _, wt in sparts)
    r = dict(k=k, panel=pnl, month=str(sdate[k])[:7], act=bool((f[tl] <= 0.1).all()))
    s_ = best_set(pn, e); u = ((s_ | y) & e).sum(); r["nn"] = 1.0 if u == 0 else float((s_ & y & e).sum() / u)
    for n_ in MODELS:
        ph = predict(n_, Xw, vcut(pnl))
        o = np.argsort(-np.where(e, ph, -1.0)); rr = np.empty(len(o), int); rr[o] = np.arange(len(o)); r[f"rk_{n_}"] = int(rr[tl].min())
        for mx, fn in mix.items():
            s_ = best_set(fn(pn, ph, 0.3), e); u = ((s_ | y) & e).sum(); r[f"{n_}_{mx}"] = 1.0 if u == 0 else float((s_ & y & e).sum() / u)
    rows.append(r)
R = pd.DataFrame(rows); R.to_csv(W + "g92_rank_m5.csv", index=False)


def paired(g, a_, b_):
    d = g[b_] - g[a_]; cl = g.panel + "|" + g.month; cm = d.groupby(cl).sum(); n = len(d); nc = cl.value_counts()
    se = np.sqrt(len(cm) / max(len(cm) - 1, 1) * ((cm - cm.index.map(nc) * d.mean()) ** 2).sum()) / n
    return d.mean(), se


print(f"M5 C24 onset windows {len(R)}, activated {int(R.act.sum())}")
for name, g in (("activated", R[R.act]), ("usual", R[~R.act])):
    print(f"  {name}: top-1/3 " + ", ".join(f"{n_} {np.mean(g[f'rk_{n_}'] == 0):.2f}/{np.mean(g[f'rk_{n_}'] < 3):.2f}" for n_ in MODELS))
for c in [c for c in R.columns if c.endswith(("_lin", "_logit")) and c != "base_lin"]:
    out = []
    for name, g in (("activated", R[R.act]), ("usual", R[~R.act])):
        m_, s_ = paired(g, "base_lin", c); out.append(f"{name} {m_:+.4f} ({m_ / s_:+.1f} SE)" if s_ > 0 else f"{name} {m_:+.4f}")
    print(f"  base_lin -> {c}: " + " | ".join(out))
m_, s_ = paired(R[R.act], "nn", "base_lin"); print(f"  (reference) NN -> base_lin on activated: {m_:+.4f} ({m_ / s_:+.1f} SE)")
