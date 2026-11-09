"""G94 (10/23, strategy §21-2 (iv) / §17: the synthetic newly-activated onsets as TRAINING data for the location-agnostic
rank model, not the NN). The physics onset model (physon_base: binary LightGBM on within-window percentiles, train-month
onset windows with a T+30 queue) is retrained with the C24 synthetic onset windows (T+30 queue) added:
  syn    : + every synthetic onset window of the training worlds
  synact : + only the ACTIVATED ones (every true T+30 link rare in the real train months, G73)
World split: the 24 C24 worlds (synthetic calendar months) in two halves (alternate in date order); a model trained with
one half is scored on the other half's windows (M5, cross-fit, so no synthetic window is scored by a model that saw it).
M3 v3 dev (real March, the judge): models trained with all 24 worlds, mixed as production (lin, month-adaptive a, evidence
rule, decoder, k expectation), onset base = q24f 8 seeds; paired vs base (physon_base).
M5: NN = the s46 blend on the synthetic windows, p = 0.7 p_NN + 0.3 p_ph (as G73), decoder on eligible cells; paired vs
base, SE clustered by panel x world; activated / usual; ranks of the best true link.
G94b (mode "family"): the split unit is the corridor family instead of the world (4 folds): a synthetic window is scored
by models that never saw ITS corridors -> whether the M5 gain is transferable physics or the simulator's site memory.
usage: python g94_phys_synth_train.py [family]"""
import sys, os, json, pickle
os.environ.setdefault("TFB_M3VER", "v3"); os.environ["TFB_T2CACHE"] = "2_noT_af"
sys.path.insert(0, r"C:/Users/user/tfb/src"); argv0 = sys.argv[0]; MODE = sys.argv[1] if len(sys.argv) > 1 else ""
import numpy as np, pandas as pd, lightgbm as lgb
import t2_phys_onset as PO
sys.argv = [argv0, "0"]
import t2_m3_gate as G, t2_m3v2_gate as V, m3v2
from common import NN
from t2_model import best_set

W = G.W; F_ = PO.rarity_table()
Xr, yr = PO.rows("nn_data_afdem")
S = f"{NN}/nn_data_twinC24"; ks = pickle.load(open(W + "g73_c24on_ks.pkl", "rb"))       # synthetic onset windows with a T+30 queue
sX = np.load(f"{S}/X.npy", mmap_mode="r"); sY = np.load(f"{S}/Y.npy", mmap_mode="r"); sE = np.load(f"{S}/EL.npy", mmap_mode="r")
soff = np.load(f"{S}/off.npy"); span = np.load(f"{S}/panel.npy", allow_pickle=True); stt = np.load(f"{S}/t.npy"); sdate = np.load(f"{S}/date.npy", allow_pickle=True)
world = {k: str(sdate[k])[:7] for k in ks}; WL = sorted(set(world.values())); HALF = {w: i % 2 for i, w in enumerate(WL)}
FAMSPLIT = MODE == "family"   # G94b: by corridor family
fam = lambda pnl: "_".join(str(pnl).split("_")[:2])
if FAMSPLIT:                                                    # the split unit = corridor family (D7_I10, D7_I210, D7_I405, D12_I5)
    FL = sorted({fam(span[k]) for k in ks}); HALF = {k: FL.index(fam(span[k])) for k in ks}
    world_of = lambda k: k
else:
    world_of = lambda k: world[k]
SF, SYv, ACT = {}, {}, {}
for k in ks:
    a, b = soff[k], soff[k + 1]; y = np.asarray(sY[a:b, 5]).astype(bool); e = np.asarray(sE[a:b, 5]).astype(bool)
    key = (str(span[k]), pd.Timestamp(str(sdate[k])[:10]).dayofweek >= 5, (int(stt[k]) % 288) // 24)
    f = F_.get(key, np.zeros(len(y))); tl = np.flatnonzero(y & e); ACT[k] = bool((f[tl] <= 0.1).all())
    SF[k] = PO.feats(np.asarray(sX[a:b])); SYv[k] = (y, e)
print(f"synthetic onset windows {len(ks)} (activated {sum(ACT.values())}), worlds {len(WL)}", flush=True)


def train(extra):
    X = Xr + [SF[k][SYv[k][1]] for k in extra]; y = yr + [SYv[k][0][SYv[k][1]].astype(int) for k in extra]
    return lgb.train(PO.PRM, lgb.Dataset(np.vstack(X), np.concatenate(y)), 400)


base = lgb.Booster(model_file=W + "physon_base.txt")
HM = {}                                                         # (variant, half) -> model trained with the OTHER half's worlds / families
for v in ("syn", "synact"):
    for h in sorted(set(HALF.values())):
        ext = [k for k in ks if HALF[world_of(k)] != h and (v == "syn" or ACT[k])]
        HM[(v, h)] = train(ext); print(f"{v} half {h}: + {len(ext)} synthetic windows", flush=True)
if not FAMSPLIT:
    FULL = {v: train([k for k in ks if v == "syn" or ACT[k]]) for v in ("syn", "synact")}
    for v, m_ in FULL.items():
        m_.save_model(W + f"g94_phys_{v}.txt")

# M5 (cross-fit)
spec = json.load(open(r"C:/Users/user/tfb/src/t2_specs.json"))["s37"]["spec"]
sparts = [(x.split(":")[0], float(x.split(":")[1]) if ":" in x else 1.0) for x in spec.split("+")]
SPR = {t: pickle.load(open(W + f"probs_c24on__{t}.pkl", "rb")) for t, _ in sparts}; rows = []
for k in ks:
    y, e = SYv[k]; tl = np.flatnonzero(y & e); pn = sum(wt * SPR[t][f"C24_{k}"][5] for t, wt in sparts) / sum(wt for _, wt in sparts)
    r = dict(k=k, panel=str(span[k]), month=world[k], act=ACT[k])
    for v, mdl in (("base", base), ("syn", HM[("syn", HALF[world_of(k)])]), ("synact", HM[("synact", HALF[world_of(k)])])):
        ph = mdl.predict(SF[k]); o = np.argsort(-np.where(e, ph, -1.0)); rr = np.empty(len(o), int); rr[o] = np.arange(len(o)); r[f"rk_{v}"] = int(rr[tl].min())
        s_ = best_set(0.7 * pn + 0.3 * ph, e); u = ((s_ | y) & e).sum(); r[v] = 1.0 if u == 0 else float((s_ & y & e).sum() / u)
    rows.append(r)
R = pd.DataFrame(rows); R.to_csv(W + ("g94b_m5_family.csv" if FAMSPLIT else "g94_m5.csv"), index=False)


def paired(g, a_, b_):
    d = g[b_] - g[a_]; cl = g.panel + "|" + g.month; cm = d.groupby(cl).sum(); n = len(d); nc = cl.value_counts()
    se = np.sqrt(len(cm) / max(len(cm) - 1, 1) * ((cm - cm.index.map(nc) * d.mean()) ** 2).sum()) / n
    return d.mean(), se


print(f"M5 (cross-fit by {'corridor family' if FAMSPLIT else 'world halves'}): {len(R)} windows, activated {int(R.act.sum())}")
for name, g in (("activated", R[R.act]), ("usual", R[~R.act])):
    print(f"  {name}: top-1/3 " + ", ".join(f"{v} {np.mean(g[f'rk_{v}'] == 0):.2f}/{np.mean(g[f'rk_{v}'] < 3):.2f}" for v in ("base", "syn", "synact"))
          + " | " + " | ".join(f"base -> {v} {paired(g, 'base', v)[0]:+.4f} ({paired(g, 'base', v)[0] / paired(g, 'base', v)[1]:+.1f} SE)" for v in ("syn", "synact")))

if FAMSPLIT:
    for f_ in sorted(set(HALF.values())):
        g = R[(R.act) & (R.k.map(HALF) == f_)]
        print(f"  family {f_} activated n {len(g)}: top-1 base {np.mean(g.rk_base == 0):.2f} synact {np.mean(g.rk_synact == 0):.2f} | base -> synact {(g.synact - g.base).mean():+.4f}")
    sys.exit()
# M3 v3 dev (real)
CF5 = (("attnglob", 2.0), ("attnwp05", 2.0), ("attn1", 2.0), ("drop1", 1.0), ("r13", 1.0))
parts = [(f"q24f_{c}_s{s}", w) for s in range(8) for c, w in CF5]
PR = {t: pickle.load(open(W + f"probs_{G.PRE}__{t}.pkl", "rb")) for t, _ in parts}
XM = np.load(f"{NN}/nn_test_m3{G.VER}/X.npy", mmap_mode="r"); OFFM = np.load(f"{NN}/nn_test_m3{G.VER}/off.npy")
ev = PO.month_events("validation", F_)
WON = [w for w in G.WID if G.META[w]["cond"] == "queue_onset" and not m3v2.is_holdout(G.META[w]["day"])]
cols = {"nn": {}, "base": {}, "syn": {}, "synact": {}}
for w in WON:
    m = G.META[w]; i = G.WID.index(w); Fw = PO.feats(np.asarray(XM[OFFM[i]:OFFM[i + 1]]))
    p = sum(wt * PR[t][w] for t, wt in parts) / sum(wt for _, wt in parts); a = PO.adaptive_a(ev, m["day"]); cols["nn"][w] = V.score(p, m)
    for v, mdl in (("base", base), ("syn", FULL["syn"]), ("synact", FULL["synact"])):
        q = p.copy(); q[5] = (1 - a) * p[5] + a * mdl.predict(Fw); cols[v][w] = V.score(q, m)
D = V.frame(cols).dropna(subset=["nn"])
G64 = pd.read_csv(W + "g64_onset_rare_diag.csv").set_index("w"); D["rare"] = D.w.map(G64.rare_true)
D.to_csv(W + f"g94_m3{G.VER}.csv", index=False)
for a_, b_ in (("nn", "base"), ("base", "syn"), ("base", "synact")):
    d = D[b_] - D[a_]
    print(m3v2.fmt(f"M3 {G.VER} dev onset {a_} -> {b_}", m3v2.summary(D, a_, b_, "private")), f"| rare {d[D.rare == True].mean():+.4f} common {d[D.rare == False].mean():+.4f}")
