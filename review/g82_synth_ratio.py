"""G82 (INSTRUMENT_V3_SPEC §2.1 revised, 10/23): the history-coverage ratio r of the MONTH-SHIFT methods, measured where the
month shift exists: the synthetic C worlds (a different configuration from the train months, a full month with earlier
days). The 6570 C24 onset windows with a T+30 queue (G73) rebuilt in two history conditions from the world panels
(twinC<k>, P.blank = the world's Task 1 mask, P.regime = the day's regime):
  off : official-type history (T-12..T-1 unmasked)     m3 : M3-type (the day's mask also on T-12..T-1; TFB_T2_MASKHIST)
Same feature code as the real pipeline (TFB_T2_NOT=1, TFB_T2_ALLFIN=1; demand profile from the REAL train months for both
conditions). Predictions: the production onset blend (s46 spec, CPU), the physics onset model physon_base (real train
months). Scored at T+30 (expected-IoU set, eligible cells), production order: physics mixed (a = 0.3) BEFORE the
evidence rule (x 0.25 on links unseen at the slot on the k = min(3, day - 1) earlier days of the world's month, from
the masked layer = identical in both conditions).
Effects vs NN, per condition; r = effect(off) / effect(m3); by the day's regime for m3; activated (all true links rare in
the real months, G73) vs usual.
usage: python g82_synth_ratio.py build | score"""
import os, sys, pickle, subprocess, json, time
os.environ.update(TFB_T2_NOT="1", TFB_T2_ALLFIN="1")
sys.path.insert(0, r"C:/Users/user/tfb/src")
step = sys.argv[1]
import numpy as np, pandas as pd
from cachelib import load
from common import NN
import t2_model
from t2_model import PanelInputs, best_set
from t2_cache import window_X
from t2_arrival import arrival_features, _dmin
from t2_demand import demand_features, profile

W = "C:/Users/user/tfb/work/t2/"; SRC = "C:/Users/user/tfb/src"; S = f"{NN}/nn_data_twinC24"
ks = pickle.load(open(W + "g73_c24on_ks.pkl", "rb"))
panel = np.load(f"{S}/panel.npy", allow_pickle=True); tt = np.load(f"{S}/t.npy"); date = np.load(f"{S}/date.npy", allow_pickle=True)
cond = np.load(f"{S}/cond.npy", allow_pickle=True); off = np.load(f"{S}/off.npy")
names = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]
I_PON, I_POG = names.index("prior_on"), names.index("prior_og")
WORLD = {}                                                     # (panel, month) -> world index
def world_of(p, mo):
    if (p, mo) not in WORLD:
        for k in range(24):
            WORLD[(p, load(p, f"twinC{k}").times[0][:7])] = k
    return WORLD[(p, mo)]

if step == "build":
    t0 = time.time(); PIs, PROF = {}, {}; rows = {"off": [], "m3": []}; meta = []
    for i in ks:
        p, t, mo = str(panel[i]), int(tt[i]), str(date[i])[:7]; k = world_of(p, mo)
        if (p, k) not in PIs:
            P = load(p, f"twinC{k}"); P.split = "train"; PIs[(p, k)] = PanelInputs(P)
        if p not in PROF:
            PR = load(p, "train"); PROF[p] = (profile(PR, None), _dmin(PR, PanelInputs(PR).x))
        PI = PIs[(p, k)]; P = PI.P
        for c in ("off", "m3"):
            t2_model.MASKHIST = (c == "m3")
            Xw, _ = window_X(PI, t, "queue_onset"); Xw[:, I_PON] = 0.0; Xw[:, I_POG] = 0.0
            V, Q, tl = PI.window(t); Xa, _ = arrival_features(P, V, Q, tl, PI.x)
            Xd = demand_features(P, Q, tl, t, P.times[t][:10], PROF[p][0], _dmin(P, PI.x))
            rows[c].append(np.column_stack([Xw, Xa, Xd]).astype(np.float32))
        meta.append((int(i), p, k, t, int(P.regime[t])))
    t2_model.MASKHIST = False
    for c in ("off", "m3"):
        D = f"{NN}/nn_test_c24on_{c}"; os.makedirs(D, exist_ok=True)
        np.save(f"{D}/X.npy", np.vstack(rows[c])); np.save(f"{D}/off.npy", np.concatenate([[0], np.cumsum([len(r) for r in rows[c]])]))
        np.save(f"{D}/window_id.npy", np.array([f"C24_{i}" for i in ks])); np.save(f"{D}/cond.npy", np.array(["queue_onset"] * len(ks)))
    pickle.dump(meta, open(W + "g82_meta.pkl", "wb"))
    XS = np.load(f"{S}/X.npy", mmap_mode="r"); cols = [j for j, n in enumerate(names) if n not in ("prior_on", "prior_og")]
    ref = np.vstack([np.asarray(XS[off[i]:off[i + 1]]) for i in ks]); X = np.vstack(rows["off"])
    d = np.where(np.isnan(ref[:, cols]) & np.isnan(X[:, cols]), 0, np.abs(ref[:, cols] - X[:, cols]))
    print(f"built {len(ks)} windows x 2 conditions ({time.time() - t0:.0f}s); official-type vs the stored C24 rows: cells differing {(d > 1e-5).mean():.4f}"
          f" (by column, top: {sorted(((float((d[:, j] > 1e-5).mean()), names[cols[j]]) for j in range(len(cols))), reverse=True)[:4]})")
    spec = json.load(open(f"{SRC}/t2_specs.json"))["s37"]["spec"]; tags = ",".join(x.split(":")[0] for x in spec.split("+"))
    for c in ("off", "m3"):
        subprocess.run([sys.executable, "t2_nn_cpu_infer.py", tags, f"nn_test_c24on_{c}", f"c24on{c}"], cwd=SRC, check=True)
    sys.exit()

# ---- score ----
import lightgbm as lgb
from scipy.stats import rankdata
meta = pickle.load(open(W + "g82_meta.pkl", "rb"))
spec = json.load(open(f"{SRC}/t2_specs.json"))["s37"]["spec"]; parts = [(x.split(":")[0], float(x.split(":")[1]) if ":" in x else 1.0) for x in spec.split("+")]
tw = sum(w for _, w in parts)
PR = {c: {t: pickle.load(open(W + f"probs_c24on{c}__{t}.pkl", "rb")) for t, _ in parts} for c in ("off", "m3")}
mdl = lgb.Booster(model_file=W + "physon_base.txt")
import t2_phys_onset as PO
G73 = pd.read_csv(W + "g73_synth_instrument.csv").set_index("k")
Y = np.load(f"{S}/Y.npy", mmap_mode="r"); E = np.load(f"{S}/EL.npy", mmap_mode="r")
QW = {}
def qmask(p, k):
    if (p, k) not in QW:
        P = load(p, f"twinC{k}"); V = np.where(P.blank, np.nan, P.V); QW[(p, k)] = np.isfinite(V) & (V <= P.vcut[None, :]) & P.E
    return QW[(p, k)]
def dil(s, r):
    o = s.copy()
    for d in range(1, r + 1):
        o[d:] |= s[:-d]; o[:-d] |= s[d:]
    return o
DX = {c: (np.load(f"{NN}/nn_test_c24on_{c}/X.npy", mmap_mode="r"), np.load(f"{NN}/nn_test_c24on_{c}/off.npy")) for c in ("off", "m3")}
rows = []
for j, (i, p, k, t, regime) in enumerate(meta):
    y = np.asarray(Y[off[i]:off[i + 1], 5]).astype(bool); e = np.asarray(E[off[i]:off[i + 1], 5]).astype(bool)
    iou = lambda p5: (lambda s_: (lambda u: 1.0 if u == 0 else float((s_ & y & e).sum() / u))(((s_ | y) & e).sum()))(best_set(p5, e))
    q = qmask(p, k); dom = t // 288 + 1; sn = np.zeros(len(y), bool); nd = 0
    for back in range(1, min(3, dom - 1) + 1):
        dd = t - back * 288
        if dd - 6 >= 0:
            sn |= q[dd - 6:dd + 12].any(axis=0); nd += 1
    evm = np.where(dil(sn, 2), 1.0, 0.25) if nd else np.ones(len(y))
    r = dict(i=i, regime=regime, act=bool(G73.loc[i, "act"]) if i in G73.index else None)
    for c in ("off", "m3"):
        p5 = (sum(w * PR[c][t_][f"C24_{i}"] for t_, w in parts) / tw)[5]
        a_, b_ = DX[c][1][j], DX[c][1][j + 1]; ph = mdl.predict(PO.feats(np.asarray(DX[c][0][a_:b_])))
        mix = 0.7 * p5 + 0.3 * ph
        r[f"nn_{c}"] = iou(p5); r[f"phys_{c}"] = iou(mix); r[f"ev_{c}"] = iou(p5 * evm); r[f"physev_{c}"] = iou(mix * evm)
    rows.append(r)
R = pd.DataFrame(rows); R.to_csv(W + "g82_synth_ratio.csv", index=False)
rng = np.random.default_rng(0)
def se(x):
    x = np.asarray(x, float); return float(np.std([x[rng.integers(0, len(x), len(x))].mean() for _ in range(1000)]))
print(f"C24 onset windows {len(R)}: NN official-type {R.nn_off.mean():.4f} / M3-type {R.nn_m3.mean():.4f} (diff {(R.nn_m3 - R.nn_off).mean():+.4f} +- {se(R.nn_m3 - R.nn_off):.4f})")
for name, g in (("all", R), ("activated", R[R.act == True]), ("usual", R[R.act == False])):
    print(f"== {name} (n {len(g)})")
    for m in ("phys", "ev", "physev"):
        a_, b_ = g[f"{m}_off"] - g.nn_off, g[f"{m}_m3"] - g.nn_m3
        print(f"  {m}: official-type {a_.mean():+.4f} +- {se(a_):.4f} | M3-type {b_.mean():+.4f} +- {se(b_):.4f} | r = {a_.mean() / b_.mean() if abs(b_.mean()) > 1e-9 else float('nan'):.2f}"
              + " | M3-type by regime " + ", ".join(f"{rg}: {(g[g.regime == rg][f'{m}_m3'] - g[g.regime == rg].nn_m3).mean():+.4f}" for rg in (0, 1, 2)))
