"""Task 2 for ONE window, from the release as given (10/22 E6; 10/23 extended to the final candidates). The organisers'
check (discussion 749601): Task 2 is re-run with every row after each window's forecast origin T deleted. Same models,
mixes and post-processing as t2_build_variant.py (the composition = the env options of t2_final_spec.py, the per-condition
choice as the 4th argument), for one window:
  1. cache  : build_cache.build(panel, split) from TFB_REL into TFB_CACHE for the window's panel, and for every panel of
              its split when the physics a or X1's training windows are recomputed (both read the month up to T); the
              train months are linked from the default cache when TFB_CACHE has none (built from the release otherwise)
  2. inputs : the window's NN features exactly as t2_nn_af.py builds nn_test_afdem -> NN/nn_test_entry_<out>; when TTA is
              on for the window, its 8 draws exactly as t2_tta_official_build.py (each draw's noise stream replayed
              through the months before the window's panel, shapes in repro/tta_shapes.json) -> NN/nn_test_entry_<out>_tta<k>
  3. probs  : "saved" = every part's stored probabilities of the window (reproduces the build bit for bit);
              "cpu"   = recomputed here: t2_nn_cpu_infer.py for the saved networks, the no-memory LightGBM and the physics
              onset model on the window's features, the physics a and the onset shift's earlier-day cells from the month's
              completed earlier days; X1 for a
              window on day D >= 2: TFB_ENTRY_X1=run fine-tunes every network here on this release's windows of days
              1 .. D-1 (twin/build_march.py -> NN/nn_data_entry_<out>; t2_nn_finetune.py as t2_x1_official.py),
              TFB_ENTRY_X1=stored (default) takes the stored X1 probabilities (their inputs: t2_trunc_physical.py)
  4. t2_final_spec.window_probs -> [t2_cone.py] -> t2_onset_evidence.py B=0.25 OGB=0.85 -> t2_far_gate.py D=3 CONTIG=1,
     each restricted to this window (TFB_T2_WINDOWS / TFB_T2_TEST)
Writes work/t2/queue_entry_<out>.parquet (the window's 6 x n rows) and work/t2/entry_<out>_parts.pkl (every part's
probabilities, the physics a).
usage: TFB_REL=<release root> TFB_CACHE=<cache dir> [composition env] python t2_window_entry.py <window_id> <out> [cpu|saved]
       [onset source,onset tta,ongoing source,ongoing tta,cone]   (default base,0,base,0,0 = the s70 composition)"""
import os, sys, json, pickle, subprocess, warnings, numpy as np, pandas as pd
os.environ["TFB_T2_NOT"] = "1"; os.environ["TFB_T2_ALLFIN"] = "1"; warnings.filterwarnings("ignore")
from common import REL, NN, ROOT, WORK
import cachelib
from cachelib import load, T2_PANELS
import t2_final_spec as FS

wid, out = sys.argv[1], sys.argv[2]; MODE = sys.argv[3] if len(sys.argv) > 3 else "cpu"
ch = (sys.argv[4] if len(sys.argv) > 4 else "base,0,base,0,0").split(",")
CHOICE, CONE = (ch[0], int(ch[1]), ch[2], int(ch[3])), int(ch[4])
CFG = FS.config(); X1MODE = os.environ.get("TFB_ENTRY_X1", "stored")
SRC = os.path.dirname(os.path.abspath(__file__)); W = WORK + "/t2/"
split = "validation" if "_validation_" in wid else "private"
panel = wid[3:wid.index(f"_{split}_")]; assert panel in T2_PANELS, panel
wi = pd.read_csv(REL / "task2" / panel / split / "window_index.csv"); r = wi[wi.window_id == wid].iloc[0]
DAY = int(str(r.date)[8:10])                                    # 1-based day of month, as t2_x1_official.py groups windows

# which parts the composition reads for this window (a dry run of the blend)
NEED = []
def _rec(pre, tag):
    NEED.append((pre, tag)); return 0.0 if pre == "physon_a" else None if pre == "e3stats" else np.zeros((6, 1))
FS.window_probs(wid, r.condition, split, CHOICE, CFG, _rec)
TTA = any(p.startswith(("afdemtta", "x1tta")) for p, _ in NEED)
ARM = CFG.get("x1arm", ""); X1LO = 1 if ARM in ("sd", "sdd3") else 2      # 11/01 ledger #125: same-day X1 also acts on day 1 (11/03 #189: sdd3 = sd on April's origins every 3 rows)
WHOLE = MODE == "cpu" and (("physon_a", "") in NEED or (X1MODE == "run" and DAY >= X1LO and any(p.startswith("x1") for p, _ in NEED)))

# 1. cache
import build_cache
default = WORK + "/cache"
for p in (T2_PANELS if WHOLE else [panel]):
    tr = cachelib.CACHE / f"{p}_train.npz"
    if not tr.exists():
        if os.path.exists(f"{default}/{p}_train.npz") and os.path.abspath(default) != os.path.abspath(cachelib.CACHE):
            os.link(f"{default}/{p}_train.npz", tr)
        else:
            build_cache.build(p, "train")
    build_cache.build(p, split)
load.cache_clear()

# 2. inputs (keep in sync with t2_nn_af.py / t2_tta_official_build.py)
from t2_model import PanelInputs
from t2_cache import window_X
from t2_arrival import arrival_features, _dmin
from t2_demand import demand_features, profile
import t2_tta_official_build as TB
names = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]
I_PON, I_POG = names.index("prior_on"), names.index("prior_og")
P0 = load(panel, split); ti = {s: i for i, s in enumerate(P0.times)}; prof = profile(load(panel, "train")); t = ti[r.forecast_origin]


def features(P):
    PI = PanelInputs(P)
    Xw, _ = window_X(PI, t, r.condition); Xw[:, I_PON] = 0.0; Xw[:, I_POG] = 0.0
    V, Q, tl = PI.window(t); Xa, _ = arrival_features(P, V, Q, tl, PI.x)
    Xd = demand_features(P, Q, tl, t, P.times[t][:10], prof, _dmin(P, PI.x))
    X = np.column_stack([Xw, Xa, Xd]).astype(np.float32); assert X.shape[1] == len(names)
    return X


def save_inputs(d, X):
    os.makedirs(f"{NN}/{d}", exist_ok=True); np.save(f"{NN}/{d}/X.npy", X); np.save(f"{NN}/{d}/off.npy", np.array([0, len(X)]))
    np.save(f"{NN}/{d}/window_id.npy", np.array([wid])); np.save(f"{NN}/{d}/cond.npy", np.array([r.condition]))


TD = f"nn_test_entry_{out}"; X = features(P0); save_inputs(TD, X)
TDT = TD + "t"; IVT, IQT = names.index("v_T"), names.index("q_T"); PIT = PanelInputs(P0); Xt = X.copy(); Xt[:, IVT] = PIT.mask_v[t] / P0.vf; Xt[:, IQT] = PIT.mask_q[t] / P0.cap; save_inputs(TDT, Xt)   # 11/02 ledger #164: the row at T, for the q24t networks
if TTA:
    SH = json.load(open(ROOT / "repro" / "tta_shapes.json"))
    assert [list(P0.V.shape), list(P0.Q.shape)] == SH[f"{split}/{panel}"], "the month's calendar grid differs from the release's"
    before = [SH[f"{s}/{p}"] for s, p in TB.ORDER[:TB.ORDER.index((split, panel))]]
    for k in range(1, TB.K + 1):
        rng = np.random.default_rng(2000 + k); TB.skip(rng, before); save_inputs(f"{TD}_tta{k}", features(TB.perturb(P0, rng, k)))

# 3. probabilities
_PK = {}
def stored(pre, tag):
    if pre == "e3stats":
        import t2_e3_official as E3
        return E3._stats().get(wid)
    f = W + ("physon_a.pkl" if pre == "physon_a" else f"probs_{pre}{tag}.pkl")
    if f not in _PK:
        _PK[f] = pickle.load(open(f, "rb"))
    return _PK[f][wid]


def cpu_infer(tags, d, pref):
    subprocess.run([sys.executable, "t2_nn_cpu_infer.py", ",".join(tags), d, pref], cwd=SRC, check=True, stdout=subprocess.DEVNULL)
    return {m: pickle.load(open(W + f"probs_{pref}__{m}.pkl", "rb"))[wid] for m in tags}


PARTS = {}
if MODE == "saved":
    PARTS = {k: stored(*k) for k in NEED}
else:
    import t2_phys_onset as PO
    x1_here = DAY >= X1LO and any(p.startswith("x1") for p, _ in NEED)
    base_tags = sorted({m for p, m in NEED if m not in CFG.get("x3", []) + CFG.get("u2d", []) + CFG.get("pc", []) and (p == "" or p.startswith("afdemtta") or (not x1_here and p.startswith("x1")))})
    if base_tags:                                               # day-1 windows: X1 keeps the saved model (t2_x1_official.py)
        pr = {}
        for ts_, d_ in (([m for m in base_tags if not m.startswith("q24t")], TD), ([m for m in base_tags if m.startswith("q24t")], TDT)):
            if ts_:
                pr.update(cpu_infer(ts_, d_, f"entry_{out}"))
        PARTS.update({(pre, m): pr[m] for m in base_tags for pre in ("", f"x1off{ARM}__")})
        for k in (range(1, TB.K + 1) if TTA else []):
            pr = cpu_infer(base_tags, f"{TD}_tta{k}", f"entry_{out}tta{k}")
            PARTS.update({(pre, m): pr[m] for m in base_tags for pre in (f"afdemtta{k}__", f"x1tta{k}__")})
    if CFG.get("x3") and r.condition == "queue_ongoing":        # 11/01 ledger #125: the raw history of the window, as t2_raw_hist.py
        import t2_raw_hist as RH, t2_nn3_infer as N3
        np.save(f"{NN}/{TD}/R.npy", RH.raw(PanelInputs(P0), P0, t).astype(np.float16))
        PARTS.update({("", m): N3.infer(m, f"{NN}/{TD}")[wid] for m in CFG["x3"]})
    if CFG.get("u2d") and r.condition == "queue_ongoing":       # 11/03 ledger #201: the 2D U-Net on the same raw history
        import t2_u2d_apply as U2
        if not CFG.get("x3"):
            import t2_raw_hist as RH
            np.save(f"{NN}/{TD}/R.npy", RH.raw(PanelInputs(P0), P0, t).astype(np.float16))
        PARTS.update({("", m): U2.infer(m, f"{NN}/{TD}")[wid] for m in CFG["u2d"]})
    if CFG.get("pc") and r.condition == "queue_onset":         # 11/03 ledger #201 (#17): the rank-input network on the window's percentile inputs
        import t2_pct
        t2_pct.pct_dir(TD, TD + "_pct"); pr = cpu_infer(CFG["pc"], TD + "_pct", f"entry_{out}pct")
        PARTS.update({("", m): pr[m] for m in CFG["pc"]})
    if x1_here:
        x1_tags = sorted({m for p, m in NEED if p.startswith("x1")})
        if X1MODE == "run":
            import t2_nn_finetune as FT
            MD = f"nn_data_entry_{out}"
            spc = "3" if ARM == "sdd3" and split == "private" else "6"   # 11/03 #189 (A13): t2_x1_sameday.py's X1_POOL_PRIV=nn_data_april_sp3
            subprocess.run([sys.executable, str(ROOT / "twin" / "build_march.py"), spc, split], cwd=SRC, check=True,
                           stdout=subprocess.DEVNULL, env=dict(os.environ, TFB_MONTH_OUT=MD))
            if any(m.startswith("q24t") for m in x1_tags):      # 11/02 ledger #164: the pool with the row at T, from the cut copy
                import t2_trow; t2_trow.fill_dir(MD, MD + "t", split)
            for m in x1_tags:
                MD_ = MD + "t" if m.startswith("q24t") else MD
                if ARM in ("sd", "sdd3"):               # as t2_x1_sameday.py: the same day's windows that end before T's hour starts
                    import torch; torch.set_num_threads(int(os.environ.get("X1_THREADS", "4")))
                    Dd = FT.data(MD_); late = set(np.flatnonzero((Dd["day"] == DAY) & (np.load(f"{NN}/{MD_}/t.npy") + 6 > t // 12 * 12)).tolist())
                    net, mu, sd, cm, _ = FT.finetune(m, MD_, 1, DAY, lr=2e-4, epochs=4, wpow=FT.wpow(m), exclude=late)
                else:
                    net, mu, sd, cm, _ = FT.finetune(m, MD, 1, DAY - 1, lr=2e-4, epochs=4, wpow=FT.wpow(m))
                PARTS[(f"x1off{ARM}__", m)] = FT.predict(net, mu, sd, cm, TDT if m.startswith("q24t") else TD, [wid])[wid]
                for k in (range(1, TB.K + 1) if TTA else []):
                    PARTS[(f"x1tta{k}__", m)] = FT.predict(net, mu, sd, cm, f"{TD}_tta{k}", [wid])[wid]
        else:
            PARTS.update({k: stored(*k) for k in NEED if k[0].startswith("x1")})
    if ("lgbnm", "") in NEED:
        PARTS[("lgbnm", "")] = FS.lgb_ongoing(X)
    if (CFG["phys"], "") in NEED:
        import lightgbm as lgb
        assert CFG["phys"] == "physon", "cpu mode recomputes the physon model only"
        ph = np.zeros((6, len(X)), np.float32)
        ph[5] = lgb.Booster(model_file=W + ("physon_base.txt" if split == "validation" else "physon_m.txt")).predict(PO.feats(X))
        PARTS[(CFG["phys"], "")] = ph
    if ("physon_a", "") in NEED:
        PARTS[("physon_a", "")] = PO.adaptive_a(PO.month_events(split, PO.rarity_table()), t // 288)
    if ("e3stats", "") in NEED:                                 # the onset shift: the earlier days' observed cells (beta = 0
        import t2_e3_official as E3                             # does not read the pseudo-origins' predictions)
        assert E3.CONST[r.condition][1] == 0, "cpu mode recomputes E3 with beta = 0 only"
        st = E3.earlier_days(P0, t); PARTS[("e3stats", "")] = None if st is None else (*st, np.zeros_like(st[0]))
assert set(NEED) <= set(PARTS), set(NEED) - set(PARTS)
PARTS = {k: PARTS[k] for k in dict.fromkeys(NEED)}         # only what the composition reads
pickle.dump(PARTS, open(W + f"entry_{out}_parts.pkl", "wb"))
pb = FS.window_probs(wid, r.condition, split, CHOICE, CFG, lambda pre, tag: PARTS[(pre, tag)])
pickle.dump({wid: pb}, open(W + f"probs_entry_{out}hyb.pkl", "wb"))

# 4. post-processing, restricted to this window
env = dict(os.environ, TFB_T2_WINDOWS=wid, TFB_T2_TEST=f"{NN}/{TD}"); src = f"entry_{out}hyb"
if CONE:
    subprocess.run([sys.executable, "t2_cone.py", src, f"entry_{out}cone"], cwd=SRC, check=True, env=env, stdout=subprocess.DEVNULL); src = f"entry_{out}cone"
subprocess.run([sys.executable, "t2_onset_evidence.py", src, f"entry_{out}on", "0.25", "0.85"], cwd=SRC, check=True, env=env)
subprocess.run([sys.executable, "t2_far_gate.py", W + f"queue_entry_{out}on.parquet", W + f"queue_entry_{out}.parquet", "3", "1"],
               cwd=SRC, check=True, env=env)
FS.ograw(W + f"queue_entry_{out}.parquet", {wid: pb}, {wid: r.condition}, {wid: split}, CFG)   # 10/30 ledger #72 (off unless set)
q = pd.read_parquet(W + f"queue_entry_{out}.parquet")
print(f"{wid} ({r.condition}, day {DAY}, T = {r.forecast_origin}, {MODE} probabilities, {len(NEED)} parts, X1 {X1MODE}): "
      f"{len(q)} rows, positives {int(q.queue_pred.sum())}")
