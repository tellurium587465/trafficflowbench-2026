"""11/05 ledger #246: the network probabilities of the official Task 2 windows, rebuilt from the release (for a newly generated month in the
same file layout; the shipped release assets carry the March / April probabilities already, so run_final's fast path does not need this).
  1. inputs  : every window of task2/<panel>/<split>/window_index.csv (both splits), the per-window features as the training windows
               (window_X for the window's condition, the priors zeroed, arrival + demand; TFB_T2_NOT=1, TFB_T2_ALLFIN=1, history as released)
               -> NN/<IN>/ (X, off, window_id, cond), its raw history R.npy (t2_raw_hist.py) and the rank inputs NN/<IN>_pct (t2_pct.pct_dir)
               checked: bit-identical to the shipped nn_test_afdem on April's D7_I10_E windows
  2. networks: the parts of the named finals (t2_final_spec.FINAL: the onset / ongoing blends of both splits, TFM, X3, U2D, PC), CPU fp32:
               plain family (q24f, pcL on the rank inputs) t2_nn_cpu_infer.infer, raw-history x3af and tfm4 t2_cpu_infer_x3_tfm.infer,
               the 2D U-Net t2_cpu_infer_u2d.infer -> WORK/t2/probs_<tag>.pkl (the names the finals read)
               CPU-made shipped parts (pcL, u2d, x3af) are reproduced bit for bit; GPU-made ones (q24f bf16, tfm4 fp16) to ~1e-2 / 1e-3, which moves
               no Task 2 cell of A17 and 2 cells of B7 (review/g311)
The X1 probabilities (fine-tuned on the month's earlier days) are run_final's t2x1 / t2x1sd stages; LightGBM and the physics onset its t2aux.
Refuses to overwrite WORK/t2/probs_*.pkl of the production work folder unless TFB_INFER_OVERWRITE=1.
usage: python t2_infer_official.py <final> [final ...]      env TFB_INFER_IN (default nn_test_new)"""
import os, sys, time, pickle, subprocess, numpy as np, pandas as pd, warnings
os.environ["TFB_T2_NOT"] = "1"; os.environ["TFB_T2_ALLFIN"] = "1"; warnings.filterwarnings("ignore")
from cachelib import load, T2_PANELS
from common import REL, NN, WORK
from t2_model import PanelInputs
from t2_cache import window_X
from t2_arrival import arrival_features, _dmin
from t2_demand import demand_features, profile
import t2_final_spec as FS

SRC = os.path.dirname(os.path.abspath(__file__)); IN = os.environ.get("TFB_INFER_IN", "nn_test_new"); T0 = time.time()
if os.path.exists(f"{WORK}/t2/probs_q24f_attnglob_s0.pkl") and os.environ.get("TFB_INFER_OVERWRITE") != "1":
    sys.exit(f"{WORK}/t2 already holds network probabilities; set TFB_INFER_OVERWRITE=1 to replace them (or use a fresh TFB_WORK)")


def build_inputs():
    names = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]; XS, OF, WID, CND = [], [0], [], []
    for split in ("validation", "private"):
        for p in T2_PANELS:
            P = load(p, split); PI = PanelInputs(P); prof = profile(load(p, "train")); ti = {s: i for i, s in enumerate(P.times)}
            for r in pd.read_csv(REL / "task2" / p / split / "window_index.csv").itertuples():
                t = ti[r.forecast_origin]; Xw, _ = window_X(PI, t, r.condition); Xw[:, names.index("prior_on")] = 0.0; Xw[:, names.index("prior_og")] = 0.0
                V, Q, tl = PI.window(t); Xa, _ = arrival_features(P, V, Q, tl, PI.x); Xd = demand_features(P, Q, tl, t, P.times[t][:10], prof, _dmin(P, PI.x))
                X = np.column_stack([Xw, Xa, Xd]).astype(np.float32); XS.append(X); OF.append(OF[-1] + len(X)); WID.append(r.window_id); CND.append(r.condition)
            load.cache_clear()
    o = f"{NN}/{IN}/"; os.makedirs(o, exist_ok=True)
    np.save(o + "X.npy", np.vstack(XS)); np.save(o + "off.npy", np.array(OF)); np.save(o + "window_id.npy", np.array(WID)); np.save(o + "cond.npy", np.array(CND))
    print(f"inputs {IN}: {len(WID)} windows, {OF[-1]} rows ({time.time() - T0:.0f}s)", flush=True)
    subprocess.run([sys.executable, "-B", "t2_raw_hist.py", IN], cwd=SRC, check=True)
    import t2_pct
    t2_pct.pct_dir(IN, IN + "_pct"); print(f"raw history and rank inputs ({time.time() - T0:.0f}s)", flush=True)


def tags_of(finals):
    out = set()
    for f in finals:
        env, _ = FS.FINAL[f]; keep = {k: os.environ.get(k) for k in env}; os.environ.update(env)
        try:
            cfg = FS.config()
            for sp in ("validation", "private"):
                for c in ("queue_onset", "queue_ongoing"):
                    out.update(t for t, _ in FS.parts(cfg, sp, c))
            for v in cfg.get("tfm", {}).values():
                out.update(v)
            out.update(cfg.get("x3", [])); out.update(cfg.get("u2d", [])); out.update(cfg.get("pc", []))
        finally:
            for k, v in keep.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    return sorted(out)


if __name__ == "__main__":
    finals = sys.argv[1:]; assert finals, "name at least one final (t2_final_spec.FINAL)"
    build_inputs(); d = f"{NN}/{IN}"; tags = tags_of(finals); print(f"{len(tags)} networks for {finals}", flush=True)
    import t2_nn_cpu_infer as CI, t2_cpu_infer_x3_tfm as XT, t2_cpu_infer_u2d as UD
    for tag in tags:
        t0 = time.time()
        if tag.startswith(("x3", "tfm")):
            pr = XT.infer(tag, d)
        elif tag.startswith("u2d_s"):
            pr = UD.infer(int(tag.split("_s")[-1]), d)
        elif tag.startswith("pcL"):
            pr = CI.infer(tag, [d + "_pct"])
        else:
            pr = CI.infer(tag, [d])
        pickle.dump(pr, open(f"{WORK}/t2/probs_{tag}.pkl", "wb")); print(f"{tag}: {len(pr)} windows {time.time() - t0:.1f}s", flush=True)
    print(f"done ({time.time() - T0:.0f}s)", flush=True)
