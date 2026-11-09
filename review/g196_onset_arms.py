"""G196 (11/01, ledger #159): the onset arms of the LB pairs on March's official windows, from finA8's blended probabilities (probs_finA8cone.pkl), the evidence rule as t2_onset_evidence.py; only March's onset rows of finA8 change.
  b35 : B 0.35 in place of 0.25
  mc  : #114 the exact expected IoU (G152's mc_set), e = 1 (the official windows score every cell)
  t12 : temperature 1.2 after the rule
  pct : ledger #172b, 0.8 blend + 0.2 the rank input networks pct_attnglob s0, s1, on the blended probabilities
usage: py -B -X utf8 g196_onset_arms.py <arm>   (TFB_WORK, TFB_REL as the branch)"""
import os, sys, pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.dont_write_bytecode = True; sys.path.insert(0, r"C:/Users/user/tfb_a8/src")
from common import WORK, REL
from cachelib import load, T2_PANELS, asof
from submit import assemble
from t2_model import best_set
import t2_final_spec as FS
ARM = sys.argv[1]; W = WORK + "/"; probs = pickle.load(open(W + "t2/probs_finA8cone.pkl", "rb")); A = pd.read_parquet(W + "t2/queue_finA8.parquet"); N = 2000; rng = np.random.default_rng(114); new = {}; nch = 0; d0 = d1 = 0


def mc_set(p):
    n = p.size // 6; p5 = p[5 * n:]; order = np.argsort(-p5)[:min(n, 80)]; Y = rng.random((N, n)) < p5; keep = Y.any(axis=1)
    if keep.sum() < 50:
        return best_set(p, np.ones(p.size, bool))
    Y = Y[keep]; ny = Y.sum(axis=1); I = np.cumsum(Y[:, order], axis=1); U = np.arange(1, len(order) + 1)[None, :] + ny[:, None] - I; k = int(np.argmax((I / np.maximum(U, 1)).mean(axis=0))) + 1; sel = np.zeros(p.size, bool); sel[5 * n + order[:k]] = True; return sel


Bv = 0.35 if ARM == "b35" else 0.25
PC = [pickle.load(open(W + f"t2/probs_pct_attnglob_s{i}.pkl", "rb")) for i in (0, 1)] if ARM == "pct" else None
for p in T2_PANELS:
    P = load(p, "validation"); ti = {s: i for i, s in enumerate(P.times)}
    for r in pd.read_csv(REL / "task2" / p / "validation" / "window_index.csv").itertuples():
        if r.condition != "queue_onset":
            continue
        t = ti[r.forecast_origin]; pb = probs[r.window_id].copy(); pb = 0.8 * pb + 0.1 * (PC[0][r.window_id] + PC[1][r.window_id]) if PC else pb; Pt = asof(P, t); q = (Pt.V <= Pt.vcut[None, :]) & np.isfinite(Pt.V); seen = np.zeros(P.n, bool); nd = 0
        for back in (1, 2, 3):
            dd = t - back * 288
            if dd - 6 >= 0:
                seen |= q[dd - 6:dd + 12].any(axis=0); nd += 1
        if nd:
            ev = seen.copy()
            for o in (1, 2):
                ev[o:] |= seen[:-o]; ev[:-o] |= seen[o:]
            pb[-1] = pb[-1] * np.where(ev, 1.0, Bv)
        if ARM == "t12":
            pb = FS.temper(pb, 1.2)
        sel = mc_set(pb.ravel()) if ARM == "mc" else best_set(pb.ravel(), np.ones(pb.size, bool)); sel = sel.reshape(pb.shape); stamps = [P.times[t + s + 1] for s in range(6)]
        new[r.window_id] = pd.DataFrame({"window_id": r.window_id, "timestamp": np.repeat(stamps, P.n), "link_id": np.tile(P.links, 6), "queue_pred": sel.ravel().astype(int)})
C = A.copy(); idx = A.reset_index().set_index(["window_id", "timestamp", "link_id"])["index"]
for w, f in new.items():
    ii = idx.loc[list(zip(f.window_id, f.timestamp, f.link_id))].to_numpy(); old = A.queue_pred.to_numpy()[ii]; v = f.queue_pred.to_numpy(); C.loc[ii, "queue_pred"] = v; nch += int((old != v).any()); d0 += int(old.sum()); d1 += int(v.sum())
print(ARM, "March onset windows", len(new), "changed", nch, "positives", d0, "->", d1, flush=True); name = f"finA8{ARM}"; C.to_parquet(W + f"t2/queue_{name}.parquet", index=False); print(assemble(pd.read_parquet(W + "t1/state_f3ens3c_ad2_gapml_clip_fd_bnd.parquet"), C, pd.read_parquet(W + "t4/odme_l2.parquet"), name + "_final"))
