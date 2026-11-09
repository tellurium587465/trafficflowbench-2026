"""G293 (11/05, ledger #237 stage 0 (a)): split cells of B7's April ongoing windows. B7's April ongoing = the raw expected-IoU set of its
mixed probabilities (TFB_T2_OGRAW_SPLITS=private, temperature 1: t2_final_spec.ograw on work/t2/probs_repro_finB7traw.pkl); identity: best_set
of those probabilities equals B7's submitted queue on every April ongoing window. Split cells: 0.35 <= p_B7 <= 0.65 on the scored horizon
(all rows). Counted: windows with >= 3 split cells where A17's and B7's decisions agree (A17 from its built queue). Writes work/t2/g293_april.pkl.
usage: py -B -X utf8 g293_b7f_split_cells.py"""
import sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb_a8/src")
from cachelib import T2_PANELS
from common import REL
from t2_model import best_set
W = "C:/Users/user/tfb/work/t2/"
PR = pickle.load(open(W + "probs_repro_finB7traw.pkl", "rb")); QB = pd.read_parquet(W + "queue_repro_finB7t.parquet"); QA = pd.read_parquet(W + "queue_repro_finA17t.parquet")
IB = QB.groupby("window_id", sort=False).indices; IA = QA.groupby("window_id", sort=False).indices; out = {}; bad = 0
for p in T2_PANELS:
    for r in pd.read_csv(REL / "task2" / p / "private" / "window_index.csv").itertuples():
        if r.condition != "queue_ongoing":
            continue
        pb = PR[r.window_id]; sel = best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape)
        b = QB.queue_pred.to_numpy()[IB[r.window_id]].reshape(pb.shape).astype(bool); a = QA.queue_pred.to_numpy()[IA[r.window_id]].reshape(pb.shape).astype(bool)
        bad += int(not np.array_equal(sel, b)); sp = (pb >= 0.35) & (pb <= 0.65); ag = sp & (a == b)
        out[r.window_id] = dict(p=pb, a=a, b=b, split=sp, agree=ag, n_split=int(sp.sum()), n_agree=int(ag.sum()))
pickle.dump(out, open(W + "g293_april.pkl", "wb")); ns = np.array([v["n_split"] for v in out.values()]); na = np.array([v["n_agree"] for v in out.values()])
print(f"April ongoing windows {len(out)} | identity best_set against B7's queue: {len(out) - bad} / {len(out)} | split cells per window: median {np.median(ns):.0f}, max {ns.max()} | "
      f"windows with >= 3 split cells on which A17 = B7: {int((na >= 3).sum())} (cells {int(na[na >= 3].sum())}) | A17 vs B7 cells differing overall {sum(int((v['a'] != v['b']).sum()) for v in out.values())}", flush=True)
