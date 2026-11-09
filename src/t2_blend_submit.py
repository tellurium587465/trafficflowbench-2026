"""Blend validation/private probabilities (probs_<tag>.pkl, same spec syntax as the OOF evaluator: "a:2+b+c")
and assemble a full submission with the given Task 1 state and Task 4 answer.

usage: python t2_blend_submit.py <spec> <submission_name> <state_parquet> <odme_parquet>
"""
import sys, pickle, numpy as np, pandas as pd
from cachelib import load, T2_PANELS
from common import REL
from t2_windows import F
from t2_model import best_set
from submit import assemble

W = "C:/Users/user/tfb/work/"
if len(sys.argv) != 5:                                          # fail closed: an old default T1 state used to slip in silently
    sys.exit("usage: python t2_blend_submit.py <spec> <submission_name> <state_parquet> <odme_parquet>")
spec, name, state, odme = sys.argv[1:5]
def _parts(sp):
    out = []
    for t, w in (x.split(":") if ":" in x else (x, "1") for x in sp.split("+")):
        pr = pickle.load(open(W + f"t2/probs_{t}.pkl", "rb"))
        bad = [k for k, v in pr.items() if not np.isfinite(v).all()]
        assert not bad, f"probs_{t} has non-finite values in {len(bad)} windows (diverged training?)"   # fail closed (10/10: q26f_r13_s0)
        out.append((float(w), pr))
    return out


# "onset_spec|ongoing_spec" gives each condition its own weights (t2_blend_opt chooses them per condition)
SPECS = dict(zip(("queue_onset", "queue_ongoing"), spec.split("|"))) if "|" in spec else {"queue_onset": spec, "queue_ongoing": spec}
PARTS = {c: _parts(s) for c, s in SPECS.items()}
out, blended = [], {}
for split in ("validation", "private"):
    for p in T2_PANELS:
        P = load(p, split); ti = {s: i for i, s in enumerate(P.times)}
        for r in pd.read_csv(REL / "task2" / p / split / "window_index.csv").itertuples():
            t = ti[r.forecast_origin]
            parts = PARTS[r.condition]; pb = sum(w * d[r.window_id] for w, d in parts) / sum(w for w, _ in parts); blended[r.window_id] = pb
            sel = best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape)
            stamps = [P.times[t + s + 1] for s in range(F)]
            out.append(pd.DataFrame({"window_id": r.window_id, "timestamp": np.repeat(stamps, P.n),
                                     "link_id": np.tile(P.links, F), "queue_pred": sel.ravel().astype(int)}))
q = pd.concat(out, ignore_index=True)
q.to_parquet(W + f"t2/queue_{name}.parquet", index=False); pickle.dump(blended, open(W + f"t2/probs_{name}.pkl", "wb"))
print("windows", len(blended), "positives", int(q.queue_pred.sum()))
assemble(pd.read_parquet(state), q, pd.read_parquet(odme), name)
