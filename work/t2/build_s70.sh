#!/bin/bash
# s70 = ongoing from the q43f (R24) + q46f (R48) blend, 4 seeds each (s60 otherwise); with the ONGOING probabilities from the sim-augmented blend (q37f, real + randomized twin worlds; local
# ongoing -0.001 = no in-distribution cost) and the ONSET probabilities unchanged (s46 blend; the sim-augmented onset
# loses -0.024 locally, F57). Then s60's post-processing (B 0.25, OGB 0.85, far gate + contiguous growth).
set -euo pipefail; export PATH="/usr/bin:$PATH"; cd /c/Users/user/tfb/src; W=C:/Users/user/tfb/work/
SPEC=""
for s in 0 1 2 3; do SPEC="$SPEC+q43f_attnglob_s$s:2+q43f_attnwp05_s$s:2+q43f_attn1_s$s:2+q43f_drop1_s$s+q43f_r13_s$s+q46f_attnglob_s$s:2+q46f_attnwp05_s$s:2+q46f_attn1_s$s:2+q46f_drop1_s$s+q46f_r13_s$s"; done; SPEC=${SPEC#+}
N=s70raw_t1Adapt2GapMLClipFdBnd_t2simR24R48NN_t4l2
py t2_blend_submit.py "$SPEC" $N ${W}t1/state_f3ens3c_ad2_gapml_clip_fd_bnd.parquet ${W}t4/odme_l2.parquet | tail -1
py - <<'PY'
import pickle, pandas as pd
from common import REL
from cachelib import T2_PANELS
W = "C:/Users/user/tfb/work/t2/"
a = pickle.load(open(W + "probs_s46raw_t1Adapt2GapMLClipFdBnd_t2nomemNNonly_t4l2.pkl", "rb"))
b = pickle.load(open(W + "probs_s70raw_t1Adapt2GapMLClipFdBnd_t2simR24R48NN_t4l2.pkl", "rb"))
cond = {}
for sp in ("validation", "private"):
    for p in T2_PANELS:
        w = pd.read_csv(REL / "task2" / p / sp / "window_index.csv"); cond.update(dict(zip(w.window_id, w.condition)))
assert set(a) == set(b) == set(cond)
h = {k: (b[k] if cond[k] == "queue_ongoing" else a[k]) for k in a}
pickle.dump(h, open(W + "probs_s70raw_t1Adapt2GapMLClipFdBnd_t2OnsetS46OngoingSimR24R48_t4l2.pkl", "wb"))
print("hybrid: ongoing from q43f+q46f", sum(cond[k] == "queue_ongoing" for k in h), "windows, onset from s46", sum(cond[k] == "queue_onset" for k in h))
PY
py t2_onset_evidence.py s70raw_t1Adapt2GapMLClipFdBnd_t2OnsetS46OngoingSimR24R48_t4l2 s70on_t2HybridR24R48OnB025Og085 0.25 0.85 | tail -1
py t2_far_gate.py ${W}t2/queue_s70on_t2HybridR24R48OnB025Og085.parquet ${W}t2/queue_s70_t2HybridR24R48OnB025Og085GateContig.parquet 3 1 | tail -1
py -c "
import pandas as pd
from submit import assemble
W='C:/Users/user/tfb/work/'
a=pd.read_parquet(W+'t2/queue_s70_t2HybridR24R48OnB025Og085GateContig.parquet'); b=pd.read_parquet(W+'t2/queue_s60_t2OnB025Og085GateContig.parquet')
d=a.queue_pred!=b.queue_pred; w=a.window_id[d].unique(); print('s70 vs s60: cells', int(d.sum()), 'windows', len(w), 'validation', sum('validation' in x for x in w))
from common import REL; from cachelib import T2_PANELS; cond={}
for sp in ('validation','private'):
    for p in T2_PANELS:
        x=pd.read_csv(REL/'task2'/p/sp/'window_index.csv'); cond.update(dict(zip(x.window_id,x.condition)))
c=a.window_id.map(cond); assert not d[c=='queue_onset'].any(), 'onset windows changed'; print('onset windows identical to s60: True')
print(assemble(pd.read_parquet(W+'t1/state_f3ens3c_ad2_gapml_clip_fd_bnd.parquet'), a, pd.read_parquet(W+'t4/odme_l2.parquet'), 's70_t1Adapt2GapMLClipFdBnd_t2OnsetS46OngoingSimR24R48_t4l2'))"
