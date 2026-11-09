#!/bin/bash
# Reproducibility check of the T2 post-model stages: rebuild s60 and s63 from the saved FINAL model predictions
# (probs_<tag>.pkl) and compare the T2 queue tables with the ones that were submitted (exact equality).
set -euo pipefail; export PATH="/usr/bin:$PATH"; cd /c/Users/user/tfb/src; W=C:/Users/user/tfb/work/
T1=${W}t1/state_f3ens3c_ad2_gapml_clip_fd_bnd.parquet; T4=${W}t4/odme_l2.parquet
SPEC46=$(py -c "import json; print(json.load(open('t2_specs.json'))['s37']['spec'])")
py t2_blend_submit.py "$SPEC46" repro_s46raw $T1 $T4 | tail -1
py t2_onset_evidence.py repro_s46raw repro_s60on 0.25 0.85 | tail -1
py t2_far_gate.py ${W}t2/queue_repro_s60on.parquet ${W}t2/queue_repro_s60.parquet 3 1 | tail -1
SPEC37=""; for s in 0 1; do SPEC37="$SPEC37+q37f_attnglob_s$s:2+q37f_attnwp05_s$s:2+q37f_attn1_s$s:2+q37f_drop1_s$s+q37f_r13_s$s"; done; SPEC37=${SPEC37#+}
py t2_blend_submit.py "$SPEC37" repro_s62raw $T1 $T4 | tail -1
py - <<'PY'
import pickle, numpy as np, pandas as pd
from common import REL
from cachelib import T2_PANELS
W = "C:/Users/user/tfb/work/t2/"
a = pickle.load(open(W + "probs_repro_s46raw.pkl", "rb")); b = pickle.load(open(W + "probs_repro_s62raw.pkl", "rb"))
a0 = pickle.load(open(W + "probs_s46raw_t1Adapt2GapMLClipFdBnd_t2nomemNNonly_t4l2.pkl", "rb"))
b0 = pickle.load(open(W + "probs_s62raw_t1Adapt2GapMLClipFdBnd_t2simRNN_t4l2.pkl", "rb"))
print("s46 blend probs identical:", all(np.array_equal(a[k], a0[k]) for k in a0), "| q37f blend probs identical:", all(np.array_equal(b[k], b0[k]) for k in b0))
cond = {}
for sp in ("validation", "private"):
    for p in T2_PANELS:
        w = pd.read_csv(REL / "task2" / p / sp / "window_index.csv"); cond.update(dict(zip(w.window_id, w.condition)))
pickle.dump({k: (b[k] if cond[k] == "queue_ongoing" else a[k]) for k in a}, open(W + "probs_repro_s63raw.pkl", "wb"))
PY
py t2_onset_evidence.py repro_s63raw repro_s63on 0.25 0.85 | tail -1
py t2_far_gate.py ${W}t2/queue_repro_s63on.parquet ${W}t2/queue_repro_s63.parquet 3 1 | tail -1
py -c "
import pandas as pd
W='C:/Users/user/tfb/work/t2/'
for new, old in (('queue_repro_s60', 'queue_s60_t2OnB025Og085GateContig'), ('queue_repro_s63', 'queue_s63_t2HybridOnB025Og085GateContig')):
    a=pd.read_parquet(W+new+'.parquet'); b=pd.read_parquet(W+old+'.parquet')
    print(new, 'identical to the submitted T2:', a.equals(b))"
