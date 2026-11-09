#!/bin/bash
# s64 = domain-token, half-synthetic sim-to-real NN blend (q41f: real train + randomized twin worlds, s46's 5 configs x seeds 0,1, s46 weights)
# + s60 post-processing (onset evidence B 0.25, ongoing OGB 0.85, far gate with contiguous-growth exemption)
set -euo pipefail; export PATH="/usr/bin:$PATH"; cd /c/Users/user/tfb/src; W=C:/Users/user/tfb/work/
SPEC=""
for s in 0 1; do SPEC="$SPEC+q41f_attnglob_s$s:2+q41f_attnwp05_s$s:2+q41f_attn1_s$s:2+q41f_drop1_s$s+q41f_r13_s$s"; done; SPEC=${SPEC#+}
N=s64raw_t1Adapt2GapMLClipFdBnd_t2simRd50NN_t4l2
py t2_blend_submit.py "$SPEC" $N ${W}t1/state_f3ens3c_ad2_gapml_clip_fd_bnd.parquet ${W}t4/odme_l2.parquet | tail -1
py t2_onset_evidence.py $N s64on_t2simRd50OnB025Og085 0.25 0.85 | tail -1
py t2_far_gate.py ${W}t2/queue_s64on_t2simRd50OnB025Og085.parquet ${W}t2/queue_s64_t2simRd50OnB025Og085GateContig.parquet 3 1 | tail -1
py -c "
import pandas as pd
from submit import assemble
W='C:/Users/user/tfb/work/'
a=pd.read_parquet(W+'t2/queue_s64_t2simRd50OnB025Og085GateContig.parquet'); b=pd.read_parquet(W+'t2/queue_s60_t2OnB025Og085GateContig.parquet')
d=a.queue_pred!=b.queue_pred; w=a.window_id[d].unique(); print('s64 vs s60: cells', int(d.sum()), 'windows', len(w), 'validation', sum('validation' in x for x in w), 'positives', int(a.queue_pred.sum()), 'vs', int(b.queue_pred.sum()))
print(assemble(pd.read_parquet(W+'t1/state_f3ens3c_ad2_gapml_clip_fd_bnd.parquet'), a, pd.read_parquet(W+'t4/odme_l2.parquet'), 's64_t1Adapt2GapMLClipFdBnd_t2simRd50OnB025Og085GateContig_t4l2'))"
