#!/bin/bash
# s73 = s70 + onset Platt calibration (s71) + evidence rule B1 (s72). B1 alone (s72) lost 0.0021 on its 3 changed validation
# windows: the x0.25 shrink also acted as calibration of the over-confident onset network (G49 b ~ 0.64); with
# the network calibrated first, the uniform shrink is a pure distortion, as in the twin v4 instrument (+0.0200, 3.0 SE).
# Decision (10/14 morning): compare cells vs s70; submit when a slot is free; kept unless LB < s70 - 0.004.
set -euo pipefail; export PATH="/usr/bin:$PATH"; cd /c/Users/user/tfb/src; W=C:/Users/user/tfb/work/
SRC=s71raw_t1Adapt2GapMLClipFdBnd_t2OnsetS46PlattOngoingSimR24R48_t4l2
py t2_onset_evidence.py $SRC s73on_t2HybridR24R48OnPlattB025Og085Skip 0.25 0.85 1 | tail -1
py t2_far_gate.py ${W}t2/queue_s73on_t2HybridR24R48OnPlattB025Og085Skip.parquet ${W}t2/queue_s73_t2HybridR24R48OnPlattB025Og085SkipGateContig.parquet 3 1 | tail -1
py -c "
import pandas as pd
from submit import assemble
from common import REL; from cachelib import T2_PANELS
W='C:/Users/user/tfb/work/'
a=pd.read_parquet(W+'t2/queue_s73_t2HybridR24R48OnPlattB025Og085SkipGateContig.parquet'); b=pd.read_parquet(W+'t2/queue_s70_t2HybridR24R48OnB025Og085GateContig.parquet')
assert (a.window_id.to_numpy() == b.window_id.to_numpy()).all()
cond={}
for sp in ('validation','private'):
    for p in T2_PANELS:
        x=pd.read_csv(REL/'task2'/p/sp/'window_index.csv'); cond.update(dict(zip(x.window_id,x.condition)))
c=a.window_id.map(cond); d=a.queue_pred!=b.queue_pred
assert not d[c=='queue_ongoing'].any(), 'ongoing windows changed'
w=a.window_id[d].unique(); print('s73 vs s70: cells', int(d.sum()), 'onset windows changed', len(w), 'validation', sum('validation' in x for x in w), '| predicted onset cells s70', int(b.queue_pred[c=='queue_onset'].sum()), 's73', int(a.queue_pred[c=='queue_onset'].sum()))
print(assemble(pd.read_parquet(W+'t1/state_f3ens3c_ad2_gapml_clip_fd_bnd.parquet'), a, pd.read_parquet(W+'t4/odme_l2.parquet'), 's73_t1Adapt2GapMLClipFdBnd_t2OnsetS46PlattB1OngoingSimR24R48_t4l2'))"
