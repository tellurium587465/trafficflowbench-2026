#!/bin/bash
# s71 = s70 with the ONSET probabilities Platt-calibrated before the decoder (review/g49_calibration.py: month-CV onset
# +0.0022, SE 0.0009, +2.3 SE; ongoing already calibrated -> unchanged). Same post-processing as s70 (B 0.25, OGB 0.85,
# far gate + contiguous growth). Pre-declared (P11, 10/14): kept as a final candidate unless LB < s70 - 0.004 (0.89127).
set -euo pipefail; export PATH="/usr/bin:$PATH"; cd /c/Users/user/tfb/src; W=C:/Users/user/tfb/work/
SRC=s70raw_t1Adapt2GapMLClipFdBnd_t2OnsetS46OngoingSimR24R48_t4l2
RAW=s71raw_t1Adapt2GapMLClipFdBnd_t2OnsetS46PlattOngoingSimR24R48_t4l2
py t2_onset_platt.py fit
py t2_onset_platt.py apply $SRC $RAW
py t2_onset_evidence.py $RAW s71on_t2HybridR24R48OnPlattB025Og085 0.25 0.85 | tail -1
py t2_far_gate.py ${W}t2/queue_s71on_t2HybridR24R48OnPlattB025Og085.parquet ${W}t2/queue_s71_t2HybridR24R48OnPlattB025Og085GateContig.parquet 3 1 | tail -1
py -c "
import pandas as pd
from submit import assemble
from common import REL; from cachelib import T2_PANELS
W='C:/Users/user/tfb/work/'
a=pd.read_parquet(W+'t2/queue_s71_t2HybridR24R48OnPlattB025Og085GateContig.parquet'); b=pd.read_parquet(W+'t2/queue_s70_t2HybridR24R48OnB025Og085GateContig.parquet')
assert (a.window_id.to_numpy() == b.window_id.to_numpy()).all()
cond={}
for sp in ('validation','private'):
    for p in T2_PANELS:
        x=pd.read_csv(REL/'task2'/p/sp/'window_index.csv'); cond.update(dict(zip(x.window_id,x.condition)))
c=a.window_id.map(cond); d=a.queue_pred!=b.queue_pred
assert not d[c=='queue_ongoing'].any(), 'ongoing windows changed'
w=a.window_id[d].unique(); print('s71 vs s70: cells', int(d.sum()), 'onset windows changed', len(w), 'validation', sum('validation' in x for x in w), '| predicted onset cells s70', int(b.queue_pred[c=='queue_onset'].sum()), 's71', int(a.queue_pred[c=='queue_onset'].sum()))
print(assemble(pd.read_parquet(W+'t1/state_f3ens3c_ad2_gapml_clip_fd_bnd.parquet'), a, pd.read_parquet(W+'t4/odme_l2.parquet'), 's71_t1Adapt2GapMLClipFdBnd_t2OnsetS46PlattOngoingSimR24R48_t4l2'))"
