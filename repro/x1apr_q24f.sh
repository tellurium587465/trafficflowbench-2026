#!/bin/bash
# 10/23 (協議 13:40: B = q24f + X1): X1 of the q24f seeds 0-3 networks on the April windows (arm "apr", no TTA draws),
# then the April entries merged into probs_x1off__q24f_* (the March X1 kept; the previous files backed up as *_before_apr).
cd /c/Users/user/tfb/src
LOG=/c/Users/user/tfb/work/x1apr_q24f.log
T=$(py -c "print(','.join(f'q24f_{c}_s{s}' for s in range(4) for c in ('attnglob','attnwp05','attn1','drop1','r13')))")
X1_TTA=0 X1_SPLITS=private py t2_x1_official.py $T 2e-4 apr 2>&1 | grep -v "Warning\|x\[j" > $LOG
py - >> $LOG 2>&1 <<'PY'
import pickle, shutil, pandas as pd
from common import REL
from cachelib import T2_PANELS
W = "C:/Users/user/tfb/work/t2/"
priv = {w for p in T2_PANELS for w in pd.read_csv(REL / "task2" / p / "private" / "window_index.csv").window_id}
for s in range(4):
    for c in ("attnglob", "attnwp05", "attn1", "drop1", "r13"):
        t = f"q24f_{c}_s{s}"; f = W + f"probs_x1off__{t}.pkl"; shutil.copyfile(f, W + f"probs_x1off__{t}_before_apr.pkl")
        old = pickle.load(open(f, "rb")); new = pickle.load(open(W + f"probs_x1offapr__{t}.pkl", "rb"))
        out = {w: (new[w] if w in priv else old[w]) for w in old}; pickle.dump(out, open(f, "wb"))
print("merged the April X1 into probs_x1off__q24f_* (March kept)")
PY
echo X1APR_Q24F_DONE >> $LOG
