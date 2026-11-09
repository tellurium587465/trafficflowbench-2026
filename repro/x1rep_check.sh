#!/bin/bash
# 10/23 night: the reproduction's X1 stage at full scale under a separate arm ("rep") -> every final network's X1 compared
# with the stored X1 that the finals were built from (bit for bit, all 160 windows).
cd /c/Users/user/tfb
LOG=/c/Users/user/tfb/work/x1rep_check.log
STAGES="t2x1" X1_ARM=rep bash repro/run_final.sh > $LOG 2>&1
cd src && py - >> $LOG 2>&1 <<'PY'
import pickle, numpy as np
W = "C:/Users/user/tfb/work/t2/"; C = ("attnglob", "attnwp05", "attn1", "drop1", "r13")
tags = [f"q24f_{c}_s{s}" for s in range(8) for c in C] + [f"e4f_{c}_s{s}" for s in range(12) for c in C]
bad, n, dmax = [], 0, 0.0
for t in tags:
    a = pickle.load(open(W + f"probs_x1offrep__{t}.pkl", "rb")); b = pickle.load(open(W + f"probs_x1off__{t}.pkl", "rb"))
    for w in b:
        n += 1; d = float(np.abs(np.asarray(a[w], float) - np.asarray(b[w], float)).max()); dmax = max(dmax, d)
        if not np.array_equal(a[w], b[w]):
            bad.append((t, w))
print(f"X1 rebuilt (arm rep) vs stored: {len(tags)} networks, {n} window predictions, not bit-identical {len(bad)}, max |dp| {dmax:.2e}")
print("first differences:", bad[:5])
PY
echo X1REP_DONE >> $LOG
