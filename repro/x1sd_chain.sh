#!/bin/bash
# 11/01 ledger #125: same-day X1 of the final networks (t2_x1_sameday.py): q24f seeds 0-3 on both months, seeds 4-7 on April. Four queues of 4 threads, restart-safe.
# usage: bash repro/x1sd_chain.sh <queue 0-3>
cd /c/Users/user/tfb_a8/src; export TFB_WORK=C:/Users/user/tfb/work TFB_REL=C:/Users/user/tfb/data/kaggle_public X1_THREADS=4
Q=$1; i=0; L=$TFB_WORK/x1sd_q$Q.log
for s in 0 1 2 3 4 5 6 7; do
  for c in attnglob attnwp05 attn1 drop1 r13; do
    if [ $((i % 4)) = "$Q" ]; then
      t=q24f_${c}_s$s; f=$TFB_WORK/t2/probs_x1offsd__$t.pkl
      if [ ! -f "$f.done" ]; then
        if [ $s -lt 4 ]; then sp=validation,private; else sp=private; fi
        X1_SPLITS=$sp py -B -X utf8 t2_x1_sameday.py $t 2>&1 | grep -v Warning | tail -1 >> $L && touch "$f.done"
      fi
    fi
    i=$((i + 1))
  done
done
echo QUEUE_DONE $Q $(date) >> $L
