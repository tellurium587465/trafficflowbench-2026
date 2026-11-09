#!/bin/bash
# 11/04 ledger #230 (A15): April's same-day X1 on origins every 3 rows (as repro/x1sdd3_chain.sh, X1_ARM=sdd3, pool nn_data_april_sp3) for
# the q24f networks of seeds 8-15 (#107), April only (A's March ongoing uses seeds 0-3, so no March keys are needed: no X1_FILL).
# Four queues of 4 threads, restart-safe (.done per network). usage: bash repro/x1sdd3_s16_chain.sh <queue 0-3>
cd /c/Users/user/tfb_a8/src; export TFB_WORK=C:/Users/user/tfb/work TFB_REL=C:/Users/user/tfb/data/kaggle_public X1_THREADS=4
export X1_SPLITS=private X1_ARM=sdd3 X1_POOL_PRIV=nn_data_april_sp3
Q=$1; i=0; L=$TFB_WORK/x1sdd3s16_q$Q.log
for s in 8 9 10 11 12 13 14 15; do
  for c in attnglob attnwp05 attn1 drop1 r13; do
    if [ $((i % 4)) = "$Q" ]; then
      t=q24f_${c}_s$s; f=$TFB_WORK/t2/probs_x1offsdd3__$t.pkl
      if [ ! -f "$f.done" ]; then
        py -B -X utf8 t2_x1_sameday.py $t 2>&1 | grep -v Warning | tail -1 >> $L && [ -f "$f" ] && touch "$f.done"
      fi
    fi
    i=$((i + 1))
  done
done
echo QUEUE_DONE $Q $(date) >> $L
