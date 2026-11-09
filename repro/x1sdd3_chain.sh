#!/bin/bash
# 11/03 ledger #189 (A13): April's same-day X1 on origins every 3 rows (t2_x1_sameday.py X1_ARM=sdd3, pool nn_data_april_sp3 =
# twin/build_march.py 3 private with TFB_MONTH_OUT), q24f seeds 0-7, March kept from the sd files (X1_FILL=sd).
# Four queues of 4 threads, restart-safe (.done per network). usage: bash repro/x1sdd3_chain.sh <queue 0-3>
cd /c/Users/user/tfb_a8/src; export TFB_WORK=C:/Users/user/tfb/work TFB_REL=C:/Users/user/tfb/data/kaggle_public X1_THREADS=4
export X1_SPLITS=private X1_ARM=sdd3 X1_POOL_PRIV=nn_data_april_sp3 X1_FILL=sd
Q=$1; i=0; L=$TFB_WORK/x1sdd3_q$Q.log
for s in 0 1 2 3 4 5 6 7; do
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
