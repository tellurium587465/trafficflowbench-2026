#!/bin/bash
# 11/05 ledger #252: as each q24m network finishes training (C:/tfb_data/tfb_nn/c18), copy its weights to NN/models, infer its base
# probabilities on the official windows (CPU, -> work/t2/probs_<tag>.pkl) and run April's dense same-day X1 on it (sdd3, pool
# nn_data_april_sp3, X1_SPLITS=private, X1_THREADS=4 as #230). One network at a time, in training order; restart-safe (.done).
cd /c/Users/user/tfb_a8/src; export TFB_WORK=C:/Users/user/tfb/work TFB_REL=C:/Users/user/tfb/data/kaggle_public X1_THREADS=4
C18=C:/tfb_data/tfb_nn/c18; M=C:/tfb_data/tfb_nn/models; W=$TFB_WORK/t2; L=$TFB_WORK/x1_q24m.log
for t in $(cut -d, -f1 $C18/queue.txt | grep "^q24m_"); do
  f=$W/probs_x1offsdd3__$t.pkl
  [ -f "$f.done" ] && continue
  until grep -q "^done $t " $C18/log.txt 2>/dev/null && [ -f $C18/$t.pt ]; do sleep 60; done
  cp -n $C18/$t.pt $M/$t.pt
  if [ ! -f $W/probs_$t.pkl ]; then
    py -B -X utf8 t2_nn_cpu_infer.py $t nn_test_afdem q24mb 2>&1 | grep -v Warning | tail -1 >> $L && mv $W/probs_q24mb__$t.pkl $W/probs_$t.pkl
  fi
  t0=$(date +%s)
  X1_SPLITS=private X1_ARM=sdd3 X1_POOL_PRIV=nn_data_april_sp3 py -B -X utf8 t2_x1_sameday.py $t 2>&1 | grep -v Warning | tail -1 >> $L && [ -f "$f" ] && touch "$f.done"
  echo "$t X1 $(( $(date +%s) - t0 ))s $(date +%H:%M:%S)" >> $L
done
echo ALLDONE $(date) >> $L
