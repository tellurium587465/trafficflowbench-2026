#!/bin/bash
# 10/23 after each c24f production model (c24_prod_chain.sh writes work/t2/probs_<tag>.pkl at its end), on the CPU:
#  X1 on the official windows of both months (t2_x1_official.py, current recipe, no TTA draws), and for attnglob seeds
#  0 / 1 the M3 v3 inference + M3 v3 X1 (the B base of the production-order stack, METAGAME_REPLY §23 / §24); at the end
#  the M3 v3 inference of all 20.
export MSYS_NO_PATHCONV=1
W=/c/Users/user/tfb/work; LOG=$W/c24_down.log
step() { echo "$(date '+%m-%d %H:%M') $1" >> $LOG; }
cd /c/Users/user/tfb/src
ALL=""
for s in 0 1 2 3; do for c in attnglob attnwp05 attn1 drop1 r13; do
  tag=c24f_${c}_s$s; ALL="$ALL,$tag"
  until [ -s $W/t2/probs_$tag.pkl ]; do sleep 60; done
  step "X1 official $tag"
  X1_TTA=0 py t2_x1_official.py $tag 2>&1 | grep -v "Warning\|x\[j" >> $LOG || step "X1_FAILED $tag"
  if [ "$c" = attnglob ] && [ $s -le 1 ]; then
    step "M3 v3 inference + X1 $tag"
    py t2_nn_cpu_infer.py $tag nn_test_m3v3 m3v3 >> $LOG 2>&1 || step "M3_INFER_FAILED $tag"
    TFB_M3VER=v3 py t2_m3v2_x1.py $tag 2>&1 | grep -v "Warning\|x\[j" >> $LOG || step "M3_X1_FAILED $tag"
  fi
done; done
step "M3 v3 inference of all 20"
py t2_nn_cpu_infer.py ${ALL#,} nn_test_m3v3 m3v3 >> $LOG 2>&1 || step "M3_INFER_FAILED all"
step "C24_DOWN_DONE"
