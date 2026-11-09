#!/bin/bash
# 10/23 (協議 16:55): X1 of the q24f seeds 4-7 networks on the April windows (B's April ongoing to 8 seeds); no X1 file
# existed for these seeds, so the arm-"apr" output (March windows = the saved model, unused) becomes probs_x1off__<tag>.
cd /c/Users/user/tfb/src
LOG=/c/Users/user/tfb/work/x1apr_q24f47.log
T=$(py -c "print(','.join(f'q24f_{c}_s{s}' for s in range(4, 8) for c in ('attnglob','attnwp05','attn1','drop1','r13')))")
X1_TTA=0 X1_SPLITS=private py t2_x1_official.py $T 2e-4 apr 2>&1 | grep -v "Warning\|x\[j" > $LOG
for s in 4 5 6 7; do for c in attnglob attnwp05 attn1 drop1 r13; do
  f=/c/Users/user/tfb/work/t2/probs_x1off__q24f_${c}_s$s.pkl
  [ -e $f ] && { echo "EXISTS $f" >> $LOG; exit 1; }
  cp /c/Users/user/tfb/work/t2/probs_x1offapr__q24f_${c}_s$s.pkl $f
done; done
echo X1APR_Q24F47_DONE >> $LOG
