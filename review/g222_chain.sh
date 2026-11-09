#!/bin/bash
# 11/03 ledger #189: the two dense X1 arms on the two M3 networks, four processes of 4 threads.
cd /c/Users/user/tfb/review; export TFB_M3VER=v3; L=/c/Users/user/tfb/work/t2/g222
for s in 0 1; do
  for a in "3 4" "1 1"; do
    set -- $a
    f=/c/Users/user/tfb/work/t2/probs_m3v3v2x1_d$1e$2__c24f_attnglob_s$s.pkl
    [ -f "$f" ] || py -B -X utf8 g222_x1_dense.py run c24f_attnglob_s$s $1 $2 4 > ${L}_d$1e$2_s$s.log 2>&1 &
  done
done
wait
echo ALLDONE $(date) >> ${L}_done.log
