#!/bin/bash
# 11/04 ledger #212: G248 (M3 dev same-day X1 with the fine-tune's seed 1) for c24f_attnglob s0, s1 side by side, 3 threads each; restart-safe per day.
cd /c/Users/user/tfb/review; export TFB_M3VER=v3; O=/c/Users/user/tfb/work/t2/x1_arms
for t in c24f_attnglob_s0 c24f_attnglob_s1; do py -B -X utf8 g248_x1_seed1.py $t > $O/g248_$t.out 2>&1 & done
wait; echo G248_DONE $(date) >> $O/g248.status
