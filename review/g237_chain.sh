#!/bin/bash
# 11/03 ledger #189: G237 (the hold-out days' same-day X1 on the March pool every 3 rows) for c24f_attnglob s0, s1 side by side, 3 threads each; restart-safe per day.
cd /c/Users/user/tfb/review; export TFB_M3VER=v3; O=/c/Users/user/tfb/work/t2/x1_arms
for t in c24f_attnglob_s0 c24f_attnglob_s1; do py -B -X utf8 g237_sdd3_holdout.py $t > $O/g237_$t.out 2>&1 & done
wait; echo G237_DONE $(date) >> $O/g237.status
