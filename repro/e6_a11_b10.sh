#!/bin/bash
# 11/03 ledger #201: E6 phases 1 and 3 for A12 and B10 (phase 2, X1's training windows, is the code path A8 / B9s passed). Detached, restart-safe by candidate.
cd /c/Users/user/tfb_a8/src; export TFB_WORK=C:/Users/user/tfb/work TFB_REL=C:/Users/user/tfb/data/kaggle_public
L=../repro/logs
run() { grep -q "PASS\|FAIL" $L/trunc_fin$1.txt 2>/dev/null || TFB_E6_TAG=$1 TFB_E6_ROOT=C:/tfb_data/e6_trunc_$1 py -B t2_trunc_physical.py $1 C:/Users/user/tfb/work/t2/queue_repro_fin$1t.parquet --cpu 4 --only 1,3 > $L/trunc_fin$1.txt 2>&1; }
run A12 & run B10 &
wait; echo ALLDONE $(date) >> $L/e6_a11_b10.status
