#!/bin/bash
# 11/04 ledgers #216, #179 6: E6 phases 1 and 3 (4 CPU threads), then phase 2 for B12, own entry names; detached, restart-safe.
cd /c/Users/user/tfb_a8/src; export TFB_WORK=C:/Users/user/tfb/work TFB_REL=C:/Users/user/tfb/data/kaggle_public; L=../repro/logs
Q=C:/Users/user/tfb/work/t2/queue_repro_finB12t.parquet
grep -q "PASS\|FAIL" $L/trunc_finB12.txt 2>/dev/null || TFB_E6_TAG=B12 TFB_E6_ROOT=C:/tfb_data/e6_trunc_B12 py -B t2_trunc_physical.py B12 $Q --cpu 4 --only 1,3 > $L/trunc_finB12.txt 2>&1
grep -q "PASS\|FAIL" $L/trunc_finB12_p2.txt 2>/dev/null || TFB_E6_TAG=B12p2 TFB_E6_ROOT=C:/tfb_data/e6_trunc_B12p2 py -B t2_trunc_physical.py B12 $Q --only 2 > $L/trunc_finB12_p2.txt 2>&1
echo ALLDONE $(date) >> $L/e6_b12.status
