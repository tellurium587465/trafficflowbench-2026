#!/bin/bash
# 11/04 ledgers #107, #179 6: E6 phases 1 and 3 (4 CPU threads), then phase 2 for B11, own entry names; detached, restart-safe.
cd /c/Users/user/tfb_a8/src; export TFB_WORK=C:/Users/user/tfb/work TFB_REL=C:/Users/user/tfb/data/kaggle_public; L=../repro/logs
Q=C:/Users/user/tfb/work/t2/queue_repro_finB11t.parquet
grep -q "PASS\|FAIL" $L/trunc_finB11.txt 2>/dev/null || TFB_E6_TAG=B11 TFB_E6_ROOT=C:/tfb_data/e6_trunc_B11 py -B t2_trunc_physical.py B11 $Q --cpu 4 --only 1,3 > $L/trunc_finB11.txt 2>&1
grep -q "PASS\|FAIL" $L/trunc_finB11_p2.txt 2>/dev/null || TFB_E6_TAG=B11p2 TFB_E6_ROOT=C:/tfb_data/e6_trunc_B11p2 py -B t2_trunc_physical.py B11 $Q --only 2 > $L/trunc_finB11_p2.txt 2>&1
echo ALLDONE $(date) >> $L/e6_b11.status
