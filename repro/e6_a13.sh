#!/bin/bash
# 11/03 ledger #189: E6 phases 1-4 for A13, one after another, own entry names: phases 1 and 3 on 4 CPU threads, phase 2 (X1's training
# windows: April's on origins every 3 rows), phase 4 on the two windows A12 used (April's dense X1 and March's sd). Detached, restart-safe.
cd /c/Users/user/tfb_a8/src; export TFB_WORK=C:/Users/user/tfb/work TFB_REL=C:/Users/user/tfb/data/kaggle_public; L=../repro/logs
Q=C:/Users/user/tfb/work/t2/queue_repro_finA13t.parquet
grep -q "PASS\|FAIL" $L/trunc_finA13.txt 2>/dev/null || TFB_E6_TAG=A13 TFB_E6_ROOT=C:/tfb_data/e6_trunc_A13 py -B t2_trunc_physical.py A13 $Q --cpu 4 --only 1,3 > $L/trunc_finA13.txt 2>&1
grep -q "PASS\|FAIL" $L/trunc_finA13_p2.txt 2>/dev/null || TFB_E6_TAG=A13p2 TFB_E6_ROOT=C:/tfb_data/e6_trunc_A13p2 py -B t2_trunc_physical.py A13 $Q --only 2 > $L/trunc_finA13_p2.txt 2>&1
grep -q "PASS\|FAIL" $L/trunc_finA13_x1.txt 2>/dev/null || TFB_E6_TAG=A13x1 TFB_E6_ROOT=C:/tfb_data/e6_trunc_A13x1 py -B t2_trunc_physical.py A13 $Q --only 4 Q2_D12_I5_S_private_007 Q2_D7_I10_W_validation_008 > $L/trunc_finA13_x1.txt 2>&1
echo ALLDONE $(date) >> $L/e6_a13.status
