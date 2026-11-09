#!/bin/bash
# 11/05 ledgers #230, #231: E6 phases 1-4 for A15, one after another, own entry names: phases 1 and 3 on 4 CPU threads, phase 2 (X1's training
# windows: April's on origins every 3 rows), phase 4 on two April ongoing windows whose X1 now includes seeds 8-15 (one A13 used, one where A15 changed a cell). Detached, restart-safe.
cd /c/Users/user/tfb_a8/src; export TFB_WORK=C:/Users/user/tfb/work TFB_REL=C:/Users/user/tfb/data/kaggle_public; L=../repro/logs
Q=C:/Users/user/tfb/work/t2/queue_repro_finA15t.parquet
grep -q "PASS\|FAIL" $L/trunc_finA15.txt 2>/dev/null || TFB_E6_TAG=A15 TFB_E6_ROOT=C:/tfb_data/e6_trunc_A15 py -B t2_trunc_physical.py A15 $Q --cpu 4 --only 1,3 > $L/trunc_finA15.txt 2>&1
grep -q "PASS\|FAIL" $L/trunc_finA15_p2.txt 2>/dev/null || TFB_E6_TAG=A15p2 TFB_E6_ROOT=C:/tfb_data/e6_trunc_A15p2 py -B t2_trunc_physical.py A15 $Q --only 2 > $L/trunc_finA15_p2.txt 2>&1
grep -q "PASS\|FAIL" $L/trunc_finA15_x1.txt 2>/dev/null || TFB_E6_TAG=A15x1 TFB_E6_ROOT=C:/tfb_data/e6_trunc_A15x1 py -B t2_trunc_physical.py A15 $Q --only 4 Q2_D12_I5_S_private_007 Q2_D12_I5_S_private_010 > $L/trunc_finA15_x1.txt 2>&1
echo ALLDONE $(date) >> $L/e6_a15.status
