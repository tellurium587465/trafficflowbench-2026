#!/bin/bash
# 11/05 ledger #232: E6 phases 1-4 for A16, one after another, own entry names: phases 1 and 3 on 4 CPU threads, phase 2 (X1's training
# windows: April's on origins every 3 rows), phase 4 on one April ongoing window (X1 unchanged from A16; phase 3 covers the rank-input network on its own entry). Detached, restart-safe.
cd /c/Users/user/tfb_a8/src; export TFB_WORK=C:/Users/user/tfb/work TFB_REL=C:/Users/user/tfb/data/kaggle_public; L=../repro/logs
Q=C:/Users/user/tfb/work/t2/queue_repro_finA16t.parquet
grep -q "PASS\|FAIL" $L/trunc_finA16.txt 2>/dev/null || TFB_E6_TAG=A16 TFB_E6_ROOT=C:/tfb_data/e6_trunc_A16 py -B t2_trunc_physical.py A16 $Q --cpu 4 --only 1,3 > $L/trunc_finA16.txt 2>&1
grep -q "PASS\|FAIL" $L/trunc_finA16_p2.txt 2>/dev/null || TFB_E6_TAG=A16p2 TFB_E6_ROOT=C:/tfb_data/e6_trunc_A16p2 py -B t2_trunc_physical.py A16 $Q --only 2 > $L/trunc_finA16_p2.txt 2>&1
grep -q "PASS\|FAIL" $L/trunc_finA16_x1.txt 2>/dev/null || TFB_E6_TAG=A16x1 TFB_E6_ROOT=C:/tfb_data/e6_trunc_A16x1 py -B t2_trunc_physical.py A16 $Q --only 4 Q2_D12_I5_S_private_007 > $L/trunc_finA16_x1.txt 2>&1
echo ALLDONE $(date) >> $L/e6_a16.status
