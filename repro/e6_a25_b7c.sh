#!/bin/bash
# 11/05 ledger #283: E6 phases 1-4 for A25 (= A21 + the selection-conditioned ongoing, t2_selcond) and phases 1-4 for B7c (= B7 + the same),
# one after another, own entry names (as e6_a21.sh / e6_b7.sh). Detached, restart-safe.
cd /c/Users/user/tfb_a8/src; export TFB_WORK=C:/Users/user/tfb/work TFB_REL=C:/Users/user/tfb/data/kaggle_public; L=../repro/logs
QA=C:/Users/user/tfb/work/t2/queue_repro_finA25t.parquet; QB=C:/Users/user/tfb/work/t2/queue_repro_finB7ct.parquet
grep -q "PASS\|FAIL" $L/trunc_finA25.txt 2>/dev/null || TFB_E6_TAG=A25 TFB_E6_ROOT=C:/tfb_data/e6_trunc_A25 py -B t2_trunc_physical.py A25 $QA --cpu 4 --only 1,3 > $L/trunc_finA25.txt 2>&1
grep -q "PASS\|FAIL" $L/trunc_finA25_p2.txt 2>/dev/null || TFB_E6_TAG=A25p2 TFB_E6_ROOT=C:/tfb_data/e6_trunc_A25p2 py -B t2_trunc_physical.py A25 $QA --only 2 > $L/trunc_finA25_p2.txt 2>&1
grep -q "PASS\|FAIL" $L/trunc_finA25_x1.txt 2>/dev/null || TFB_E6_TAG=A25x1 TFB_E6_ROOT=C:/tfb_data/e6_trunc_A25x1 py -B t2_trunc_physical.py A25 $QA --only 4 Q2_D12_I5_S_private_007 > $L/trunc_finA25_x1.txt 2>&1
grep -q "PASS\|FAIL" $L/trunc_finB7c.txt 2>/dev/null || TFB_E6_TAG=B7c TFB_E6_ROOT=C:/tfb_data/e6_trunc_B7c py -B t2_trunc_physical.py B7c $QB --cpu 4 > $L/trunc_finB7c.txt 2>&1
echo ALLDONE $(date) >> $L/e6_a25_b7c.status
