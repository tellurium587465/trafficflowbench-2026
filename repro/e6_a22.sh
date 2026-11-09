#!/bin/bash
# 11/05 ledger #252: E6 phases 1-4 for A22 and A22c (A21 / A25 with the April windows from q24m),
# one after another, own entry names (as e6_a21.sh / e6_b7.sh). Detached, restart-safe.
cd /c/Users/user/tfb_a8/src; export TFB_WORK=C:/Users/user/tfb/work TFB_REL=C:/Users/user/tfb/data/kaggle_public; L=../repro/logs
QA=C:/Users/user/tfb/work/t2/queue_repro_finA22t.parquet; QB=C:/Users/user/tfb/work/t2/queue_repro_finA22ct.parquet
grep -q "PASS\|FAIL" $L/trunc_finA22.txt 2>/dev/null || TFB_E6_TAG=A22 TFB_E6_ROOT=C:/tfb_data/e6_trunc_A22 py -B t2_trunc_physical.py A22 $QA --cpu 4 --only 1,3 > $L/trunc_finA22.txt 2>&1
grep -q "PASS\|FAIL" $L/trunc_finA22_p2.txt 2>/dev/null || TFB_E6_TAG=A22p2 TFB_E6_ROOT=C:/tfb_data/e6_trunc_A22p2 py -B t2_trunc_physical.py A22 $QA --only 2 > $L/trunc_finA22_p2.txt 2>&1
grep -q "PASS\|FAIL" $L/trunc_finA22_x1.txt 2>/dev/null || TFB_E6_TAG=A22x1 TFB_E6_ROOT=C:/tfb_data/e6_trunc_A22x1 py -B t2_trunc_physical.py A22 $QA --only 4 Q2_D12_I5_S_private_007 > $L/trunc_finA22_x1.txt 2>&1
grep -q "PASS\|FAIL" $L/trunc_finA22c.txt 2>/dev/null || TFB_E6_TAG=A22c TFB_E6_ROOT=C:/tfb_data/e6_trunc_A22c py -B t2_trunc_physical.py A22c $QB --cpu 4 > $L/trunc_finA22c.txt 2>&1
echo ALLDONE $(date) >> $L/e6_a22.status
