#!/bin/bash
# 11/03 ledger #201: after phases 1 and 3, phase 2 for A12 and B10 and phase 4 for A12 (the two windows A8 used), one after another, own entry names.
cd /c/Users/user/tfb_a8/src; export TFB_WORK=C:/Users/user/tfb/work TFB_REL=C:/Users/user/tfb/data/kaggle_public; L=../repro/logs
until grep -q ALLDONE $L/e6_a11_b10.status 2>/dev/null; do sleep 60; done
for c in A12 B10; do grep -q "PASS\|FAIL" $L/trunc_fin${c}_p2.txt 2>/dev/null || TFB_E6_TAG=${c}p2 TFB_E6_ROOT=C:/tfb_data/e6_trunc_${c}p2 py -B t2_trunc_physical.py $c C:/Users/user/tfb/work/t2/queue_repro_fin${c}t.parquet --only 2 > $L/trunc_fin${c}_p2.txt 2>&1; done
grep -q "PASS\|FAIL" $L/trunc_finA12_x1.txt 2>/dev/null || TFB_E6_TAG=A12x1 TFB_E6_ROOT=C:/tfb_data/e6_trunc_A12x1 py -B t2_trunc_physical.py A12 C:/Users/user/tfb/work/t2/queue_repro_finA12t.parquet --only 4 Q2_D12_I5_S_private_007 Q2_D7_I10_W_validation_008 > $L/trunc_finA12_x1.txt 2>&1
echo ALLDONE $(date) >> $L/e6_a12_b10_p24.status
