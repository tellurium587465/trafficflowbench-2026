#!/bin/bash
# 11/02 ledger #164: E6 for A10 (rows after T physically deleted, the row at T kept), phases 1-3, then phase 4 (same-day X1 fine-tuned in place, the pool with the row at T) on five windows.
cd /c/Users/user/tfb_a8/src; export TFB_WORK=C:/Users/user/tfb/work TFB_REL=C:/Users/user/tfb/data/kaggle_public TFB_E6_ROOT=C:/tfb_data/e6_trunc_a10
L=../repro/logs; date > $L/e6_a10.status
py -B t2_trunc_physical.py A10 C:/Users/user/tfb/work/t2/queue_finA10.parquet --cpu 4 > $L/trunc_finA10.txt 2>&1; echo "A10 exit $? $(date)" >> $L/e6_a10.status
echo ALLDONE >> $L/e6_a10.status
