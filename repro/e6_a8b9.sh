#!/bin/bash
# 11/01 ledger #125: E6 for A8 and B9 (rows after T physically deleted), phases 1-3, then phase 4 (same-day X1 fine-tuned in place) on five windows of A8.
cd /c/Users/user/tfb_a8/src; export TFB_WORK=C:/Users/user/tfb/work TFB_REL=C:/Users/user/tfb/data/kaggle_public TFB_E6_ROOT=C:/tfb_data/e6_trunc_a8
L=../repro/logs; date > $L/e6_a8b9.status
py -B t2_trunc_physical.py A8 C:/Users/user/tfb/work/t2/queue_finA8.parquet --cpu 4 > $L/trunc_finA8.txt 2>&1; echo "A8 exit $? $(date)" >> $L/e6_a8b9.status
py -B t2_trunc_physical.py B9 C:/Users/user/tfb/work/t2/queue_finB9.parquet --cpu 4 > $L/trunc_finB9.txt 2>&1; echo "B9 exit $? $(date)" >> $L/e6_a8b9.status
echo ALLDONE >> $L/e6_a8b9.status
