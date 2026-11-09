#!/bin/bash
# 11/01 ledger #171: E6 for B9s (rows after T physically deleted), phases 1-3.
cd /c/Users/user/tfb_a8/src; export TFB_WORK=C:/Users/user/tfb/work TFB_REL=C:/Users/user/tfb/data/kaggle_public TFB_E6_ROOT=C:/tfb_data/e6_trunc_a8
L=../repro/logs; date > $L/e6_b9s.status
py -B t2_trunc_physical.py B9s C:/Users/user/tfb/work/t2/queue_finB9s.parquet --cpu 4 > $L/trunc_finB9s.txt 2>&1; echo "B9s exit $? $(date)" >> $L/e6_b9s.status
echo ALLDONE >> $L/e6_b9s.status
