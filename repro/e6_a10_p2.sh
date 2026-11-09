#!/bin/bash
# 11/02: E6 phase 2 for A10 again, now also checking X1's row-T training inputs (nn_data_marcht, aprilt) rebuilt from the cut copy.
cd /c/Users/user/tfb_a8/src; export TFB_WORK=C:/Users/user/tfb/work TFB_REL=C:/Users/user/tfb/data/kaggle_public TFB_E6_ROOT=C:/tfb_data/e6_trunc_a10p2
L=../repro/logs
py -B t2_trunc_physical.py A10 C:/Users/user/tfb/work/t2/queue_finA10.parquet --only 2 > $L/trunc_finA10_p2.txt 2>&1; echo "A10 p2 exit $? $(date)" >> $L/trunc_finA10_p2.txt
