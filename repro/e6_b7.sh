#!/bin/bash
# 10/31 B7: E6 (rows after T physically deleted), phases 1-4 (ledger #97). Log in repro/logs/ (read-only for the finals).
cd /c/Users/user/tfb/src; date > ../repro/logs/e6_b7.status
py t2_trunc_physical.py B7 ../work/t2/queue_finB7.parquet --cpu 4 > ../repro/logs/trunc_finB7.txt 2>&1
echo "B7 exit $? $(date)" >> ../repro/logs/e6_b7.status; echo ALLDONE >> ../repro/logs/e6_b7.status
tail -1 ../repro/logs/trunc_finB7.txt
