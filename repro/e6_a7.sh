#!/bin/bash
# 10/31 A7: E6 (rows after T physically deleted), phases 1-3 (ledger #110). Log in repro/logs/ (read-only for the finals).
cd /c/Users/user/tfb/src; date > ../repro/logs/e6_a7.status
py t2_trunc_physical.py A7 ../work/t2/queue_finA7.parquet --cpu 4 > ../repro/logs/trunc_finA7.txt 2>&1
echo "A7 exit $? $(date)" >> ../repro/logs/e6_a7.status; echo ALLDONE >> ../repro/logs/e6_a7.status
tail -1 ../repro/logs/trunc_finA7.txt
