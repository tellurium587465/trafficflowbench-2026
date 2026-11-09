#!/bin/bash
# 10/31 finals A6 / B6: E6 (rows after T physically deleted), phases 1-4, one after the other. Logs in repro/logs/ (read-only for the finals).
cd /c/Users/user/tfb/src
date > ../repro/logs/e6_1031.status
py t2_trunc_physical.py A6 ../work/t2/queue_finA6.parquet --cpu 4 > ../repro/logs/trunc_finA6.txt 2>&1
echo "A6 exit $? $(date)" >> ../repro/logs/e6_1031.status
py t2_trunc_physical.py B6 ../work/t2/queue_repro_finB6.parquet --cpu 4 > ../repro/logs/trunc_finB6_1031.txt 2>&1
echo "B6 exit $? $(date)" >> ../repro/logs/e6_1031.status
echo ALLDONE >> ../repro/logs/e6_1031.status
