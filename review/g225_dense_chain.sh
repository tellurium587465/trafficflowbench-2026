#!/bin/bash
# 11/03 ledger #189: the January / February dense arm d3e4 (M3's better arm on the top coverage third), restart-safe per day, alongside the
# base arm (g222_225_resume.sh): seeds 0, 1 now with 3 threads; seeds 2, 3 with 2 threads once the base arm's seeds 0, 1 are done (16 threads
# in all: base 2 x 3 + dense 2 x 3 + 2 x 2). Detached.
cd /c/Users/user/tfb/review; L=/c/Users/user/tfb/work/t2/x1_months_dense; mkdir -p $L
jd() { [ -f $L/d3e4/capB0_s$1_fold2030-12.pkl ] || py -B -X utf8 g225_x1_jf_dense.py fit capB0_s$1_fold2030-12 3 4 $2 >> $L/d3e4_s$1.log 2>&1; }
jd 0 3 & jd 1 3 &
until [ -f $L/d6e4/capB0_s0_fold2030-12.pkl ] && [ -f $L/d6e4/capB0_s1_fold2030-12.pkl ]; do sleep 60; done
jd 2 2 & jd 3 2 &
wait
echo DENSEDONE $(date) >> $L/resume.log
