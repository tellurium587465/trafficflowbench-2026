#!/bin/bash
# 11/03 after the 05:51 reboot: #189 d3e4 (M3) and the January / February base arm, restart-safe per day; 3 threads each, detached.
cd /c/Users/user/tfb/review; export TFB_M3VER=v3; W=/c/Users/user/tfb/work/t2; L=$W/x1_months_dense; mkdir -p $L
m3() { [ -f $W/probs_m3v3v2x1_d3e4__c24f_attnglob_s$1.pkl ] || py -B -X utf8 g222_x1_dense.py run c24f_attnglob_s$1 3 4 3 >> $W/g222_d3e4_s$1.log 2>&1; }
jf() { [ -f $L/d6e4/capB0_s$1_fold2030-12.pkl ] || py -B -X utf8 g225_x1_jf_dense.py fit capB0_s$1_fold2030-12 6 4 3 >> $L/d6e4_s$1.log 2>&1; }
m3 0 & m3 1 & jf 0 & jf 1 &
wait
jf 2 & jf 3 &
wait
echo ALLDONE $(date) >> $L/resume.log
