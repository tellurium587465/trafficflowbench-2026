#!/bin/bash
# 11/03 ledger #189: the January / February base arm (spacing 6, 4 epochs) on the four fold-2 networks, two at a time as the M3 arms of g222 free their threads.
cd /c/Users/user/tfb/review; W=/c/Users/user/tfb/work/t2; L=$W/x1_months_dense; mkdir -p $L
run() { [ -f $L/d6e4/capB0_s$1_fold2030-12.pkl ] || py -B -X utf8 g225_x1_jf_dense.py fit capB0_s$1_fold2030-12 6 4 4 > $L/d6e4_s$1.log 2>&1; }
until [ -f $W/probs_m3v3v2x1_d1e1__c24f_attnglob_s0.pkl ] && [ -f $W/probs_m3v3v2x1_d1e1__c24f_attnglob_s1.pkl ]; do sleep 30; done
run 0 & run 1 &
until [ -f $W/probs_m3v3v2x1_d3e4__c24f_attnglob_s0.pkl ] && [ -f $W/probs_m3v3v2x1_d3e4__c24f_attnglob_s1.pkl ]; do sleep 30; done
run 2 & run 3 &
wait
echo BASE_DONE $(date) >> $L/chain.log
