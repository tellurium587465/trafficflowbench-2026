#!/bin/bash
# Disjoint-day test of the randomized-world effect: train worlds use template months 0-5 only (A6 = replay, S = 8
# randomized), test worlds U use template months 6-8 only (no shared day, demand series or mask with training).
set -e; export PATH="/usr/bin:$PATH"; cd /c/Users/user/tfb/twin; L=/c/Users/user/tfb/work/twin
py gen.py ALL 8 2 A6 0,1,2,3,4,5 > $L/gen_A6.log 2>&1
py gen.py ALL 8 3 S 0,1,2,3,4,5 > $L/gen_S.log 2>&1
py gen.py ALL 8 4 U 6,7,8 > $L/gen_U.log 2>&1
py build_nn2.py A6 - 0 U 8 djA > $L/build_djA.log 2>&1
py build_nn2.py A6 S 8 U 8 djAS > $L/build_djAS.log 2>&1
grep -E "done|Traceback" $L/build_djA.log $L/build_djAS.log
