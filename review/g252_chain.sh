#!/bin/bash
# 11/04 ledger #217: G252's three refits side by side (3 threads each).
cd /c/Users/user/tfb/review; export TFB_M3VER=v3; O=/c/Users/user/tfb/work/t2/onset_place
for h in 2031-01 2031-02 all; do [ -f $O/g252_$h.pt ] || py -B -X utf8 g252_site_fit.py $h > $O/g252_fit_$h.out 2>&1 & done
wait; echo G252_DONE $(date) >> $O/g252.status
