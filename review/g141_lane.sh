#!/bin/bash
# G141 (ledger #101-#103): one CPU lane = a list of arm:seed, run in order; tton = the onset networks at the flow scales
cd /c/Users/user/tfb/review; export TFB_M3VER=v3
for a in "$@"; do
  if [ "$a" = tton ]; then
    py g141_x1_arms.py tton $(py -c "print(','.join(f'q24f_{c}_s{s}' for s in range(8) for c in ('attnglob', 'attnwp05', 'attn1', 'drop1', 'r13')))") >> ../work/t2/x1_arms/tton.log 2>&1
  else
    py g141_x1_arms.py fit ${a%:*} c24f_attnglob_s${a#*:} >> ../work/t2/x1_arms/lane.log 2>&1
  fi
  echo "$a done $(date)" >> ../work/t2/x1_arms/lane.log
done
