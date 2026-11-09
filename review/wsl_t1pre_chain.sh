#!/bin/bash
# 11/03 ledger #192: after the seed-5 reference read (t1d.log ends with DONE), pretrain + read seed 4, then seed 5. Each step restart-safe.
until tail -3 /mnt/c/tfb_data/tfb_nn/t1d.log | grep -q "^DONE"; do sleep 30; done
for s in 4 5; do grep -q "READ DONE seed $s 0" /mnt/c/tfb_data/tfb_nn/t1pre.log 2>/dev/null || bash /mnt/c/Users/user/tfb/review/wsl_t1pre.sh $s; done
echo "CHAIN DONE $(date)" >> /mnt/c/tfb_data/tfb_nn/t1pre.log
