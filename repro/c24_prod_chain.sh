#!/bin/bash
# 10/23 C-world ongoing production (METAGAME_REPLY §22 / §24: the R world (q43f/q46f) out of A and B for review safety;
# s74 step 3): E4's recipe without March = train months + the C24 ongoing windows, the five production configurations x
# seeds 0-3, FINAL, DOM=1, SIMW=0.5, EMA=0.998 (r13: DOM=0, BUCKET 2 has no DOM, as e4f), one job per daemon process.
export MSYS_NO_PATHCONV=1
L=/c/tfb_data/tfb_nn; LOG=$L/c24_prod.log
step() { echo "$(date '+%m-%d %H:%M') $1" >> $LOG; }
step "merge nn_data_afdemsimC24og"
cd /c/Users/user/tfb/twin && py build_merge.py nn_data_afdemsimC24og nn_data_afdem "nn_data_twinC24:queue_ongoing:C" >> $LOG 2>&1 || { step "MERGE_FAILED"; exit 1; }
cd /c/Users/user/tfb/work && py make_e4_queue.py nn_data_afdemsimC24og c24f "DOM=1,SIMW=0.5,EMA=0.998" >> $LOG 2>&1 || { step "QUEUE_FAILED"; exit 1; }
i=0
while IFS= read -r line; do
  [ "$line" = "STOP" ] && break
  i=$((i+1)); case "$line" in *_r13_*) line=${line/DOM=1/DOM=0};; esac
  printf '%s\nSTOP\n' "$line" > $L/queue_c24f_j$i.txt
done < $L/queue_c24f.txt
for j in $(seq 1 $i); do
  step "C24 production job $j / $i: $(head -c 24 $L/queue_c24f_j$j.txt)"
  wsl.exe -d Ubuntu-24.04 -- bash /mnt/c/Users/user/tfb/repro/nn_configs/wsl_nnd_afdemsimC24og.sh /mnt/c/tfb_data/tfb_nn/queue_c24f_j$j.txt
done
step "C24_PROD_DONE"
