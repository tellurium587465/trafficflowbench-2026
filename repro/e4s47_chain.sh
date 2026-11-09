#!/bin/bash
# 10/23 (協議 12:40): E4 seeds 4-7 (A's April base to 8 seeds, the E5 rule applied to A as well), the e4f recipe unchanged
# (nn_data_afdemMsimC24, DOM=1, SIMW=0.5, EMA=0.998, MW=1; r13 DOM=0), one job per daemon process, after the c24f
# production; then X1 on the April windows for each model as it finishes (X1_TTA=0, X1_SPLITS=private).
export MSYS_NO_PATHCONV=1
L=/c/tfb_data/tfb_nn; LOG=$L/e4s47.log; W=/c/Users/user/tfb/work
step() { echo "$(date '+%m-%d %H:%M') $1" >> $LOG; }
until grep -q "C24_PROD_DONE" $L/c24_prod.log 2>/dev/null; do sleep 60; done
while wsl.exe -d Ubuntu-24.04 -- pgrep -f "t2_nn2.py daemon" > /dev/null; do sleep 20; done
cp $L/queue_e4f.txt $L/queue_e4f_s0to3_backup.txt
cd $W && py make_e4_queue.py nn_data_afdemMsimC24 e4f "DOM=1,SIMW=0.5,EMA=0.998" 4,5,6,7 >> $LOG 2>&1 || { step "QUEUE_FAILED"; exit 1; }
mv $L/queue_e4f.txt $L/queue_e4f47.txt; cp $L/queue_e4f_s0to3_backup.txt $L/queue_e4f.txt
i=0
while IFS= read -r line; do
  [ "$line" = "STOP" ] && break
  i=$((i+1)); case "$line" in *_r13_*) line=${line/DOM=1/DOM=0};; esac
  printf '%s\nSTOP\n' "$line" > $L/queue_e4f47_j$i.txt
done < $L/queue_e4f47.txt
for j in $(seq 1 $i); do
  step "E4 seeds 4-7 job $j / $i: $(head -c 24 $L/queue_e4f47_j$j.txt)"
  wsl.exe -d Ubuntu-24.04 -- bash /mnt/c/Users/user/tfb/repro/nn_configs/wsl_nnd_afdemMsimC24.sh /mnt/c/tfb_data/tfb_nn/queue_e4f47_j$j.txt
  tag=$(head -1 $L/queue_e4f47_j$j.txt | cut -d, -f1)
  if [ -s $W/t2/probs_$tag.pkl ]; then
    (cd /c/Users/user/tfb/src && X1_TTA=0 X1_SPLITS=private py t2_x1_official.py $tag 2>&1 | grep -v "Warning\|x\[j" >> $LOG) &
  else
    step "NO_PROBS $tag"
  fi
done
wait
step "E4S47_DONE"
