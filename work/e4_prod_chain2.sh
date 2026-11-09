#!/bin/bash
# 10/22 E4 production, continued (replaces e4_prod_chain.sh after job 3): wait for the running job's daemon, run the
# speed test (BUCKET=3 + COMPILE=2 dynamic shapes, fold 2, real data, seeds 0,1; compared with a1_real_b0 = BUCKET 0),
# then wait for C:/tfb_data/tfb_nn/e4_bucket.txt ("0" keep, "3" switch BUCKET 0 jobs to BUCKET=3,COMPILE=2) and run the
# remaining jobs 4..20 one per daemon process.
export MSYS_NO_PATHCONV=1
L=/c/tfb_data/tfb_nn; LOG=$L/e4_prod.log
step() { echo "$(date '+%m-%d %H:%M') $1" >> $LOG; }
busy() { wsl.exe -d Ubuntu-24.04 -- pgrep -f "t2_nn2.py daemon" > /dev/null; }
while busy; do sleep 20; done
step "speed test BUCKET=3 COMPILE=2 (fold 2, real)"
wsl.exe -d Ubuntu-24.04 -- bash /mnt/c/Users/user/tfb/repro/nn_configs/wsl_nnd_afdem.sh /mnt/c/tfb_data/tfb_nn/queue_speedtest.txt
step "SPEED_TEST_DONE"
until [ -s $L/e4_bucket.txt ]; do sleep 15; done
BK=$(tr -d '\r\n ' < $L/e4_bucket.txt)
for j in $(seq 4 20); do
  f=$L/queue_e4f_j$j.txt
  if [ "$BK" = "3" ] && grep -q "BUCKET=0" $f; then sed -i 's/BUCKET=0/BUCKET=3/; s/COMPILE=1/COMPILE=2/' $f; fi
  step "E4 production job $j / 20 (bucket $BK): $(head -c 24 $f)"
  wsl.exe -d Ubuntu-24.04 -- bash /mnt/c/Users/user/tfb/repro/nn_configs/wsl_nnd_afdemMsimC24.sh /mnt/c/tfb_data/tfb_nn/queue_e4f_j$j.txt
done
step "E4_PROD_DONE"
