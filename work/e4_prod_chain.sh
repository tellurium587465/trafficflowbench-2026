#!/bin/bash
# 10/22 E4 production (STATE.md 10/22: A1 -> C worlds, DOM=1, SIMW=0.5; EMA=0.998 under the statistical-only rule;
# MW from the E4 proxy, written to C:/tfb_data/tfb_nn/e4_mw.txt after the pre-registered judgement). Waits for the
# proxy / sanity chain to finish (GPU idle), merges train months + March + the C24 ongoing windows, then runs the
# 20 FINAL jobs one per daemon process (26M rows: the V24 OOM lesson).
export MSYS_NO_PATHCONV=1
L=/c/tfb_data/tfb_nn; LOG=$L/e4_prod.log
step() { echo "$(date '+%m-%d %H:%M') $1" >> $LOG; }
until grep -q "E4_CHAIN_DONE\|INFER_FAILED" $L/e4_chain.log 2>/dev/null && [ -s $L/e4_mw.txt ]; do sleep 60; done
MW=$(tr -d '\r\n ' < $L/e4_mw.txt)
step "merge nn_data_afdemMsimC24 (MW=$MW)"
cd /c/Users/user/tfb/twin && py build_merge.py nn_data_afdemMsimC24 nn_data_afdem nn_data_march "nn_data_twinC24:queue_ongoing:C" >> $LOG 2>&1 || { step "MERGE_FAILED"; exit 1; }
EXTRA="DOM=1,SIMW=0.5,EMA=0.998"; [ "$MW" != "1" ] && EXTRA="$EXTRA,MW=$MW"
cd /c/Users/user/tfb/work && py make_e4_queue.py nn_data_afdemMsimC24 e4f "$EXTRA" >> $LOG 2>&1 || { step "QUEUE_FAILED"; exit 1; }
i=0
while IFS= read -r line; do
  [ "$line" = "STOP" ] && break
  i=$((i+1)); printf '%s\nSTOP\n' "$line" > $L/queue_e4f_j$i.txt
done < $L/queue_e4f.txt
for j in $(seq 1 $i); do
  step "E4 production job $j / $i: $(head -c 24 $L/queue_e4f_j$j.txt)"
  wsl.exe -d Ubuntu-24.04 -- bash /mnt/c/Users/user/tfb/repro/nn_configs/wsl_nnd_afdemMsimC24.sh /mnt/c/tfb_data/tfb_nn/queue_e4f_j$j.txt
done
step "E4_PROD_DONE"
