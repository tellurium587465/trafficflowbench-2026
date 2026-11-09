#!/bin/bash
# 10/22 A1 screen: after the real-data daemon ends, run the C24 then the V24 daemon (one GPU, data loaded once each).
while pgrep -f "t2_nn2.py daemon" > /dev/null; do sleep 20; done
echo "$(date '+%m-%d %H:%M') start C24" >> /mnt/c/tfb_data/tfb_nn/a1_chain.log
bash /mnt/c/Users/user/tfb/repro/nn_configs/wsl_nnd_afdemsimC24.sh /mnt/c/tfb_data/tfb_nn/queue_a1_C.txt
echo "$(date '+%m-%d %H:%M') start V24" >> /mnt/c/tfb_data/tfb_nn/a1_chain.log
bash /mnt/c/Users/user/tfb/repro/nn_configs/wsl_nnd_afdemsimV24.sh /mnt/c/tfb_data/tfb_nn/queue_a1_V.txt
echo "$(date '+%m-%d %H:%M') A1_CHAIN_DONE" >> /mnt/c/tfb_data/tfb_nn/a1_chain.log
