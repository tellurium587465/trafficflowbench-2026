#!/bin/bash
# 10/22 GPU chain, second version (replaces e4_gpu_chain.sh after its V24 OOM finding): the EMA A/B daemon is already
# running; then CPU-infer its FINAL models on M3, rerun the five A1 V24 jobs that hit HIP OOM (one job per daemon
# process: 28.2M rows leave too little VRAM after the first job), then the E4 proxy (x1, x4; none) and the sanity
# arm (b), CPU-infer (b) on M3. Windows side: py for inference, wsl for the GPU daemons.
export MSYS_NO_PATHCONV=1
L=/c/tfb_data/tfb_nn; LOG=$L/e4_chain.log
step() { echo "$(date '+%m-%d %H:%M') $1" >> $LOG; }
busy() { wsl.exe -d Ubuntu-24.04 -- pgrep -f "t2_nn2.py daemon" > /dev/null; }
while busy; do sleep 30; done
step "infer EMA A/B on M3"
cd /c/Users/user/tfb/src && py t2_nn_cpu_infer.py e4s_a_s0,e4s_a_s1,e4s_aema_s0,e4s_aema_s1 nn_test_m3 m3 >> $LOG 2>&1 || step "INFER_EMA_FAILED"
step "EMA_AB_DONE"
for i in 2 3 4 5 6; do
  step "A1 V24 rerun job $i"
  wsl.exe -d Ubuntu-24.04 -- bash /mnt/c/Users/user/tfb/repro/nn_configs/wsl_nnd_afdemsimV24.sh /mnt/c/tfb_data/tfb_nn/queue_a1_V_r$i.txt
done
step "A1_V_RERUN_DONE"
step "train proxy x1 x4"
wsl.exe -d Ubuntu-24.04 -- bash /mnt/c/Users/user/tfb/repro/nn_configs/wsl_nnd_e4p1.sh /mnt/c/tfb_data/tfb_nn/queue_e4p_1.txt
step "train proxy none"
wsl.exe -d Ubuntu-24.04 -- bash /mnt/c/Users/user/tfb/repro/nn_configs/wsl_nnd_e4p0.sh /mnt/c/tfb_data/tfb_nn/queue_e4p_0.txt
step "E4_PROXY_TRAINED"
step "train sanity b"
wsl.exe -d Ubuntu-24.04 -- bash /mnt/c/Users/user/tfb/repro/nn_configs/wsl_nnd_afdemM15.sh /mnt/c/tfb_data/tfb_nn/queue_e4s_b.txt
step "infer sanity b"
cd /c/Users/user/tfb/src && py t2_nn_cpu_infer.py e4s_b_s0,e4s_b_s1 nn_test_m3 m3 >> $LOG 2>&1 || { step "INFER_FAILED"; exit 1; }
step "E4_CHAIN_DONE"
