#!/bin/bash
# 11/03 ledger #192 fair test: the local baseline (no test-month cells, February in training, b64, 45 epochs) -> the same stage A read as wsl_t1pre.sh.
# (rho 1, 32 epochs on the odd days, even days read at rho_read 0.15 / 0.3 / 1), compared with the seed-4 pretrained run (c14) read the same way.
# Local GPU under the Adrenalin limit, C: only, restart-safe (per-epoch checkpoint). usage: wsl_t1base.sh <seed> [dry]
source ~/rocm721venv/bin/activate
export HSA_ENABLE_DXG_DETECTION=1 MIOPEN_USER_DB_PATH=/home/user/miopen721 MIOPEN_CUSTOM_CACHE_DIR=/home/user/miopen721 MIOPEN_FIND_MODE=FAST MIOPEN_LOG_LEVEL=1 TORCH_BLAS_PREFER_HIPBLASLT=1
export U_ROOT=/mnt/c/tfb_data/kaggle_up/d_t1img/ PYTHONUNBUFFERED=1 U_BASE=64 U_EP=${U_EP:-45} U_PRE_DAYS=none
S=$1; O=/mnt/c/tfb_data/tfb_nn/c16/; mkdir -p $O; L=/mnt/c/tfb_data/tfb_nn/t1base.log
cp /mnt/c/Users/user/tfb/review/t1_unet_base192.py /tmp/t1base.py; cp /mnt/c/tfb_data/kaggle_up/k_t1mo/tfb-t1-ftd.py /tmp/t1d.py; cd /tmp
if [ "$2" = "dry" ]; then mkdir -p /mnt/c/tfb_data/tfb_nn/c16dry/; U_DRY=1 U_OUT=/mnt/c/tfb_data/tfb_nn/c16dry/ python -u /tmp/t1base.py $S >> $L 2>&1; echo "DRY DONE $? $(date)" >> $L; exit; fi
U_OUT=$O python -u /tmp/t1base.py $S >> $L 2>&1; echo "PRE DONE seed $S $? $(date)" >> $L
U_W=$O U_RHO=1 U_OUT=$O python -u /tmp/t1d.py $S >> $L 2>&1; echo "READ DONE seed $S $? $(date)" >> $L
