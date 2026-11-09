#!/bin/bash
# 11/02 ledger #150: the image network on the local GPU, under the Adrenalin limit, everything on C:. usage: wsl_t1u.sh <seed> <log> [epochs]
source ~/rocm721venv/bin/activate
export HSA_ENABLE_DXG_DETECTION=1 MIOPEN_USER_DB_PATH=/home/user/miopen721 MIOPEN_CUSTOM_CACHE_DIR=/home/user/miopen721 MIOPEN_FIND_MODE=FAST MIOPEN_LOG_LEVEL=1 TORCH_BLAS_PREFER_HIPBLASLT=1
export U_ROOT=/mnt/c/tfb_data/kaggle_up/d_t1img/ U_OUT=/mnt/c/tfb_data/tfb_nn/c9/ PYTHONUNBUFFERED=1 U_EP=${3:-60} U_DRY=${U_DRY:-0}
mkdir -p $U_OUT; cp /mnt/c/tfb_data/kaggle_up/k_t1mo/tfb-t1-nomem.py /tmp/t1u.py; cd /tmp
timeout ${LIM:-1500} python -u /tmp/t1u.py $1 >> /mnt/c/tfb_data/tfb_nn/$2 2>&1
echo DONE $? $(date) >> /mnt/c/tfb_data/tfb_nn/$2
