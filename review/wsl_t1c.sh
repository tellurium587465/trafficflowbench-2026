#!/bin/bash
# 11/02 ledger #148: the conservation term on seeds 4, 5, local GPU, under the Adrenalin limit, everything on C:.
source ~/rocm721venv/bin/activate
export HSA_ENABLE_DXG_DETECTION=1 MIOPEN_USER_DB_PATH=/home/user/miopen721 MIOPEN_CUSTOM_CACHE_DIR=/home/user/miopen721 MIOPEN_FIND_MODE=FAST MIOPEN_LOG_LEVEL=1 TORCH_BLAS_PREFER_HIPBLASLT=1
export U_ROOT=/mnt/c/tfb_data/kaggle_up/d_t1img/ U_W=/mnt/c/tfb_data/kaggle_out/t1unet3/ U_OUT=/mnt/c/tfb_data/tfb_nn/c10/ PYTHONUNBUFFERED=1
mkdir -p $U_OUT; cp /mnt/c/tfb_data/kaggle_up/k_t1mo/tfb-t1-cons.py /tmp/t1c.py; cd /tmp
python -u /tmp/t1c.py 4,5 >> /mnt/c/tfb_data/tfb_nn/t1c.log 2>&1
echo DONE $? $(date) >> /mnt/c/tfb_data/tfb_nn/t1c.log
