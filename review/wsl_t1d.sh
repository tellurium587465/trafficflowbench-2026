#!/bin/bash
# 11/03 ledger #194: the month fine-tune with the hide pattern thinned (rho 1, 0.3, 0.15), seed 4, stage A, local GPU under the Adrenalin limit, everything on C:.
source ~/rocm721venv/bin/activate
export HSA_ENABLE_DXG_DETECTION=1 MIOPEN_USER_DB_PATH=/home/user/miopen721 MIOPEN_CUSTOM_CACHE_DIR=/home/user/miopen721 MIOPEN_FIND_MODE=FAST MIOPEN_LOG_LEVEL=1 TORCH_BLAS_PREFER_HIPBLASLT=1
export U_ROOT=/mnt/c/tfb_data/kaggle_up/d_t1img/ U_W=/mnt/c/tfb_data/kaggle_out/t1unet3/ U_OUT=/mnt/c/tfb_data/tfb_nn/c13/ PYTHONUNBUFFERED=1
mkdir -p $U_OUT; cp /mnt/c/tfb_data/kaggle_up/k_t1mo/tfb-t1-ftd.py /tmp/t1d.py; cd /tmp
for r in ${RHOS:-1 0.3 0.15}; do U_RHO=$r python -u /tmp/t1d.py ${SEED:-4} >> /mnt/c/tfb_data/tfb_nn/t1d.log 2>&1; done
echo DONE $(date) >> /mnt/c/tfb_data/tfb_nn/t1d.log
