#!/bin/bash
# 11/02 ledger #150: month fine-tune stage A of the nets without memory, local GPU, under the Adrenalin limit, everything on C:.
source ~/rocm721venv/bin/activate
export HSA_ENABLE_DXG_DETECTION=1 MIOPEN_USER_DB_PATH=/home/user/miopen721 MIOPEN_CUSTOM_CACHE_DIR=/home/user/miopen721 MIOPEN_FIND_MODE=FAST MIOPEN_LOG_LEVEL=1 TORCH_BLAS_PREFER_HIPBLASLT=1
export U_ROOT=/mnt/c/tfb_data/kaggle_up/d_t1img/ U_W=/mnt/c/tfb_data/tfb_nn/c9/ U_OUT=/mnt/c/tfb_data/tfb_nn/c9/ PYTHONUNBUFFERED=1
cp /mnt/c/tfb_data/kaggle_up/k_t1mo/tfb-t1-nomem-ft.py /tmp/t1ft.py; cd /tmp
for s in 6 7; do python -u /tmp/t1ft.py $s >> /mnt/c/tfb_data/tfb_nn/t1ft_nm.log 2>&1; done
echo DONE $(date) >> /mnt/c/tfb_data/tfb_nn/t1ft_nm.log
