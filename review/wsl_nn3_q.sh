#!/bin/bash
# 10/31 local GPU queue for review/t2_nn3.py (under the Adrenalin power / voltage limit). usage: wsl_nn3_q.sh <dir under C:/tfb_data/tfb_nn> <raw 0|1>
# <dir>/queue.txt holds the config lines (STOP last). Everything on C:. Restart-safe: tags with an oof_<tag>.pkl are skipped.
source ~/rocm721venv/bin/activate
export HSA_ENABLE_DXG_DETECTION=1 MIOPEN_USER_DB_PATH=/home/user/miopen721 MIOPEN_CUSTOM_CACHE_DIR=/home/user/miopen721
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 MIOPEN_LOG_LEVEL=1 TORCH_BLAS_PREFER_HIPBLASLT=1
O=/mnt/c/tfb_data/tfb_nn/$1; mkdir -p $O
export NN_DATA=nn_data_afdem NN_TEST=nn_test_afdem NN_RAW=$2 NN_RAWFILE=/mnt/c/tfb_data/tfb_nn/nn_data_afdem/R.npy NN_SAVEF=1 TFB_OUT=$O/ TFB_MODELS=$O PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/c/Users/user/tfb/src
cp /mnt/c/Users/user/tfb/review/t2_nn3.py /tmp/t2_nn3.py
cd /mnt/c/Users/user/tfb/src
echo "START $(date)" >> $O/log.txt
NN_QUEUE=$O/queue.txt python -u /tmp/t2_nn3.py daemon 0 >> $O/log.txt 2>&1
echo "ALLEXIT $? $(date)" >> $O/log.txt
