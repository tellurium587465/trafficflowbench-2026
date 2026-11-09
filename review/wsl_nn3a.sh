#!/bin/bash
# 11/03 ledger #196: review/t2_nn3a.py (t2_nn3.py + AW, jam-age weights) on the local GPU under the Adrenalin limit, everything on C:; the same environment
# as wsl_nn3_q.sh (b0L). Restart-safe per run: tags with an oof_<tag>.pkl are skipped.
source ~/rocm721venv/bin/activate
export HSA_ENABLE_DXG_DETECTION=1 MIOPEN_USER_DB_PATH=/home/user/miopen721 MIOPEN_CUSTOM_CACHE_DIR=/home/user/miopen721
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 MIOPEN_LOG_LEVEL=1 TORCH_BLAS_PREFER_HIPBLASLT=1
O=/mnt/c/tfb_data/tfb_nn/c15; mkdir -p $O
export NN_DATA=nn_data_afdem NN_TEST=nn_test_afdem NN_RAW=0 NN_RAWFILE=/mnt/c/tfb_data/tfb_nn/nn_data_afdem/R.npy NN_SAVEF=1 TFB_OUT=$O/ TFB_MODELS=$O PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/c/Users/user/tfb/src
export NN_AGEFILE=/mnt/c/tfb_data/tfb_nn/age196/aw.npy
cp /mnt/c/Users/user/tfb/review/t2_nn3a.py /tmp/t2_nn3a.py
cd /mnt/c/Users/user/tfb/src
echo "START $(date)" >> $O/log.txt
NN_QUEUE=$O/queue.txt python -u /tmp/t2_nn3a.py daemon 0 >> $O/log.txt 2>&1
echo "ALLEXIT $? $(date)" >> $O/log.txt
