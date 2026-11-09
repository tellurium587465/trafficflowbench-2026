#!/bin/bash
# 11/05 ledger #252: q24f recipe on the nine training months + all of March (nn_data_afdem_mar), 8 seeds x CF5 = 40 networks (q24m_*), FINAL=1.
source ~/rocm721venv/bin/activate
export HSA_ENABLE_DXG_DETECTION=1 MIOPEN_USER_DB_PATH=/home/user/miopen721 MIOPEN_CUSTOM_CACHE_DIR=/home/user/miopen721
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 MIOPEN_LOG_LEVEL=1 TORCH_BLAS_PREFER_HIPBLASLT=1
cp /mnt/c/Users/user/tfb/review/t2_nn3t.py /tmp/t2_nn3t.py
cd /mnt/c/Users/user/tfb/src
run() {
  O=/mnt/c/tfb_data/tfb_nn/$1; mkdir -p $O; echo "START $(date)" >> $O/log.txt
  NN_DATA=$2 NN_TEST=nn_test_afdem NN_RAW=0 NN_RAWFILE=/mnt/c/tfb_data/tfb_nn/nn_data_afdem_mar/R.npy NN_SAVEF=1 TFB_OUT=$O/ TFB_MODELS=$O PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/c/Users/user/tfb/src NN_TROW=$3 NN_QUEUE=$O/queue.txt python -u /tmp/t2_nn3t.py daemon 0 >> $O/log.txt 2>&1
  echo "ALLEXIT $? $(date)" >> $O/log.txt
}
run c18 nn_data_afdem_mar ""
