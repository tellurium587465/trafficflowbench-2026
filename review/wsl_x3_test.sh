#!/bin/bash
# 10/31 local GPU test under the Adrenalin power / voltage limit (the user's condition): X3b then X3a, seed 1 = the same config and seed as
# Kaggle tfb-t2-x3 (a platform check). Everything on C:. Restart-safe: the daemon skips tags whose oof_<tag>.pkl exists.
source ~/rocm721venv/bin/activate
export HSA_ENABLE_DXG_DETECTION=1 MIOPEN_USER_DB_PATH=/home/user/miopen721 MIOPEN_CUSTOM_CACHE_DIR=/home/user/miopen721
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 MIOPEN_LOG_LEVEL=1 TORCH_BLAS_PREFER_HIPBLASLT=1
O=/mnt/c/tfb_data/tfb_nn/x3L; mkdir -p $O
export NN_DATA=nn_data_afdem NN_TEST=nn_test_afdem NN_RAW=1 NN_RAWFILE=/mnt/c/tfb_data/tfb_nn/nn_data_afdem/R.npy NN_SAVEF=1 TFB_OUT=$O/ TFB_MODELS=$O PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/c/Users/user/tfb/src
Z=pos+tod+dow+capl+lanes+length+vf+w
B="H=128,DEPTH=10,EP=24,LR=0.002,WD=0.001,K=5,FDROP=0.05,B=256,COMPILE=0,DROP=0,BUCKET=0,WPOW=1,ATTN=1,GLOBAL=1,FOLDS=2,FINAL=0"
cp /mnt/c/Users/user/tfb/review/t2_nn3.py /tmp/t2_nn3.py
cd /mnt/c/Users/user/tfb/src
Q=$O/queue.txt
printf 'x3bL_s1,%s,ZERO=%s,SEED=1\nx3aL_s1,%s,ZERO=ALL,COND=og,SEED=1\nSTOP\n' "$B" "$Z" "$B" > $Q
echo "START $(date)" >> $O/log.txt
NN_QUEUE=$Q python -u /tmp/t2_nn3.py daemon 0 >> $O/log.txt 2>&1
echo "ALLEXIT $? $(date)" >> $O/log.txt
