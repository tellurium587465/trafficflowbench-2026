#!/bin/bash
# P2(c) (10/31, ledger #86) first local GPU job after the 10/21 resets: one epoch of X3b, fold 2, no torch.compile, Linear-only ops
# (no MIOpen kernels). Writes only to /mnt/c/tfb_data/tfb_nn/x3_smoke/.
source ~/rocm721venv/bin/activate
export HSA_ENABLE_DXG_DETECTION=1 MIOPEN_USER_DB_PATH=/home/user/miopen721 MIOPEN_CUSTOM_CACHE_DIR=/home/user/miopen721
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 MIOPEN_LOG_LEVEL=1 TORCH_BLAS_PREFER_HIPBLASLT=1
O=/mnt/c/tfb_data/tfb_nn/x3_smoke; mkdir -p $O
export NN_DATA=nn_data_afdem NN_TEST=nn_test_afdem NN_RAW=1 NN_RAWFILE=/mnt/c/tfb_data/tfb_nn/nn_data_afdem/R.npy NN_SAVEF=1 TFB_OUT=$O/ TFB_MODELS=$O PYTHONUNBUFFERED=1
Z=pos+tod+dow+capl+lanes+length+vf+w
export NN_CONFIGS="x3b_smoke,H=128,DEPTH=10,EP=1,LR=0.002,WD=0.001,K=5,FDROP=0.05,B=256,COMPILE=0,DROP=0,BUCKET=0,ZERO=$Z,WPOW=1,ATTN=1,GLOBAL=1,FOLDS=2,SEED=0,FINAL=0"
cd /mnt/c/Users/user/tfb/src
date > $O/log.txt
cp ../review/t2_nn3.py /tmp/t2_nn3.py && PYTHONPATH=/mnt/c/Users/user/tfb/src python -u /tmp/t2_nn3.py daemon 0 >> $O/log.txt 2>&1
echo "EXIT $?" >> $O/log.txt; date >> $O/log.txt
