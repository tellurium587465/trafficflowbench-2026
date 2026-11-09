#!/bin/bash
# GPU daemon: data + compiled graphs stay resident; jobs = lines appended to C:/tfb_data/tfb_nn/queue.txt ("STOP" ends)
source ~/rocm721venv/bin/activate   # 10/22: torch 2.10 + ROCm 7.2.1 (stack updated 10/21)
export HSA_ENABLE_DXG_DETECTION=1 MIOPEN_USER_DB_PATH=/home/user/miopen721 MIOPEN_CUSTOM_CACHE_DIR=/home/user/miopen721   # 10/22: ROCDXG path (ROCm 7.2.1 on WSL)
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 MIOPEN_LOG_LEVEL=1 TORCH_BLAS_PREFER_HIPBLASLT=1
export TORCHINDUCTOR_CACHE_DIR=/dev/shm/inductor TRITON_CACHE_DIR=/dev/shm/triton   # compile caches in RAM, not the C: SSD
export NN_DATA=nn_data_afdemmp2 NN_TEST=nn_test_afdemmp2; export NN_QUEUE=${1:-/mnt/c/tfb_data/tfb_nn/queue.txt}
cd /mnt/c/Users/user/tfb/src
python -u t2_nn2.py daemon 0 >> /mnt/c/tfb_data/tfb_nn/daemon.log 2>&1