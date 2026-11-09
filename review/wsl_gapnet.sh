#!/bin/bash
# P9 (10/31, ledger #94): GapNet seeds 0 and 1 on the local GPU, everything on C:. Restart-safe: rerun this script (epoch checkpoints, finished seeds skipped).
source ~/rocm721venv/bin/activate
export HSA_ENABLE_DXG_DETECTION=1 MIOPEN_USER_DB_PATH=/home/user/miopen721 MIOPEN_CUSTOM_CACHE_DIR=/home/user/miopen721 PYTHONUNBUFFERED=1
O=/mnt/c/tfb_data/tfb_nn/gapnet; cp /mnt/c/Users/user/tfb/review/t1_gapnet.py /tmp/t1_gapnet.py
for s in ${1:-0 1}; do
  [ -f $O/pred_s$s.pkl ] || python -u /tmp/t1_gapnet.py $s ${2:-10} >> $O/log.txt 2>&1
done
echo "ALLEXIT $? $(date)" >> $O/log.txt
