#!/bin/bash
# 11/02 ledger #164: the clean clone check of the A10 / B9s package on Linux, with time and peak memory. The repository comes from a plain clone on C: (the worktree's .git points at a Windows path).
export PY=$HOME/arc2venv/bin/python REPO=$HOME/tfb_repo_a10 ASSETS=/mnt/c/tfb_release/tfb_final_assets_a10.zip D=$HOME/tfb_clone_a10 TFB_REL=/mnt/c/Users/user/tfb/data/kaggle_public FINAL="${FINAL:-A10t}"
rm -rf $REPO; git clone -q /mnt/c/tfb_clone_a10/tfb $REPO
/usr/bin/time -v bash /mnt/c/Users/user/tfb_a8/repro/clone_check.sh > /mnt/c/tfb_data/clone_a10_wsl.log 2>&1
echo "exit $?" >> /mnt/c/tfb_data/clone_a10_wsl.log; rm -rf $HOME/tfb_clone_a10 $HOME/tfb_repo_a10
