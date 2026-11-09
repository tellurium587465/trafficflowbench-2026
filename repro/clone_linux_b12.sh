#!/bin/bash
# 11/04 ledger #221 / the strategy's step 5: the Linux (WSL) clone pass of the finals with B11 / B12 (README's distribution: git clone of
# the branch + the release assets tfb_final_assets_b12.zip + the public release, then run_final.sh: caches -> t2aux -> t2final -> verify).
# The worktree's .git points at a Windows path, so the repository comes from the branch bundle of the 11/04 backup (as recheck_a10_wsl.sh).
export PY=$HOME/arc2venv/bin/python REPO=$HOME/tfb_repo_b12 ASSETS=/mnt/c/tfb_release/tfb_final_assets_b12.zip D=$HOME/tfb_clone_b12 \
       TFB_REL=/mnt/c/Users/user/tfb/data/kaggle_public FINAL="A13t B12t B11t B10t A12t"
L=/mnt/c/Users/user/tfb_a8/repro/logs/clone_linux_b12.txt
rm -rf $REPO; git clone -q -b a8b9 /mnt/c/tfb_final_backup_1104/tfb_a8b9.bundle $REPO > $L 2>&1
/usr/bin/time -v bash /mnt/c/Users/user/tfb_a8/repro/clone_check.sh >> $L 2>&1
echo "EXIT $? $(date)" >> $L; rm -rf $HOME/tfb_clone_b12 $HOME/tfb_repo_b12
