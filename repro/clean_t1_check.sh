#!/bin/bash
# 10/23 audit day 1 (2), Task 1: the run_final.sh t1 stage in the clean copy (repro/clean_folder_check.sh made it) with only
# the shipped T1 inputs (the three LightGBM pairs, the boundary models, the gap model's window list featcache2_noT.npz;
# the gap model itself is retrained in the stage, deterministic) -> compare with the submitted T1 state.
set -euo pipefail; export MSYS_NO_PATHCONV=1
SRCROOT=/c/Users/user/tfb; DEV=$SRCROOT/work; C=/c/tfb_clean/tfb; CW=$C/work
mkdir -p $CW/t1 $CW/t2
for m in f3big f3all f3all2; do cp $DEV/t1/speed_$m.txt $DEV/t1/flow_$m.txt $CW/t1/; done
cp $DEV/t1/bnd3_speed_final.txt $DEV/t1/bnd3_flow_final.txt $CW/t1/; cp $DEV/t2/gapml_windows.npz $CW/t2/
export TFB_WORK=$(cygpath -m $CW) TFB_NN=$(cygpath -m /c/tfb_clean/nn) TFB_CACHE=$(cygpath -m $CW/cache) TFB_REL=$(cygpath -m $SRCROOT/data/kaggle_public)
cd $C/src; t0=$(date +%s); R1=$TFB_WORK/t1/repro_state
py build_cache.py | tail -1                                   # every panel (Task 1 covers more corridors than Task 2) x split
for m in f3big f3all f3all2; do TFB_T1_ADAPT=1 TFB_T1_FEAT_TRAINSTAT=1 TFB_T1_OUTSUF=_ad2 py t1_infer2.py 3 $m | tail -1; done
py t1_ens.py _ad2 ${R1}_ad2.parquet | tail -1
py t1_gapml_apply.py 3000 1900 0.1 ${R1}_ad2.parquet ${R1}_ad2_gapml.parquet | tail -1
py -W ignore t1_clip.py ${R1}_ad2_gapml.parquet ${R1}_ad2_gapml_clip.parquet | tail -1
py -W ignore t1_fd.py ${R1}_ad2_gapml_clip.parquet ${R1}_ad2_gapml_clip_fd.parquet | tail -1
py -W ignore t1_bnd_apply.py bnd3 0.75 ${R1}_ad2_gapml_clip_fd.parquet ${R1}_ad2_gapml_clip_fd_bnd.parquet | tail -1
echo "clean T1: $(( $(date +%s) - t0 )) s"
py -c "
import pandas as pd
a = pd.read_parquet('${R1}_ad2_gapml_clip_fd_bnd.parquet'); b = pd.read_parquet('$(cygpath -m $DEV)/t1/state_f3ens3c_ad2_gapml_clip_fd_bnd.parquet')
print('clean T1 == submitted T1:', a.equals(b), len(a))"
