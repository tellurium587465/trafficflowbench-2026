#!/bin/bash
# 10/23 audit day 1 (2): rebuild the final A and B from a FRESH copy of the code in another folder with only the shipped
# inputs (no access to the development work folder): caches from the release, the cache-free mixes, the finals by name;
# then compare the submission.csv with the submitted finA4 / finB4 byte for byte.
# Inputs copied: code (src, twin/build_march.py, repro); work/t1 state, work/t4 odme; work/t2: the final networks' saved
# probabilities and X1 probabilities, the LightGBM / physics model files, the rarity table; NN: the official windows'
# features, the feature names. The release (TFB_REL) is read in place (the organisers have it).
set -euo pipefail; export MSYS_NO_PATHCONV=1
SRCROOT=${SRCROOT:-/c/Users/user/tfb}; DEV=$SRCROOT/work; DEVNN=${DEVNN:-/c/tfb_data/tfb_nn}
C=${CLEAN:-/c/tfb_clean/tfb}; CW=$C/work; CNN=/c/tfb_clean/nn
rm -rf $C $CNN; mkdir -p $C/twin $CW/t1 $CW/t4 $CW/t2 $CW/subs $CNN/nn_data_afdem $CNN/nn_test_afdem
cp -r $SRCROOT/src $SRCROOT/repro $C/; cp $SRCROOT/twin/build_march.py $C/twin/
cp $DEV/t1/state_f3ens3c_ad2_gapml_clip_fd_bnd.parquet $CW/t1/; cp $DEV/t4/odme_l2.parquet $CW/t4/
for s in 0 1 2 3 4 5 6 7; do for c in attnglob attnwp05 attn1 drop1 r13; do
  for f in q24f e4f; do cp $DEV/t2/probs_${f}_${c}_s$s.pkl $DEV/t2/probs_x1off__${f}_${c}_s$s.pkl $CW/t2/; done
done; done
cp $DEV/t2/lgbnm_queue_onset.txt $DEV/t2/lgbnm_queue_ongoing.txt $DEV/t2/physon_base.txt $DEV/t2/physon_m.txt $DEV/t2/rarity_table.pkl $CW/t2/
cp $DEVNN/nn_data_afdem/names.npy $CNN/nn_data_afdem/; cp $DEVNN/nn_test_afdem/*.npy $CNN/nn_test_afdem/
echo "copied: $(du -sh $C | cut -f1) code+inputs, $(du -sh $CNN | cut -f1) NN"
export TFB_WORK=$(cygpath -m $CW) TFB_NN=$(cygpath -m $CNN) TFB_CACHE=$(cygpath -m $CW/cache) TFB_REL=${TFB_REL:-$(cygpath -m $SRCROOT/data/kaggle_public)}   # the release: in place
cd $C/src; t0=$(date +%s)
py -c "import build_cache
from cachelib import T2_PANELS
for p in T2_PANELS:
    for s in ('validation', 'private'):
        build_cache.build(p, s)" | tail -1
py t2_lgbnm_apply.py | tail -1
py t2_phys_onset.py apply | tail -1
py t2_phys_onset.py adaptive | tail -1
py t2_build_final.py A clean_finA | tail -1
py t2_build_final.py B clean_finB | tail -1
echo "clean rebuild: $(( $(date +%s) - t0 )) s"
py - <<'PY'
import zipfile, hashlib, os
W = os.environ["TFB_WORK"]; D = "C:/Users/user/tfb/work/subs/"
for a, b in (("clean_finA", "finA4"), ("clean_finB", "finB4")):
    h = [hashlib.sha1(zipfile.ZipFile(p).read("submission.csv")).hexdigest() for p in (f"{W}/subs/{a}_final.zip", f"{D}{b}_final.zip")]
    print(f"{a} submission.csv == submitted {b}: {h[0] == h[1]} ({h[0][:12]})")
PY
