#!/bin/bash
# T1 without location/time memory (CAUSAL_MAP P1/P7): 3 models, ADAPT inference, same post chain as s30.
export PATH="/usr/bin:$PATH"; cd /c/Users/user/tfb/src; W=C:/Users/user/tfb/work/t1/; L=E:/tfb_nn/t1nm.log
export TFB_T1_NOMEM=1 PYTHONIOENCODING=utf-8
py guard_mem.py 9000 || { echo "guard failed" >> $L; exit 1; }
for spec in "f3nm_a 1 0.8" "f3nm_b 7 0.6" "f3nm_c 3 0.8"; do set -- $spec
  [ -f ${W}speed_$1.txt ] && [ -f ${W}flow_$1.txt ] || py -W ignore t1_full.py $1 2000000 6500 3600 $2 $3 >> $L 2>&1 || { echo "FAILED train $1" >> $L; exit 1; }
  TFB_T1_ADAPT=1 TFB_T1_OUTSUF=_ad2nm py -W ignore t1_infer2.py 3 $1 >> $L 2>&1 || { echo "FAILED infer $1" >> $L; exit 1; }
done
TFB_T1_ENS=f3nm_a,f3nm_b,f3nm_c py t1_ens.py _ad2nm ${W}state_f3nm_ad2nm.parquet >> $L 2>&1 || { echo "FAILED ens" >> $L; exit 1; }
unset TFB_T1_NOMEM
py -W ignore t1_gapml_apply.py 3000 1900 0.1 ${W}state_f3nm_ad2nm.parquet ${W}state_f3nm_ad2nm_gapml.parquet >> $L 2>&1 || { echo "FAILED gapml" >> $L; exit 1; }
py t1_clip.py ${W}state_f3nm_ad2nm_gapml.parquet ${W}state_f3nm_ad2nm_gapml_clip.parquet >> $L 2>&1
py t1_fd.py ${W}state_f3nm_ad2nm_gapml_clip.parquet ${W}state_f3nm_ad2nm_gapml_clip_fd.parquet 0.75 >> $L 2>&1
py -W ignore t1_bnd_apply.py bnd3 0.75 ${W}state_f3nm_ad2nm_gapml_clip_fd.parquet ${W}state_f3nm_ad2nm_gapml_clip_fd_bnd.parquet >> $L 2>&1 || { echo "FAILED bnd" >> $L; exit 1; }
echo "T1NM DONE" >> $L
