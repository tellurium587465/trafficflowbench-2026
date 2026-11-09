#!/bin/bash
# 11/05 ledger #246: Task 2 of the named finals on a newly generated month in the release's file layout, from the public data alone
# (no shipped probabilities, X1 files, network inputs or fine-tuning pools). README "Running on a newly generated month".
# 11/06: the default finals are A25t / B7ct; their selection-conditioned ongoing step (src/t2_selcond.py) reads the new release's
# task2/<panel>/<split>/window_index.csv (TFB_REL) for the other official windows' origins <= T.
#   TFB_REL   = the new release (its private split = the new month; the validation split = the month before, used by the physics model
#               only through the shipped physon_m.txt, and by the March windows of the finals)
#   TFB_WORK, TFB_NN, TFB_CACHE = FRESH folders holding only:
#       NN/models/<the finals' networks>.pt and NN/nn_data_afdem/names.npy                       (release assets: nn/...)
#       WORK/t2/{lgbnm_queue_onset.txt, lgbnm_queue_ongoing.txt, physon_base.txt, physon_m.txt, rarity_table.pkl} (release assets: work/t2/...)
#       WORK/t1/state_t1final.parquet and WORK/t4/odme_l2.parquet when the zip is assembled (Tasks 1 and 4)
#   TFB_SPLIT_DATES = "private=YYYY-MM-01,YYYY-MM-01" when the new month's dates are not April 2031 (src/build_cache.py)
# CPU only. Measured on one panel (April as the new month, ledger #246): 7 min; a whole month is ~8x the pool, the same-day X1 of
# 100 networks being the long part (about 15-20 h on 16 threads as 4 queues x 4 threads; X1_THREADS=4 is part of the bit-for-bit result).
# usage: FINAL="A25t B7ct" PY=python3 bash repro/run_new_month.sh        (stages: STAGES="cache infer aux pools x1 build", default all)
set -euo pipefail; export PATH="/usr/bin:$PATH"
cd "$(dirname "${BASH_SOURCE[0]}")/../src"; PY=${PY:-python3}; FINAL=${FINAL:-"A25t B7ct"}; STAGES=${STAGES:-"cache infer aux pools x1 build"}
export TFB_INFER_IN=nn_test_afdem X1_THREADS=${X1_THREADS:-4}; W=${TFB_WORK:?set TFB_WORK to a fresh folder}/
has() { [[ " $STAGES " == *" $1 "* ]]; }; NAMES=""; for f in $FINAL; do NAMES="$NAMES ${f%t}"; done
C5="attnglob attnwp05 attn1 drop1 r13"; tags() { for s in "$@"; do for c in $C5; do printf "q24f_%s_s%s " $c $s; done; done; }
if has cache; then $PY -c "
import build_cache
from common import PANELS
for p in PANELS:
    for s in ('train', 'validation', 'private'):
        build_cache.build(p, s)
print('caches ready')"; fi
if has infer; then $PY t2_infer_official.py $NAMES | tail -2; fi                       # inputs (NN/nn_test_afdem, R, _pct) + every network part
if has aux; then
  $PY t2_lgbnm_apply.py | tail -1; $PY t2_phys_onset.py apply | tail -1; $PY t2_phys_onset.py adaptive | tail -3
fi
if has pools; then                                                                   # the months' fine-tuning windows, rebuilt from the release
  [ -f "$TFB_NN/nn_data_march/X.npy" ] || $PY ../twin/build_march.py 6 validation | tail -1
  [ -f "$TFB_NN/nn_data_april_sp3/X.npy" ] || TFB_MONTH_OUT=nn_data_april_sp3 $PY ../twin/build_march.py 3 private | tail -1
fi
if has x1; then                                                                      # same-day X1: March sd (s0-3), the new month sdd3 (s0-15)
  for t in $(tags 0 1 2 3); do X1_SPLITS=validation X1_ARM=sd $PY t2_x1_sameday.py $t | tail -1; done
  for t in $(tags 0 1 2 3); do X1_SPLITS=private X1_ARM=sdd3 X1_POOL_PRIV=nn_data_april_sp3 X1_FILL=sd $PY t2_x1_sameday.py $t | tail -1; done
  for t in $(tags 4 5 6 7 8 9 10 11 12 13 14 15); do X1_SPLITS=private X1_ARM=sdd3 X1_POOL_PRIV=nn_data_april_sp3 $PY t2_x1_sameday.py $t | tail -1; done
  X1_TTA=0 X1_SPLITS=validation $PY t2_x1_official.py $(tags 0 1 2 3 | tr ' ' ',' | sed 's/,$//') | tail -1    # B's March ongoing (plain X1)
fi
if has build; then
  for f in $NAMES; do                                                               # 11/06: a failure stops the run unless only the zip step failed
    rm -f "${W}t2/queue_newmonth_fin${f}.parquet"
    if ! $PY t2_build_final.py $f newmonth_fin${f} ${W}t1/state_t1final.parquet ${W}t4/odme_l2.parquet | tail -2; then
      [ -f "${W}t2/queue_newmonth_fin${f}.parquet" ] || { echo "$f: Task 2 build failed (no queue written)" >&2; exit 1; }
      echo "($f: the Task 2 queue is in ${W}t2/queue_newmonth_fin${f}.parquet; the zip step needs the new month's submission key)"
    fi
  done
fi
