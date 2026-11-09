#!/bin/bash
# 10/22: rule-clean synthetic worlds for the ongoing retrain (749601: no test-month statistic in the method).
# C = gen_clean (C-world randomization, link cells), V = gen_v4 (block cells, block-site thresholds, branch recording).
# Each step is skipped when its output exists (resumable after a reboot). CPU only; no GPU while this writes.
set -uo pipefail; export PATH="/usr/bin:$PATH"; cd /c/Users/user/tfb/twin
NN=C:/tfb_data/tfb_nn; C=/c/Users/user/tfb/work/cache
step() { echo "$(date '+%m-%d %H:%M:%S') $*"; }
[ -f $C/D12_I5_S_twinC23.npz ] || { step "gen_clean ALL 24"; py gen_clean.py ALL 24 31; }
[ -f $C/D12_I5_S_twinV23.npz ] || { step "gen_v4 ALL 24"; py gen_v4.py ALL 24 41; }
for fam in C V; do
  [ -f /c/tfb_data/tfb_nn/nn_data_twin${fam}24/Y.npy ] || { step "build_nn2 $fam"; BASE_PROFILE_ONLY=1 py build_nn2.py A6 $fam 24 - 0 twin${fam}24; }
  [ -f /c/tfb_data/tfb_nn/nn_data_afdemsim${fam}24/names.npy ] || { step "build_simR $fam"; SIM_S=nn_data_twin${fam}24 SIM_D=nn_data_afdemsim${fam}24 SIM_COND=queue_ongoing SIM_FOLD_TEMPLATE=1 SIM_FAMILY=$fam py build_simR.py; }
done
step "CHAIN_DONE"
