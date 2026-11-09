#!/bin/bash
# Single entry point that rebuilds the final submissions from the released data and the saved models (10/13; 10/23 the
# final candidates A / B, built by name from src/t2_final_spec.FINAL; 10/24 the fast path is the default).
#
#   bash repro/run_final.sh                 # the selected finals A25t and B7ct: caches from the release -> t2aux -> t2final -> verify (~2 min, CPU)
#   FINAL=A25t bash repro/run_final.sh      # one final by name (or: bash repro/run_final.sh A25t); any key with a SHA[...] line below
#   STAGES="cache t1 t4 t2aux t2final verify" bash repro/run_final.sh     # Task 1 and Task 4 rebuilt too (~60 min)
#   stages: cache t1 t4 t2aux t2x1 t2final verify | verifylocal t2legacy (our machine: the submission log, s60 / s70)
#   PY=$PY on Windows (default python3); paths: TFB_REL (release), TFB_WORK, TFB_NN, TFB_CACHE (README.md)
#
# T2 of the finals: the saved networks' probabilities ($TFB_WORK/t2/probs_<tag>.pkl; q24f seeds 0-7 = onset, e4f = A's
# April base) and their X1 predictions are the start. From them, on the CPU and deterministic: t2aux = the no-memory
# LightGBM and the physics onset model applied from their saved model files + the month-adaptive a; t2x1 = X1 recomputed
# (each network fine-tuned on the month's completed earlier days, 6-8 h; X1_ARM=rep writes a separate copy); t2final = the
# named final (t2_build_final.py: blends -> mixes -> cone -> evidence rule -> far gate -> zip); verify = sha256 of the
# rebuilt submission.csv against the submitted file's.
# The one-window organisers' check (rows after T deleted): src/t2_trunc_physical.py <A|B> <queue> (README.md).
set -euo pipefail; export PATH="/usr/bin:$PATH"
ROOT=${TFB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && (pwd -W 2>/dev/null || pwd))}   # 10/24: the repository root from the script's place
W=${TFB_WORK:-$ROOT/work}/; NND=${TFB_NN:-C:/tfb_data/tfb_nn}; export TFB_RUN_W=$W; cd $ROOT/src
STAGES=${STAGES:-"cache t2aux t2final verify"}; FINAL=${FINAL:-${1:-"A25t B7ct"}}; PY=${PY:-python3}   # 11/06: the selected finals A25 / B7c (ledger #283); earlier finals by name
declare -A SHA=([A]=7f6d3e983ed69c429692f92650778a7c0f29a6058ea22486a8f581121dcce84e [B]=406a23b099d549b7c8d45f6de4ccd6a06ae5ba3fc0c46c0a1e67138630a62491)   # submitted finA5 / finB4
SHA[A6]=27bd4fafd262249b58a7e6485db61fee8df0d4aa4873b0e70576facc70a8230c; SHA[B5]=cfc9f6e88b01b4cf49cda1d356306a0cd8c4a89f6ea6921cf5288b9a3fa94126   # 10/29 candidates finA6 / finB5 (lane 9, ledger #56)
SHA[B6]=4648e7a72f209aabdfa6499da22983f4fb854a1ef30f223a7f9ca896b3ec2cd2   # 10/30 finB6 = finB5 with the April ongoing left raw (ledger #72, J8)
SHA[B7]=d0673807782917ad3c278f0747974cb8c03694b325b2511430c3aa319cfe11cf   # 10/31 finB7 = finB6 without X1 on the April ongoing windows (ledger #95-#97)
SHA[B8]=98b943ada7e459da38c1a9b202542f713cac76df2588a90a0d80a0b6c1816196   # 10/31 B8 = finB4 March + finB7 April, built here, submitted only if #63 keeps finA5
SHA[A7]=2022885978b48f1576055e85169781d5d6b37a177eecb94bb8c316e148729c2e   # 10/31 A7 = A6 without E4 and synthetic data, the alternative A (ledger #110)
SHA[A7t]=29a082b1e646367d2c96601c58e3e1ea6e35c76845b643eb106f8ac096fe2326   # 11/02 ledgers #118c, #132, #164: A7 on the final Task 1 (the image network, seeds 4, 5, month fine-tune)
SHA[A8t]=e34ee76a62820832ae09565d13d9efc0a33310f5234ab63ffc0993860699aa1a   # 11/02 ledgers #118c, #132, #164: A8 on the final Task 1 (the image network, seeds 4, 5, month fine-tune)
SHA[A10t]=f2ef607297b27e08867bdc7117c146ab6b50f9094b4ff75aee512f6b01e860f0   # 11/02 ledgers #118c, #132, #164: A10 on the final Task 1 (the image network, seeds 4, 5, month fine-tune)
SHA[B7t]=fc7f8547c3c99d2411f4dc18255f797c2cc54468a50875f78fba80c708f245fa   # 11/02 ledgers #118c, #132, #164: B7 on the final Task 1 (the image network, seeds 4, 5, month fine-tune)
SHA[B9st]=e41310c14e33803407da3806cda58f0e6ba3138829150a08480af8deebed8d0b   # 11/02 ledgers #118c, #132, #164: B9s on the final Task 1 (the image network, seeds 4, 5, month fine-tune)
SHA[A11t]=14b5d8d8fb5af0b830aceb470c7dd2e27694d85e5e90c6b79b60a0e76f6482b6   # 11/03 ledgers #197, #201: A8 + the row at T as the last queue + the 2D U-Net 0.2 (ongoing)
SHA[A12t]=342515f1ea79aa34068b0a3110a2081103b7fe00e5772af58eaf6dd39e077815   # 11/03 ledger #201: A11 + the rank-input network 0.2 (onset, #17)
SHA[B10t]=37bbd8dc273a7266d1f0a794bdcb804057627fc1302100c39b1dde8e5943a90a   # 11/03 ledger #201: B9s + the 2D U-Net 0.2 (ongoing)
SHA[A13t]=16bcab21c2fa9ae901c211125ca06a46c1865dea1587f52f9608fd7114cfd4a9   # 11/03 ledger #189: A12 with April's same-day X1 on origins every 3 rows
SHA[B11t]=9259eae7e78696f89d6e2637618c551572152c4f14094f4629ff173f978d48eb   # 11/04 ledgers #107, #179 6: B10 with April's ongoing over seeds 0-15 (not submitted: March as B10)
SHA[B12t]=06ca82bdf5946970b992f12f8bf28af9bc93c29f7aea76bdb3931bfe0cfcc006   # 11/04 ledgers #216, #179 6: B11 with the ongoing decode by the exact expected IoU (TFB_T2_OGDEX)
SHA[A15t]=69c19f33b43d2bfdfd2c1fdf5a08265d46ed716adcc4b512c4340770352a8657   # 11/05 ledgers #230, #231: A13 with April's ongoing over seeds 0-15 (dense same-day X1 sdd3 for seeds 8-15)
SHA[A16t]=7bac846217800fa7b3d08776f3c121c12735fcaf004a10aede3e535739a1c178   # 11/05 ledger #232: A15 with the rank-input onset network over seeds 0-3 (weight 0.2)
SHA[A17t]=c4e11f018904986d3ddae767f76d93867e95ce189f4730c8876e48ba748e2d73   # 11/05 ledger #234: A16 with the ongoing temperature 0.8 (cell log loss)
SHA[A21t]=c7847e167af61348986d7f7ecfd1fccb757f7aa9bb9f36c6b23b5642cb0bab0e   # 11/05 ledger #245: A17 without the 2D U-Net in the ongoing mix
SHA[A25t]=139592a38586a256336bb11243dd6d91d960d374e6da87732b6169b74bbff01d   # 11/05 ledger #283 / #286: A21 + the selection-conditioned ongoing (t2_selcond V11R)
SHA[B7ct]=eac4a4c8df3796274574db5a4c6c9dd67fcd6ab54fdfcf2dd45e074d6eb1ab45   # 11/05 ledger #283 / #286: B7 + the selection-conditioned ongoing (t2_selcond V11R)
SHA[A22t]=c7501d64537348cd7459d06088771b7b440348576655206276b70c10aa5fb9bc   # 11/06 ledger #252: A21 with the April windows from q24m (training months + March) x 8 seeds
SHA[A22ct]=2a2bd56c4190fc7b0f11fa2eef7011cce23598a55856ba191dc418a6a3942841   # 11/06 ledger #252 / #283: A22 + the selection-conditioned ongoing
has() { [[ " $STAGES " == *" $1 "* ]]; }
T1=${T1_STATE:-${W}t1/state_t1final.parquet}; T4=${W}t4/odme_l2.parquet   # 11/02: the final Task 1; T1_STATE=${W}t1/state_f3ens3c_ad2_gapml_clip_fd_bnd.parquet gives the chain without the image network (the keys without n)
R1=${W}t1/repro_state; RT2=repro

if has cache; then                                 # per-panel caches from the release (Task 2 panels; every panel when t1 runs)
  $PY -c "
import build_cache
from cachelib import CACHE, T2_PANELS
from common import PANELS
ps = PANELS if '$STAGES'.split().count('t1') else T2_PANELS
for p in ps:
    for s in (build_cache.SPLITS if '$STAGES'.split().count('t1') else ('validation', 'private')):
        if not (CACHE / f'{p}_{s}.npz').exists():
            build_cache.build(p, s)
print('caches ready:', CACHE)"
fi
if has t1; then                                    # Task 1: three LightGBM pairs with ADAPT, ensemble, gap model, clip, FD, boundary
  for m in f3big f3all f3all2; do TFB_T1_ADAPT=1 TFB_T1_FEAT_TRAINSTAT=1 TFB_T1_OUTSUF=_ad2 $PY t1_infer2.py 3 $m | tail -1; done
  $PY t1_ens.py _ad2 ${R1}_ad2.parquet | tail -1
  $PY t1_gapml_apply.py 3000 1900 0.1 ${R1}_ad2.parquet ${R1}_ad2_gapml.parquet | tail -1
  $PY -W ignore t1_clip.py ${R1}_ad2_gapml.parquet ${R1}_ad2_gapml_clip.parquet | tail -1
  $PY -W ignore t1_fd.py ${R1}_ad2_gapml_clip.parquet ${R1}_ad2_gapml_clip_fd.parquet | tail -1
  $PY -W ignore t1_bnd_apply.py bnd3 0.75 ${R1}_ad2_gapml_clip_fd.parquet ${R1}_ad2_gapml_clip_fd_bnd.parquet | tail -1
  $PY -W ignore t1_unet_apply.py 0.7 0.1 ${R1}_ad2_gapml_clip_fd_bnd.parquet ${R1}_final.parquet | tail -1   # the image network, needs work/t1/unet_ft (t1_unet_train.py, t1_unet_ft.py on a GPU)
  T1=${R1}_final.parquet
fi
if has t4; then                                    # Task 4: L2 projection of the weak prior onto {A f = c, f >= 0}
  $PY t4_candidates.py | tail -1                    # writes work/t4/odme_*.parquet (odme_l2 is the submitted one)
fi
if has t2legacy; then                              # (10/13) s60 / s70: blends -> onset/ongoing hybrid -> same-month evidence -> far gate
  SPEC46=$($PY -c "import json; print(json.load(open('t2_specs.json'))['s37']['spec'])")
  SPEC70=""; for s in 0 1 2 3; do SPEC70="$SPEC70+q43f_attnglob_s$s:2+q43f_attnwp05_s$s:2+q43f_attn1_s$s:2+q43f_drop1_s$s+q43f_r13_s$s+q46f_attnglob_s$s:2+q46f_attnwp05_s$s:2+q46f_attn1_s$s:2+q46f_drop1_s$s+q46f_r13_s$s"; done; SPEC70=${SPEC70#+}
  $PY t2_blend_submit.py "$SPEC46" ${RT2}_s46raw $T1 $T4 | tail -1
  $PY t2_blend_submit.py "$SPEC70" ${RT2}_s70sim $T1 $T4 | tail -1
  $PY - <<'PY'
import pickle, pandas as pd
from common import REL
from cachelib import T2_PANELS
import os
W = os.environ["TFB_RUN_W"] + "t2/"
a = pickle.load(open(W + "probs_repro_s46raw.pkl", "rb")); b = pickle.load(open(W + "probs_repro_s70sim.pkl", "rb"))
cond = {}
for sp in ("validation", "private"):
    for p in T2_PANELS:
        w = pd.read_csv(REL / "task2" / p / sp / "window_index.csv"); cond.update(dict(zip(w.window_id, w.condition)))
if not (set(a) == set(b) == set(cond)):
    raise SystemExit("window sets differ between the blends and the window index")
pickle.dump({k: (b[k] if cond[k] == "queue_ongoing" else a[k]) for k in a}, open(W + "probs_repro_s70hyb.pkl", "wb"))
print("hybrid written: onset from the s46 blend, ongoing from the R24+R48 blend")
PY
  $PY t2_onset_evidence.py ${RT2}_s46raw ${RT2}_s60on 0.25 0.85 | tail -1
  $PY t2_far_gate.py ${W}t2/queue_${RT2}_s60on.parquet ${W}t2/queue_${RT2}_s60.parquet 3 1 | tail -1
  $PY t2_onset_evidence.py ${RT2}_s70hyb ${RT2}_s70on 0.25 0.85 | tail -1
  $PY t2_far_gate.py ${W}t2/queue_${RT2}_s70on.parquet ${W}t2/queue_${RT2}_s70.parquet 3 1 | tail -1
  $PY -c "
import pandas as pd
from submit import assemble
W='$W'
T1=pd.read_parquet('$T1'); T4=pd.read_parquet('$T4')
for k in ('s60', 's70'):
    print(assemble(T1, pd.read_parquet(W + 't2/queue_repro_' + k + '.parquet'), T4, 'repro_' + k + '_final'))"
fi
if has t2aux; then                                 # T2 mixes from their saved model files (CPU, deterministic)
  $PY t2_lgbnm_apply.py | tail -1                   # work/t2/probs_lgbnm.pkl (saved model files; no 3 GB training cache)
  $PY t2_phys_onset.py apply | tail -1              # work/t2/probs_physon.pkl (March windows: train months; April: + March)
  $PY t2_phys_onset.py adaptive | tail -3           # work/t2/physon_a.pkl (each window's a from its month's completed earlier days)
fi
if has t2x1; then                                  # X1 of the finals' networks (t2_x1_official.py, CPU, seed 0, no TTA draws; ~6 h)
  [ -f "$NND/nn_data_march/X.npy" ] || py ../twin/build_march.py 6 validation | tail -1   # the months' fine-tuning windows
  [ -f "$NND/nn_data_april/X.npy" ] || py ../twin/build_march.py 6 private | tail -1      # (not shipped: rebuilt from the release)
  Q03=$($PY -c "print(','.join(f'q24f_{c}_s{s}' for s in range(4) for c in ('attnglob','attnwp05','attn1','drop1','r13')))")
  Q47=$($PY -c "print(','.join(f'q24f_{c}_s{s}' for s in range(4, 8) for c in ('attnglob','attnwp05','attn1','drop1','r13')))")
  E011=$($PY -c "print(','.join(f'e4f_{c}_s{s}' for s in range(12) for c in ('attnglob','attnwp05','attn1','drop1','r13')))")
  X1_TTA=0 $PY t2_x1_official.py $Q03 ${X1_LR:-2e-4} ${X1_ARM:-} | tail -1                    # both months (A / B March, B April)
  X1_TTA=0 X1_SPLITS=private $PY t2_x1_official.py $Q47 ${X1_LR:-2e-4} ${X1_ARM:-} | tail -1  # B April (seeds 4-7)
  X1_TTA=0 X1_SPLITS=private $PY t2_x1_official.py $E011 ${X1_LR:-2e-4} ${X1_ARM:-} | tail -1  # A April (E4, 12 seeds)
fi
if has t2x1sd; then                                # 11/02 ledgers #125, #164: same-day X1 of A8 and A10 (t2_x1_sameday.py, CPU, deterministic for X1_THREADS=4; about 15 CPU hours per family)
  [ -f "$NND/nn_data_march/X.npy" ] || $PY ../twin/build_march.py 6 validation | tail -1
  [ -f "$NND/nn_data_april/X.npy" ] || $PY ../twin/build_march.py 6 private | tail -1
  [ -f "$NND/nn_data_marcht/X.npy" ] || $PY t2_trow.py validation nn_data_march nn_data_marcht
  [ -f "$NND/nn_data_aprilt/X.npy" ] || $PY t2_trow.py private nn_data_april nn_data_aprilt
  for fam in q24t q24f; do
    Q03=$($PY -c "print(','.join(f'${fam}_{c}_s{s}' for s in range(4) for c in ('attnglob','attnwp05','attn1','drop1','r13')))")
    Q47=$($PY -c "print(','.join(f'${fam}_{c}_s{s}' for s in range(4, 8) for c in ('attnglob','attnwp05','attn1','drop1','r13')))")
    X1_THREADS=4 X1_SPLITS=validation,private $PY t2_x1_sameday.py $Q03 | tail -1
    X1_THREADS=4 X1_SPLITS=private $PY t2_x1_sameday.py $Q47 | tail -1
  done
  [ -f "$NND/nn_data_april_sp3/X.npy" ] || TFB_MONTH_OUT=nn_data_april_sp3 $PY ../twin/build_march.py 3 private | tail -1   # 11/03 ledger #189 (A13): April's pool on origins every 3 rows
  Q07=$($PY -c "print(','.join(f'q24f_{c}_s{s}' for s in range(8) for c in ('attnglob','attnwp05','attn1','drop1','r13')))")
  X1_THREADS=4 X1_SPLITS=private X1_ARM=sdd3 X1_POOL_PRIV=nn_data_april_sp3 X1_FILL=sd $PY t2_x1_sameday.py $Q07 | tail -1   # March copied from the sd files
fi
if has t2final; then                               # the final candidates by name (src/t2_final_spec.FINAL)
  for f in $FINAL; do c=${f%t}; $PY t2_build_final.py $c repro_fin$f $T1 $T4 | tail -1; done   # A10t = A10 on the final Task 1
fi
if has verify; then                                # the rebuilt submission.csv against the submitted file (sha256)
  for f in $FINAL; do
    h=$($PY -c "import zipfile, hashlib; print(hashlib.sha256(zipfile.ZipFile('${W}subs/repro_fin${f}_final.zip').read('submission.csv')).hexdigest())")
    if [ "$h" = "${SHA[$f]}" ]; then echo "final $f: submission.csv identical to the submitted one (sha256 ${h:0:16})"; else echo "final $f: DIFFERS (sha256 ${h:0:16}, submitted ${SHA[$f]:0:16})"; exit 1; fi
  done
fi
if has verifylocal; then                           # (our machine) component-by-component comparison with the submission log
  $PY - <<'PY'
import json, hashlib, os, pandas as pd
W = os.environ["TFB_RUN_W"]
rec = {json.loads(l)["zip"][:3]: json.loads(l) for l in open(W + "subs/SUBMISSIONS.jsonl", encoding="utf-8")}
import os
REF = {"finA": os.environ.get("FINAL_A_REF", "finA"), "finB": os.environ.get("FINAL_B_REF", "finB")}   # the submitted finals' build names
pairs = {k: W + f"t2/queue_repro_{k}.parquet" for k in REF if os.path.exists(W + f"t2/queue_repro_{k}.parquet")}
pairs.update({k: W + f"t2/queue_repro_{k}.parquet" for k in ("s60", "s70") if os.path.exists(W + f"t2/queue_repro_{k}.parquet")})
rt1 = W + "t1/repro_state_ad2_gapml_clip_fd_bnd.parquet"
ok = True
for k, new in pairs.items():
    old = rec[k]["components"]["t2_queue"]["path"] if k in rec else W + f"t2/queue_{REF.get(k, k)}.parquet"
    same = pd.read_parquet(new).equals(pd.read_parquet(old)); ok &= same
    print(f"{k}: T2 queue identical to the submitted one: {same}")
if os.path.exists(rt1):                                    # T1 rebuilt in this run: compare with the submitted T1
    same = pd.read_parquet(rt1).equals(pd.read_parquet(rec["s70"]["components"]["t1_state"]["path"])); ok &= same
    print(f"T1 rebuilt identical to the submitted one: {same}")
for comp in ("t1_state", "t4_odme"):
    path = rec["s70"]["components"][comp]["path"]
    h = hashlib.sha1(open(path, "rb").read()).hexdigest()
    same = h == rec["s70"]["components"][comp]["sha1"]; ok &= same
    print(f"{comp}: file unchanged since the s70 submission: {same}")
raise SystemExit(0 if ok else 1)
PY
fi
echo "run_final: done ($STAGES)"
