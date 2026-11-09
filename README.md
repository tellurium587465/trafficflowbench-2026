# TrafficFlowBench 2026 — team tellurium_rrr

Code for our solution to the IEEE BigData Cup 2026 TrafficFlowBench (Kaggle `2026-ieee-big-data-traffic-flow-bench`).
Two final submissions were selected (6 November 2026; submissions 58038628 and 58038652):

| Name | File | Public LB (validation month; the private month decides the ranking) | What differs |
|---|---|---|---|
| A | `finA25.zip` (A25t) | 0.90279 | Task 2 networks trained on the training months, X1 fine-tuned on the same day's earlier windows (April: origins every 3 rows), an attention backbone, raw-history networks, a rank-input onset network, a no-memory LightGBM, the row at T as the last queue, the reachability cone, the far gate, and the ongoing probabilities conditioned on the official window selection (`src/t2_selcond.py`); the final Task 1 (image network fine-tuned on the month) |
| B | `finB7c.zip` (B7ct) | 0.90184 | The same Task 1 and Task 4; Task 2 without X1 or post-processing on April's ongoing windows (raw expected-IoU sets of the training-month networks), with the same selection conditioning |

Alternatives built and verified the same way: `finA22c.zip` (A22ct, public 0.90279: A25 with April's networks retrained on the training months
plus all of March, 8 seeds; a new month needs that retraining too, which `repro/run_new_month.sh` does not do) and the earlier
`finA21.zip` / `finB7n.zip` (0.89963 / 0.89966: without the selection conditioning).

A and B share Task 1, Task 3 and Task 4; they differ in Task 2: in March in 116 cells of 26 windows (7 cells in 3 onset windows), in April in 250 cells of 38 windows. Whichever of the two is scored, one command below rebuilds it, and its
`submission.csv` can be compared byte for byte with the submitted one (sha256 at the end of this page).

## Layout

| Path | Contents |
|---|---|
| `src/` | all task code; the final Task 2 composition is `src/t2_final_spec.py` (`FINAL["A25"]`, `FINAL["B7c"]`; the alternatives `FINAL["A22c"]`, `FINAL["A21"]`, `FINAL["B7"]`) |
| `repro/` | entry script `run_final.sh`, reproduction checks, frozen requirements, network configs (`nn_configs/`), logs |
| `twin/` | the synthetic-month generator and the month data builder used by X1 (`build_march.py`) |
| `review/`, `audit/` | the experiments and audits behind each decision (one script per entry of the decision ledger) |
| `LINEAGE.md`, `repro/README.md` | provenance of every component and the dated reproduction log (English; Japanese originals `LINEAGE_ja.md`, `repro/README_ja.md`) |

Not in git: the public release (download it from Kaggle) and the assets, two archives
`tfb_final_assets_a22_nn.zip` (1.0 GiB: `nn/`) and `tfb_final_assets_a22_work.zip` (1.3 GiB: `work/`), both unpacked into the same folder. The two archives and `MANIFEST.json` are handed to the organizers through the link in the final-package email (a shared folder, *[link given in the email]*); they are not on GitHub. `MANIFEST.json` is also in this repository, so the unpacked files can be checked against it (sha256 of the archives: `tfb_final_assets_a22_nn.zip` 47fd7758d0598b283522ec982d8497203085fab0be6a1d10654e5520be73c407, `tfb_final_assets_a22_work.zip` f006731448af43c8321a573d65717f4c75b3a567099b2afc6d1bd23365fe52ae). The archives hold the trained networks (q24f 16 seeds, q24m 8 seeds, e4f 12 seeds and the auxiliary networks, 5 configurations each), their saved test probabilities and X1 predictions, the
LightGBM / ranker model files, the Task 1 models, two small training-month artefacts and the submitted T1 / T4 components,
with a `MANIFEST.json` in each archive (size and sha256 of every file of both; the same file as `MANIFEST.json` here).

## Setup

- Inference and X1 (the production path): CPU only, developed on Windows 11 with Python 3.14.4. Use Python 3.12, 3.13 or
  3.14 (the pinned wheels do not exist for 3.10, 3.11 or 3.15) and install `pip install -r repro/requirements_repro.txt`
  (numpy, pandas, pyarrow, LightGBM, SciPy, scikit-learn at the versions the finals were built with). PyTorch is needed only
  for the optional X1 stage (CPU build). The full environments are frozen in `repro/requirements_windows_freeze.txt`
  (Python 3.14) and `repro/requirements_linux_py312.txt` (Python 3.12). The default path was rebuilt byte-identically on
  Linux (WSL2 Ubuntu 24.04, Python 3.12.3) from a new virtual environment with these six packages only.
- Network training only: WSL2 Ubuntu 24.04, PyTorch 2.10.0 + ROCm 7.2.1 — `repro/requirements_wsl_rocm_freeze.txt`.
- Always set `PY` to the interpreter of the environment: `PY=$VIRTUAL_ENV/bin/python` on Linux,
  `PY=$VIRTUAL_ENV/Scripts/python.exe` in Git Bash on Windows. `repro/run_final.sh` puts `/usr/bin` first on `PATH`, so
  activating a virtual environment alone is not enough. Python code starts its own subprocesses with the same interpreter.
- Data: TrafficFlowBench Release 1.0, the competition's Data page as re-issued on 2026-09-12 (check: sha256 of
  `submission_key.csv` = `8232c9ce6682fdd3120e6729a30e4514866ab492ef7b167ad3952e00478bd071`).

```bash
git clone <this repository> tfb && cd tfb
# 1. the public release, unpacked so that data/kaggle_public/config/corridors.json exists (or set TFB_REL to its folder)
# 2. the assets (the two archives from the organizers' link)
python -c "import zipfile; [zipfile.ZipFile(f).extractall('assets') for f in ('tfb_final_assets_a22_nn.zip', 'tfb_final_assets_a22_work.zip')]"
export TFB_NN=$PWD/assets/nn TFB_WORK=$PWD/assets/work TFB_CACHE=$PWD/assets/work/cache
# Git Bash on Windows: the paths must be in the C:/... form
# export TFB_NN=$(pwd -W)/assets/nn TFB_WORK=$(pwd -W)/assets/work TFB_CACHE=$(pwd -W)/assets/work/cache
```

`TFB_REL` (release, default `data/kaggle_public`), `TFB_WORK` (work files and outputs), `TFB_NN` (networks and window
features) and `TFB_CACHE` (per-panel caches built from the release) are the only paths the code reads.

## Reproduce the selected final

```bash
PY=$VIRTUAL_ENV/bin/python FINAL="A25t B7ct" bash repro/run_final.sh      # the selected finals A and B
PY=$VIRTUAL_ENV/bin/python FINAL=A22ct bash repro/run_final.sh            # any other built final by name (SHA[...] in the script)
```

The single entry script builds the per-panel caches from the release, applies the saved LightGBM and ranker models,
builds each named final into `$TFB_WORK/subs/repro_fin<name>_final.zip` and checks its
`submission.csv` against the submitted file's sha256 (`final A25t: submission.csv identical to the submitted one`). About
2 minutes on a CPU from a fresh clone. Task 1 and Task 4 come from the shipped components; to rebuild them too:
`STAGES="cache t1 t4 t2aux t2final verify" bash repro/run_final.sh` (≈ 80 min on 8 cores, 13 CPU hours).

**What is byte-identical.** The default path is the official path for review. It gave the same sha256 under all 17 conditions we tried: Linux and Windows,
hash seed, time zone, locale, 1 to 4 cores, the numpy SIMD level and the BLAS kernel. The full rebuild is byte-identical
only on our Windows machine. Elsewhere it stops with `DIFFERS` although nothing is wrong: `scipy`'s L-BFGS-B stops at
a slightly different point (Task 4, 966 lines differ in the sixth digit, relative L1 change 1e-8 on Linux), and on a CPU
without AVX-512 the BLAS `sgemm` rounding reaches the Task 1 features (emulated with `OPENBLAS_CORETYPE=Haswell`: RMS
4.0e-4 km/h in speed, 0.029 veh/h in flow). In that case compare the rebuilt Task 1 and Task 4 numerically with the shipped
files. Keep `num_threads=8` in `src/t1_gapml.py`: the core count and `OMP_NUM_THREADS` do not change the blank-cell model,
but changing that value does.

What the default stages run:

```bash
cd src
python -c "import build_cache; from cachelib import T2_PANELS; [build_cache.build(p, s) for p in T2_PANELS for s in ('validation', 'private')]"
python t2_lgbnm_apply.py && python t2_phys_onset.py apply && python t2_phys_onset.py adaptive
python t2_build_final.py A25 repro_finA25t $TFB_WORK/t1/state_t1final.parquet $TFB_WORK/t4/odme_l2.parquet   # -> $TFB_WORK/subs/repro_finA25t_final.zip (B: B7c repro_finB7ct ...)
```


## Verification

- **The organisers' causality check** (every row stamped after a window's origin removed, predictions unchanged):
  `cd src && python t2_trunc_physical.py A25 $TFB_WORK/t2/queue_repro_finA25t.parquet` (and `B7c $TFB_WORK/t2/queue_repro_finB7ct.parquet`) writes, per window, a copy of the release physically cut
  at T and rebuilds that window from it (phase 1: all 160 windows; `--cpu 4`: every part recomputed on the CPU; `--only 4
  <window_id>`: X1 fine-tuned inside the cut copy). It needs the training-month caches, which the default path does not build:
  `python -c "import build_cache; from cachelib import T2_PANELS; [build_cache.build(p, 'train') for p in T2_PANELS]"` (≈ 4 min);
  phases 2-4 also need the months' fine-tuning windows (`python ../twin/build_march.py 6 validation`, the same with `private`). Set
  `TFB_E6_ROOT` to a folder with room for the cut copies. Our runs passed every phase: `repro/logs/trunc_finA25.txt`, `trunc_finA25_p2.txt`, `trunc_finA25_x1.txt` (A25, phases 1-4),
  `trunc_finB7c.txt` (B7c, phases 1-3), `trunc_finA22*.txt` (the alternative A22c). The cut copy keeps the rows of
  `task2/<panel>/<split>/window_index.csv` whose origin is at or before T, the window's own row included (`t2_trunc_physical.py` does this):
  the selection-conditioned step (`TFB_T2_SELC=1`, `src/t2_selcond.py`: A25, A22c, B7c) reads the window's own origin and the latest
  official origin before it, whose origin + 72 rows bounds the earlier origins it examines; rows after T are not read (organisers' reply 745700).
  It passed on all 160 windows, including the 31 ongoing windows less than 78 rows after the previous official window.
- **Checks we ran** (details in `repro/README.md`): rebuild in a separate folder from the shipped inputs only
  (`repro/clean_folder_check.sh`, `clean_t1_check.sh`: T1, T2 and T4 identical to the submitted ones); rebuild from a fresh
  clone plus the asset archive through the entry script (`repro/clone_check.sh`: both `submission.csv` byte-identical on
  Windows / Python 3.14 and on Linux / Python 3.12); X1 recomputed for all 100
  networks under a separate name (`repro/x1rep_check.sh`: 16,000 window predictions bit-identical).

## Running on a newly generated month

The default path above starts from the network probabilities of the March / April windows shipped in the asset archive. For a new month
in the same file layout, `repro/run_new_month.sh` rebuilds Task 2 of the named finals from the public data alone (CPU only):

1. Put the new release at `TFB_REL` (the new month as its `private` split). If its dates are not April 2031, set
   `TFB_SPLIT_DATES="private=YYYY-MM-01,YYYY-MM-01"` (read by `src/build_cache.py`; unset = the dates of this competition).
2. Point `TFB_WORK`, `TFB_NN` and `TFB_CACHE` at fresh folders and copy into them only the trained artefacts from the asset archive:
   `nn/models/<the finals' networks>.pt`, `nn/nn_data_afdem/names.npy`, `work/t2/lgbnm_queue_{onset,ongoing}.txt`,
   `work/t2/physon_base.txt`, `work/t2/physon_m.txt`, `work/t2/rarity_table.pkl` (and `work/t1/state_t1final.parquet`,
   `work/t4/odme_l2.parquet` for the zip).
3. `FINAL="A25t B7ct" bash repro/run_new_month.sh` (the default). Stages: caches of every split → `src/t2_infer_official.py` (the official windows'
   inputs with their raw history and rank inputs, then every network part on the CPU: q24f and pcL with `t2_nn_cpu_infer.py`, x3af and
   tfm4 with `t2_cpu_infer_x3_tfm.py`, the 2D U-Net with `t2_cpu_infer_u2d.py`) → LightGBM and the physics onset model → the months'
   fine-tuning pools (`twin/build_march.py`) → same-day X1 (March: sd arm, seeds 0-3; the new month: sdd3 arm, seeds 0-15; B's March:
   plain X1) → the finals' blends, cone, evidence rule and far gate (`t2_build_final.py`). The Task 2 queue is written to
   `$TFB_WORK/t2/queue_newmonth_fin<name>.parquet`; the zip assembly (`src/submit.py`) checks the row count against this competition's
   submission key and needs the new month's key.

Finals with the selection-conditioned ongoing step (`TFB_T2_SELC=1`, `src/t2_selcond.py`: A25, A22c, B7c) also read each split's
`task2/<panel>/<split>/window_index.csv`: keep every row whose origin is at or before the window's T, its own row included (the own origin
and the previous official origin + 72 rows are read; rows after T are not; organisers' reply 745700). A22c additionally needs its April
networks `q24m_*_s0-7` (trained with the previous month) and their same-day X1, which this stage does not compute.

What depends on the month: the split dates in `build_cache.py` (above); the physics onset model for the `private` split is the shipped
one trained on the training months plus March (`physon_m.txt`, organisers' reply 748789), applied as is; the evidence rule and the far gate
read only the same month's earlier days, assuming the split starts on the 1st; the folder names `nn_test_afdem`, `nn_data_march`,
`nn_data_april_sp3` are kept for the new month's files (the stage writes its inputs under those names in the fresh `TFB_NN`).

Checks (ledger #246, 11/05): the stage's inputs are bit-identical to the shipped ones (all 80 columns, April's D7_I10_E windows); the
networks first computed on the CPU (pcL, u2d, x3af) are reproduced bit for bit, those computed on the GPU differ by their precision only
(q24f bf16 up to 0.056 per cell, mean 1e-4; tfm4 fp16 up to 2.4e-3); with q24f replaced by the CPU probabilities, A17's Task 2 sets are
identical on all 160 windows (0 of 174,000 cells) and B7's differ in 2 cells. A dry run with April as the new month, one panel
(D7_I10_E) and nothing shipped except the trained artefacts, took 7 minutes with three X1 queues (10 minutes through
`repro/run_new_month.sh` with one, same queues to the cell). On it, A25 / B7c (11/06, the build stage on the same dry folder) matched the
submitted predictions in onset and in B7c's April ongoing; the ongoing cells that use X1 differed in 15 (A25) / 12 (B7c) of 17,520 cells,
because the fine-tuning pool of one panel is an eighth of the real one (11 / 14 for A17 / B7 on 11/05). A whole month takes about 15-20 hours on 16 threads, the same-day X1 of 100 networks being the
long part.

## External data, licences, seeds and hardware

- **External data: none.** No road-network files, map services, weather, calendars, public traffic feeds, pretrained weights
  or paid APIs. Every input comes from the competition release. None of the finals (A25, B7c, nor the alternative A22c) uses synthetic
  data; A22c's April networks (q24m) add all of March to the training months (ruling 748789). An earlier candidate A (finA5 / finA6,
  `e4f` / `e4tfm4`) trained its April networks on synthetic windows from `twin/gen_clean.py` (technical report, Section 8).
- **Licences.** This code: MIT (`LICENSE`). Direct dependencies: NumPy, pandas, SciPy, scikit-learn and PyTorch (BSD-style),
  PyArrow (Apache-2.0), LightGBM (MIT). The competition data stay under the competition's rules; the window features in the
  asset archive are derived from them.
- **Random seeds.** Task 2 networks: `SEED=` in each line of `repro/nn_configs/production_runs.txt` (q24f 0-15, tfm4 0-4; the
  alternative A22c's q24m 0-7 in `review/wsl_nn3t_q24m.sh`; e4f 0-11 and e4tfm4 0-3 for earlier candidates only). X1: seed 0 (`src/t2_nn_finetune.py`). LightGBM models of Task 2: seed 7 (`src/t2_model.py`, `src/t2_phys_onset.py`).
  Task 1: f3big 1, f3all 1, f3all2 7, boundary model 5 (`src/t1_bnd.py`), blank-cell model 3 and rng 11 (`src/t1_gapml.py`,
  `src/t1_gapml_apply.py`).
- **Hardware.** Developed on an AMD Ryzen 7 9700X (8 cores, 16 threads) with 31 GB RAM and a Radeon RX 7800 XT; the nine attention
  networks were trained on Kaggle T4 GPUs. The default path needs a CPU only: 73 s for one final and 90-155 s for two from a
  fresh clone (172 s on one core), about 5 GiB of RAM in the largest process (private memory 6.6 GB on Windows, so have 8 GB),
  2.6 GiB for the unpacked assets and 12 GB for the release. The full rebuild peaks at 4.9 GiB and unpacks to 4.8 GiB.

## Optional: retraining

The default path starts from the saved networks. X1 (each network fine-tuned on the month's completed earlier days) is
recomputed with `STAGES="t2x1" X1_ARM=rep bash repro/run_final.sh` (CPU, ≈ 6–8 h, deterministic for the same torch thread
count, which the code fixes at 16; the months' fine-tuning windows are rebuilt from the release when missing). The networks are trained with `src/t2_nn2.py` (configuration lines:
`repro/nn_configs/production_runs.txt` for every final network; ≈ 18–21 min each on one RX 7800 XT, so ≈ 30–35 h
for all 100). Their training sets are built by the dated chains in `work/*_chain.sh` and `twin/build_merge.py` (provenance in
`LINEAGE.md`); that path was run on our machine only and is not needed to reproduce the submissions. On the same stack a
retrained network is bit-identical; on a different stack the outputs move by |Δp| ≈ 0.002 on average, ≈ −0.0002 in the
total score.

The Task 1 models are LightGBM boosters shipped in the assets; the shipped files are authoritative, because they were trained
before the first commit of the training scripts and a retrain is equivalent but not bit-identical. `python t1_big.py 2000000 f3big 0.05`
(row count and learning rate reconstructed from the experiment log; seed 1, feature fraction 0.8), `python t1_full.py f3all 2000000
6500 3600 1 0.8`, `python t1_full.py f3all2 2000000 6500 3600 7 0.6`. The boundary model `bnd3` joins the speed model of
`audit/d3_bnd_retrain.py` (`bnd2`) and the flow model of `src/t1_bnd.py` (`bnd`).

The nine attention networks were trained with the Kaggle scripts `repro/nn_configs/kaggle_tfm1.py` and `kaggle_tfm_e4.py`
(same `src/t2_nn2.py`, fp16 on T4). Their configuration lines are in `production_runs.txt`; `kaggle_tfm1.py` is the kernel's last
version, whose job list had moved on to a record-only run, so the lines of `tfm4` come from the saved weights' config and the
kernel's `BASE` string.

## Licence and acknowledgement

MIT (`LICENSE`). The analysis and code were developed with AI assistance (Claude); design decisions, numbers and their
provenance are recorded in the decision ledger of the technical report.

sha256 of `submission.csv` in the submitted files: finA25 `139592a38586a256336bb11243dd6d91d960d374e6da87732b6169b74bbff01d` (A25t), finA22c `2a2bd56c4190fc7b0f11fa2eef7011cce23598a55856ba191dc418a6a3942841` (A22ct), finB7c `eac4a4c8df3796274574db5a4c6c9dd67fcd6ab54fdfcf2dd45e074d6eb1ab45`
(B7ct); the full list with the earlier finals is in
`repro/run_final.sh` (`SHA[...]`), which `verify` uses. Earlier finals: finA13 `16bcab21c2fa9ae901c211125ca06a46c1865dea1587f52f9608fd7114cfd4a9`,
finB11 `9259eae7e78696f89d6e2637618c551572152c4f14094f4629ff173f978d48eb`.
