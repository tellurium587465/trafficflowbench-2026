# Reproduction procedure and checks (2026-10-12, appended 10-22)

Japanese original: `repro/README_ja.md`

For the lineage, see ../LINEAGE.md.

## Checked

- `check_t2.sh`: rebuilds the later T2 stages from the saved predictions of the production models (`work/t2/probs_<tag>.pkl`).
  - The later stages are: blend → same-month evidence → far-prediction gate → combining onset and ongoing.
  - The T2 tables of s60 and s63 **matched exactly** what was submitted (10/12).
  - The blend probabilities (the s46 blend, the q37f blend) also matched exactly.

- T1 boundary stage: `py t1_bnd_apply.py bnd3 0.75 <…_fd> <out>` **matched exactly** the T1 used in the submission (10/12).

## Not checked / caveats

- GPU training (`src/t2_nn2.py`) has ROCm non-determinism, and does not match bit for bit even with the same seed.
  - 2026-11-01 correction: this is the 10/12 statement. Later measurements found that on the same stack 83 tensors matched bit for bit (README, Optional: retraining). On a different stack they do not match.
  - The results are equivalent within the seed variance (blend ±0.0005).
  - The starting point for reproduction is the saved models (from 10/22 `C:/tfb_data/tfb_nn/models/*.pt`, changeable with the environment variable `TFB_NN`) and their predictions.
- T1 (`work/t1/state_f3ens3c_ad2_gapml_clip_fd_bnd.parquet`) and T4 (`work/t4/odme_l2.parquet`) use the already-built files. The procedure to rebuild them will be written up separately (before the freeze).

## 2026-10-22 addendum: the organizers' T2 check (delete rows after T and re-run, discussion 749601)

- Entry point: `src/t2_window_entry.py <window_id> <out> [cpu|saved]`. Processes a single window from the given public data.
  - Steps: build the cache (from `TFB_REL` to `TFB_CACHE`) → features → probabilities → blend → same-month evidence → far-prediction gate.
  - There are two ways to produce the probabilities. `cpu` runs inference on the CPU from the saved weights (does not depend on ROCm). `saved` uses the GPU probabilities used in the submission as they are.
  - Output: `work/t2/queue_entry_<out>.parquet` (6 × n rows for that window).
- Check: `src/t2_trunc_physical.py` makes a copy of the public data files physically cut at the window's T, and passes it through the entry point. Targets: 10 windows (all 8 routes, March and April, onset and ongoing).
  - The features matched the saved `nn_test_afdem` bit for bit (10/10).
  - The predictions made with `saved` matched the submitted s70 (10/10).
  - The predictions made with `cpu` matched between the cut copy and the full month (10/10). 0 cells differ from the submitted version.
- The existing scripts change behaviour only when the environment variables (`TFB_CACHE`, `TFB_T2_WINDOWS`, `TFB_T2_TEST`) are given. With the defaults, `STAGES="t2 verify" bash repro/run_final.sh` reproduced s60 and s70 byte for byte identical to the submitted versions.
- The T1 model stage does not fail on a month cut at T (`src/t1_short_month_check.py`, 2/2, all predictions finite). The later T1 stages (blend to boundary) assume the files for all routes, so they are not checked.
- The predictions read only rows before T. In the later stages, `cachelib.asof(P, T)` guarantees this structurally.

## 2026-10-23 addendum: reproducing final candidates A / B

- The composition is `FINAL` in `src/t2_final_spec.py` (named envs and a selection per condition). The all-window assembly `t2_build_variant.py` and the single-window entry point `t2_window_entry.py` use the same function `window_probs`. Assembly: `py t2_build_final.py <A|B> <name>`. The contents and the reasons for adoption are in "Composition of final A / B" in DECISION_LEDGER.md.
  - March windows (A and B identical): onset = q24f seeds 0–7, ongoing = c24f (with C-world synthetic data) + X1 + cone → evidence rule → far-prediction gate.
  - B's April: onset = q24f seeds 0–7, ongoing = c24f + X1. A's April: E4 (e4f) + X1 + physical onset (a fitted to the month) + no-memory LightGBM 0.2 + cone.
- The starting point is the saved network probabilities (`work/t2/probs_<tag>.pkl`). From there on everything is deterministic on the CPU (bit match confirmed on 10/23):
  - `t2_lgb_nomem_final.py apply`, `t2_phys_onset.py apply`, `t2_phys_onset.py adaptive` match the saved versions.
  - X1 (`t2_x1_official.py`, seed 0) matches the saved version even when fine-tuned on the spot from public data cut at T (20 E4 models, difference 0).
- Stages of `bash repro/run_final.sh`: t1 t4 t2aux (LightGBM, physical, a) t2x1 (X1, about 4 hours, not run by default) t2final (A, B) verify.
- The organizers' check (delete rows after T and re-run): `py t2_trunc_physical.py <A|B> <queue parquet> [--cpu n] [--only 1,2,3,4]`.
  - Stage 1: in all 160 windows, the features rebuilt from the cut copy (also their 8 copies for configurations that use TTA) and the final predictions match the submitted version bit for bit.
  - Stage 2: for each day, the X1 training windows (up to the previous day) match the saved version bit for bit.
  - Stage 3: recompute all components on the CPU (LightGBM, physical, a and the floor-raising statistics match bit for bit; for the NN, the difference from the GPU and the difference in predicted cells).
  - Stage 4: fine-tune X1 on the spot and compare with the saved version.
  - The provisional A (finA2pre, the same components as final A except c24f) passed all stages (10/23 07:13). The cut copy covers all routes of the window's split (the physical a and X1 read the whole month up to T), and window_index keeps the windows at or before T.
- The TTA jitter uses random numbers fixed to the calendar grid, so it takes the same values on the cut copy too. For a single window, the part for earlier months and routes is skipped in the shape given by `repro/tta_shapes.json` to reproduce the same random sequence (final A and B do not use TTA).

## 2026-10-24: making run_final.sh portable
- The root comes from the script's location (can be overridden with `TFB_ROOT`), the work folder is `TFB_WORK`, the networks are `TFB_NN`. t2aux is `t2_lgbnm_apply.py`; t2x1 builds the month windows if they are missing.
- Check: `STAGES="t2aux t2final verify" FINAL_A_REF=finA4 FINAL_B_REF=finB4 bash repro/run_final.sh` reproduced A's and B's T2, T1 and T4 exactly as submitted (159 seconds).
- 10/24 06:10 (038cd47): the default is now the fast path (cache → t2aux → t2final → verify). `FINAL=B` or an argument selects B; `PY` (default python3, on Windows PY=py). verify compares the sha256 of submission.csv with the submitted version (works on any machine). The old verify (component-by-component comparison with the submission log) is `verifylocal`. `PY=py FINAL="A B" bash repro/run_final.sh` matched for both (64 seconds, with cache). The English entry point is the root README.md.

## 2026-10-23 night: rebuilding the one selected final submission (the organizers reproduce the one selected)
- The final two: **A = finA4** (`work/subs/finA4_final.zip`) / **B = finB4** (`work/subs/finB4_final.zip`). Contents: "Composition of final A / B" in DECISION_LEDGER.md; code: FINAL["A"] / FINAL["B"] in `src/t2_final_spec.py`.
- **Rebuild A** (from the saved network probabilities, CPU, about 10 minutes):
  `cd src && py t2_lgbnm_apply.py && py t2_phys_onset.py apply && py t2_phys_onset.py adaptive && py t2_build_final.py A repro_finA` (10/24: `t2_lgbnm_apply.py` is a version that does not read the 3 GB training cache; its output matches `t2_lgb_nomem_final.py apply` bit for bit)
- **Rebuild B** (does not use LightGBM or physical onset, so only the later stages):
  `cd src && py t2_build_final.py B repro_finB`
- (11/01: X1 was compared in full over 100 networks and 16,000 windows, 0 mismatches) To rebuild from X1, first run `STAGES="t2x1" bash repro/run_final.sh` (about 6–8 hours, CPU, deterministic; if the windows for the month fine-tuning, nn_data_march / nn_data_april, are missing, build them from the release with `twin/build_march.py`). **10/24 06:03: X1 for all 80 networks was rebuilt under a separate name (X1_ARM=rep), and all 12,800 window predictions matched the saved version bit for bit (work/x1rep_check.sh)**. For the whole thing including T1 / T4: `bash repro/run_final.sh` (default: t1 t4 t2aux t2final verify).
- Verification: `STAGES="verify" FINAL_A_REF=finA4 FINAL_B_REF=finB4 bash repro/run_final.sh` (whether the T2 table matches the submitted version exactly). On 10/23 21:46 an exact match was confirmed for both A and B.
- Per-window causality check (copy with everything after T deleted): `py t2_trunc_physical.py A ../work/t2/queue_finA4.parquet --cpu 4` and `--only 4 <window>` (likewise for B). On 10/23 both A and B passed all stages (160 / 160 windows, X1 training windows 22 / 22 days, 0 cell differences in the CPU recomputation, 0 difference in the on-the-spot fine-tuning of X1).

## 2026-10-23 late night: rebuilding in a separate folder from the distributed inputs only (audit day 1 (2))
- `bash repro/clean_folder_check.sh`: copies the code (src, twin/build_march.py, repro) into a new folder (C:/tfb_clean), puts in only the distributed inputs (T1 state and T4, the saved probabilities of the final networks and the X1 probabilities, the LightGBM and physical-onset models, the training-month rarity table work/t2/rarity_table.pkl, the official-window features nn_test_afdem, the feature names), and builds final A and B starting from building the cache from the public data. **finA4 / finB4 and submission.csv matched the submitted ones byte for byte (10/23 23:20, 180 seconds)**.
- Environment variables: `TFB_REL` (location of the public data; default data/kaggle_public in the repository), `TFB_WORK` (work folder; default work in the repository), `TFB_NN` (location of the network weights and features), `TFB_CACHE` (default TFB_WORK/cache). Machine-dependent paths in the code were removed on 10/23 (common.ROOT is derived automatically from src/..).
- The inference path does not use the 3 GB training cache: LightGBM uses `t2_lgbnm_apply.py` (column names in repro/featcache_names.json), and the physical onset's a uses rarity_table.pkl (both are static artifacts built from the training months only).
- **T1 in a separate folder too** (`bash repro/clean_t1_check.sh`, after clean_folder_check.sh): from the distributed T1 inputs (3 LightGBM sets, 6 files; 2 boundary models; the list of training windows of the gap-specific model, featcache2_noT.npz, 9 MB): building the cache (all 10 routes × 3 splits) → inference of the 3 sets → blend → gap-specific (relearned on the spot, deterministic) → clipping → FD → boundary = **an exact match with the submitted T1 (6,735,795 rows, 10/24 00:35, 60 minutes)**. T4 was also rebuilt in a separate work folder and matched the submitted version (10/23). → All three components of the submission (T1, T2, T4) can be rebuilt from the distributed inputs and the public data alone.

## 2026-10-31: B7 and B8
- B7 = B6 with X1 removed from April's ongoing; B8 = finB4's March + B7's April (the B for the case where finA5 remains in #63; not submitted). Both can be built from `tfb_final_assets_A6B6.zip` with no additional assets.
- Check: `bash repro/recheck_1031.sh` (fresh clone, FINAL="A6 B6 B7 B8"). All 4 matched byte for byte on Windows in 134 seconds and on Linux in 139 seconds; the log is `repro/logs/recheck_1031b.txt`.
