# Lineage of the submissions (candidates as of 2026-10-12)

Japanese original: `LINEAGE_ja.md`

**This is the history as of 10/12. The composition of the final two submissions is defined by README.md and `src/t2_final_spec.py`.** The selected finals are A25 and B7c (README.md); this file records the earlier lineage.

Everything is packed into a zip with `src/submit.assemble(T1 state, T2 queue, T4 odme, name)`. The submission log is `work/subs/SUBMISSIONS.jsonl` (with the sha1 of each component).

## Shared components

- **T1**: `work/t1/state_f3ens3c_ad2_gapml_clip_fd_bnd.parquet`
  - 3 models + ADAPT v2 + gap-specific model + ceiling + FD + boundary
  - The s30 line. The no-memory version s48 was −0.00015 on the LB, so it was not adopted.
- **T4**: `work/t4/odme_l2.parquet` (L2 projection)

## The T1 chain (state_f3ens3c_ad2_gapml_clip_fd_bnd)

1. Model training (saved)
   - `work/t1/speed_*.txt`, `flow_*.txt`: f3big (`src/t1_big.py`), f3all / f3all2 (`src/t1_full.py`)
   - Gap-specific: gapml
   - Boundary: bnd3 (`audit/d3_bnd_retrain.py`)
2. ADAPT inference: `work/t1/run_adapt2.cmd`
   - Settings: TFB_T1_ADAPT=1, TFB_T1_FEAT_TRAINSTAT=1, TFB_T1_OUTSUF=_ad2
   - Run: `t1_infer2.py 3 <model>` (f3big, f3all, f3all2)
3. Post-processing: `work/t1/run_s26.cmd`
   - `t1_ens.py _ad2`
   - `t1_gapml_apply.py 3000 1900 0.1`
   - `t1_clip.py`
   - `t1_fd.py` (0.75)
4. Boundary: `t1_bnd_apply.py bnd3 0.75 <..._fd> <..._fd_bnd>` (same form as work/t1/run_t1nomem.sh)
5. Note: on the authors' Windows machine, `python` is a Microsoft Store alias and does not run, so read `python` as `py` when running the .cmd files.

## T4

- `src/t4_candidates.py` (`task4.solve_weighted_l2`, weight 1) → `work/t4/odme_l2.parquet`

## The T2 chain

1. **Production NN training**: `src/t2_nn2.py` (daemon; the data are selected by `work/wsl_nnd_*.sh`)
   - The s46 / s60 line: q24f
     - Data: no memorization, ZERO=pos+tod+dow+capl+lanes+length+vf+w, nn_data_afdem
     - Configuration: 5 configurations × seeds 0–3
     - Blend weights: s46 is in fact 's37' in `src/t2_specs.json` (it also includes r13 seed 0 with weight 1; an exact match was confirmed with repro/check_t2.sh). The 10/13 review found that the old description "r13 1.333, seed 0 excluded" was wrong
   - For ongoing, with synthetic data
     - q37f: nn_data_afdemsimR = real + 8 months of R worlds, 5 configurations × seeds 0–3
     - q43f: nn_data_afdemsimR24h = real + 24 months of R worlds (windows thinned), 5 configurations × seeds 0–3 (s69/s70 use seeds 0–3, s68 uses seeds 0,1)
     - q46f: nn_data_afdemsimR48q = real + 48 months of R worlds (twin/build_simR48.py), 5 configurations × seeds 0–3
     - Training config lines (all 80 runs): repro/nn_configs/production_runs.txt; launch scripts that select the data: repro/nn_configs/wsl_nnd_*.sh
     - Blend weights (s70's ongoing): q43f and q46f seeds 0–3, per seed attnglob 2 / attnwp05 2 / attn1 2 / drop1 1 / r13 1 (40 runs in total)
     - Environment: repro/requirements_wsl_rocm_freeze.txt (training, torch 2.10 ROCm), repro/requirements_windows_freeze.txt (post-processing and assembly). NN training is not bit-for-bit reproducible (ROCm, bf16), so the starting point for reproduction is the saved .pt files / probs
2. **Blend**: `src/t2_blend_submit.py <spec> <name> <T1> <T4>` → `work/t2/probs_<name>.pkl`
3. **Combining onset and ongoing (s63/s66/s67/s68/s69/s70)**: the python inside `work/t2/build_s6x.sh` and `build_s70.sh` (`set -euo pipefail` from 10/13)
   - The onset probabilities are taken from the s46 blend.
   - The ongoing probabilities are taken from the blend with synthetic data.
   - An assert checks that the onset windows match s60 exactly.
4. **Same-month evidence**: `src/t2_onset_evidence.py <probs> <out> 0.25 0.85`
   - onset: links without evidence ×0.25 (previous 3 days, T−30 to T+60 min, ±2 links)
   - ongoing: links without evidence ×0.85 (T−90 to T+60 min, ±1 link)
5. **Far-prediction gate**: `src/t2_far_gate.py <in> <out> 3 1`
   - Predictions more than 3 links away from the latest congestion are removed unless there is same-month evidence.
   - However, predictions that extend contiguously from the latest congestion are kept (CONTIG=1).

## Synthetic months (twin/)

- `gen.py` (random numbers fixed with crc32)
  - World A: seed 0
  - Test worlds T0–T7: seed 0
  - R worlds for training R0–R47: seed 1
  - Retests on separate dates: A6 = seed 2, S = seed 3, U = seed 4
- Input assembly
  - `build_nn.py`: asserts that the column names match the real data
  - `build_nn2.py`
  - Combining with the real data: `build_simR.py`, `build_simR24.py`, `half_synth_r24.py`
- Scoring: `eval_twin.py`, `nested_B.py`, `oof_eval.py`

## Candidates and LB

| Submission | T2 | LB |
|---|---|---|
| s60 | q24f + evidence 0.25 / 0.85 + gate (contiguous predictions exempt) | 0.89178 (the confirmed baseline, **second of the final two**) |
| s63 | s60 with ongoing replaced by q37f seeds 0,1 | 0.89484 |
| s66 | s60 with ongoing replaced by q37f seeds 2,3 | 0.89453 (check of seed variance) |
| s67 | s60 with ongoing replaced by q37f seeds 0–3 | 0.89556 (synthetic-data ongoing confirmed) |
| s69 | s60 with ongoing replaced by q43f (R24) seeds 0–3 | 0.89479 |
| s70 | s60 with ongoing replaced by q43f (R24) + q46f (R48) seeds 0–3 | 0.89527 (**first of the final two**, per the pre-declared rule) |
