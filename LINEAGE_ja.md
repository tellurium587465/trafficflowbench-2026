# 提出物の系譜(2026-10-12 時点の候補)

**これは 10/12 時点の履歴です。最終の 2 本の組成は README.md と `src/t2_final_spec.py` を正とします。**

すべて `src/submit.assemble(T1 state, T2 queue, T4 odme, name)` で zip にする。提出の記録は `work/subs/SUBMISSIONS.jsonl`(各部品の sha1 つき)。

## 共通の部品

- **T1**: `work/t1/state_f3ens3c_ad2_gapml_clip_fd_bnd.parquet`
  - 3 モデル + ADAPT v2 + 空白専用 + 天井 + FD + 境界
  - s30 系。記憶なし版の s48 は LB で −0.00015 だったので不採用。
- **T4**: `work/t4/odme_l2.parquet`(L2 射影)

## T1 の連鎖(state_f3ens3c_ad2_gapml_clip_fd_bnd)

1. モデルの学習(保存済み)
   - `work/t1/speed_*.txt`、`flow_*.txt`:f3big(`src/t1_big.py`)、f3all / f3all2(`src/t1_full.py`)
   - 空白専用:gapml
   - 境界:bnd3(`audit/d3_bnd_retrain.py`)
2. ADAPT 推論:`work/t1/run_adapt2.cmd`
   - 設定:TFB_T1_ADAPT=1、TFB_T1_FEAT_TRAINSTAT=1、TFB_T1_OUTSUF=_ad2
   - 実行:`t1_infer2.py 3 <model>`(f3big、f3all、f3all2)
3. 後処理:`work/t1/run_s26.cmd`
   - `t1_ens.py _ad2`
   - `t1_gapml_apply.py 3000 1900 0.1`
   - `t1_clip.py`
   - `t1_fd.py`(0.75)
4. 境界:`t1_bnd_apply.py bnd3 0.75 <..._fd> <..._fd_bnd>`(work/t1/run_t1nomem.sh と同じ形)
5. 注意:`python` は Store の別名で動かないので、.cmd を実行するときは `py` に読み替える。

## T4

- `src/t4_candidates.py`(`task4.solve_weighted_l2`、重み 1)→ `work/t4/odme_l2.parquet`

## T2 の連鎖

1. **NN の本番学習**:`src/t2_nn2.py`(デーモン、`work/wsl_nnd_*.sh` でデータを選ぶ)
   - s46 / s60 系:q24f
     - データ:暗記なし、ZERO=pos+tod+dow+capl+lanes+length+vf+w、nn_data_afdem
     - 構成:5 構成 × 種 0〜3
     - 混合の重み:s46 の実体は `src/t2_specs.json` の 's37'(r13 の種 0 も重み 1 で含む。repro/check_t2.sh で完全一致を確認済み)。10/13 レビューで、旧記載「r13 1.333、種 0 除外」は誤りと判明
   - 合成入りの ongoing 用
     - q37f:nn_data_afdemsimR = 本物 + R 世界 8 か月、5 構成 × 種 0〜3
     - q43f:nn_data_afdemsimR24h = 本物 + R 世界 24 か月(窓を間引き)、5 構成 × 種 0〜3(s69/s70 は種 0〜3、s68 は種 0,1)
     - q46f:nn_data_afdemsimR48q = 本物 + R 世界 48 か月(twin/build_simR48.py)、5 構成 × 種 0〜3
     - 学習の設定行(全 80 本):repro/nn_configs/production_runs.txt、データを選ぶ起動スクリプト:repro/nn_configs/wsl_nnd_*.sh
     - 混合の重み(s70 の ongoing):q43f と q46f の種 0〜3、各種で attnglob 2 / attnwp05 2 / attn1 2 / drop1 1 / r13 1(計 40 本)
     - 環境:repro/requirements_wsl_rocm_freeze.txt(学習、torch 2.10 ROCm)、repro/requirements_windows_freeze.txt(後処理・組み立て)。NN の学習はビット単位では再現しない(ROCm・bf16)ため、再現の起点は保存済みの .pt / probs とする
2. **混合**:`src/t2_blend_submit.py <spec> <name> <T1> <T4>` → `work/t2/probs_<name>.pkl`
3. **onset と ongoing の組み合わせ(s63/s66/s67/s68/s69/s70)**:`work/t2/build_s6x.sh`・`build_s70.sh` の中の python(10/13 から `set -euo pipefail`)
   - onset の確率は s46 の混合から取る。
   - ongoing の確率は合成入りの混合から取る。
   - onset の窓が s60 と完全一致することを assert で確かめる。
4. **同じ月の根拠**:`src/t2_onset_evidence.py <probs> <out> 0.25 0.85`
   - onset:根拠のないリンクを ×0.25(前 3 日、T−30〜T+60 分、±2 リンク)
   - ongoing:根拠のないリンクを ×0.85(T−90〜T+60 分、±1 リンク)
5. **遠い予測の判定**:`src/t2_far_gate.py <in> <out> 3 1`
   - 直前の渋滞から 3 リンクより遠い予測は、同じ月の根拠がなければ消す。
   - ただし、直前の渋滞から連続して延びる予測は残す(CONTIG=1)。

## 合成の月(twin/)

- `gen.py`(乱数は crc32 で固定)
  - 世界 A:種 0
  - テストの世界 T0〜T7:種 0
  - 学習用の R 世界 R0〜R47:種 1
  - 日付を分けた再試験:A6 = 種 2、S = 種 3、U = 種 4
- 入力の組み立て
  - `build_nn.py`:列名が本物と一致することを assert
  - `build_nn2.py`
  - 本物との合成:`build_simR.py`、`build_simR24.py`、`half_synth_r24.py`
- 採点:`eval_twin.py`、`nested_B.py`、`oof_eval.py`

## 候補と LB

| 提出 | T2 | LB |
|---|---|---|
| s60 | q24f + 根拠 0.25 / 0.85 + 判定(連続除外) | 0.89178(確定の基準、**最終の 2 本目**) |
| s63 | s60 の ongoing を q37f 種 0,1 に | 0.89484 |
| s66 | s60 の ongoing を q37f 種 2,3 に | 0.89453(種のぶれの確認) |
| s67 | s60 の ongoing を q37f 種 0〜3 に | 0.89556(合成入りの ongoing を確定) |
| s69 | s60 の ongoing を q43f(R24)種 0〜3 に | 0.89479 |
| s70 | s60 の ongoing を q43f(R24)+ q46f(R48)種 0〜3 に | 0.89527(**最終の 1 本目**、事前宣言の規則どおり) |
