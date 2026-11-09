# 再現の手順と検査(2026-10-12、10-22 追記)

系譜は ../LINEAGE.md を参照。

## 検査済み

- `check_t2.sh`:保存済みの本番モデルの予測(`work/t2/probs_<tag>.pkl`)から、T2 の後段を作り直す。
  - 後段とは、混合 → 同じ月の根拠 → 遠い予測の判定 → onset と ongoing の組み合わせ。
  - s60 と s63 の T2 の表が、提出したものと**完全一致**した(10/12)。
  - 混合の確率(s46 の混合、q37f の混合)も完全一致した。

- T1 の境界の段:`py t1_bnd_apply.py bnd3 0.75 <…_fd> <out>` で、提出に使った T1 と**完全一致**した(10/12)。

## 未検査・注意

- GPU の学習(`src/t2_nn2.py`)は ROCm の非決定性があり、同じ種でもビット単位では一致しない。
  - 2026-11-01 訂正:これは 10/12 の記述。その後の実測では、同じスタックなら 83 テンソルがビット一致した(README の Optional: retraining)。別のスタックでは一致しない。
  - 種のぶれ(混合 ±0.0005)の範囲で同等になる。
  - 再現の起点は、保存済みのモデル(10/22 から `C:/tfb_data/tfb_nn/models/*.pt`、環境変数 `TFB_NN` で変更可)と、その予測とする。
- T1(`work/t1/state_f3ens3c_ad2_gapml_clip_fd_bnd.parquet`)と T4(`work/t4/odme_l2.parquet`)は、作成済みのものを使っている。作り直しの手順は別途まとめる(凍結前に)。

## 2026-10-22 追記:主催者の T2 検査(T より後の行を消して再実行、議論 749601)

- 入口:`src/t2_window_entry.py <window_id> <out> [cpu|saved]`。1 窓だけを、与えられた公開データから処理する。
  - 手順:キャッシュ作り(`TFB_REL` から `TFB_CACHE` へ)→ 特徴量 → 確率 → 混合 → 同じ月の根拠 → 遠い予測の判定。
  - 確率の作り方は 2 つ。`cpu` は保存済みの重みから CPU で推論する(ROCm に依存しない)。`saved` は提出に使った GPU の確率をそのまま使う。
  - 出力:`work/t2/queue_entry_<out>.parquet`(その窓の 6 × n 行)。
- 検査:`src/t2_trunc_physical.py` で、公開データのファイルを窓の T で物理的に切った写しを作り、入口を通した。対象は 10 窓(全 8 路線、3 月・4 月、onset・ongoing)。
  - 特徴量は、保存済みの `nn_test_afdem` とビット単位で一致した(10/10)。
  - `saved` で作った予測は、提出版の s70 と一致した(10/10)。
  - `cpu` で作った予測は、切った写しと全月とで一致した(10/10)。提出版との差は 0 セル。
- 既存の台本は、環境変数(`TFB_CACHE`、`TFB_T2_WINDOWS`、`TFB_T2_TEST`)を与えたときだけ動きが変わる。既定のままの `STAGES="t2 verify" bash repro/run_final.sh` では、s60 と s70 が提出版とバイト単位で一致した。
- T1 の模型の段は、T で切った月でも落ちない(`src/t1_short_month_check.py`、2/2、予測はすべて有限)。T1 の後段(混合〜境界)は全路線のファイルを前提にしているため、未検査。
- 予測が読むのは T より前の行だけである。後段では `cachelib.asof(P, T)` によってこれを構造的に保証している。

## 2026-10-23 追記:最終候補 A / B の再現

- 構成は `src/t2_final_spec.py` の `FINAL`(名前つきの env と条件ごとの選択)。全窓の組み立て `t2_build_variant.py` と 1 窓の入口 `t2_window_entry.py` は同じ関数 `window_probs` を使う。組み立ては `py t2_build_final.py <A|B> <名前>`。中身と採否の根拠は DECISION_LEDGER.md の「最終 A / B の構成」。
  - 3 月の窓(A・B 同一):onset = q24f 種 0〜7、ongoing = c24f(C 世界の合成入り)+ X1 + 円錐 → 根拠の規則 → 遠い予測の判定。
  - B の 4 月:onset = q24f 種 0〜7、ongoing = c24f + X1。A の 4 月:E4(e4f)+ X1 + 物理 onset(月に合わせる a)+ 記憶なし LightGBM 0.2 + 円錐。
- 起点は保存済みの網の確率(`work/t2/probs_<tag>.pkl`)。そこから先は CPU で決定的(10/23 にビット一致を確認):
  - `t2_lgb_nomem_final.py apply`、`t2_phys_onset.py apply`、`t2_phys_onset.py adaptive` は保存版と一致。
  - X1(`t2_x1_official.py`、種 0)は、T で切った公開データからその場で微調整しても保存版と一致(E4 の 20 模型、差 0)。
- `bash repro/run_final.sh` の段:t1 t4 t2aux(LightGBM・物理・a)t2x1(X1、約 4 時間、既定では回さない)t2final(A・B)verify。
- 主催者の検査(T より後の行を消して再実行):`py t2_trunc_physical.py <A|B> <queue parquet> [--cpu n] [--only 1,2,3,4]`。
  - 段 1:全 160 窓で、切った写しから作り直した特徴(TTA を使う構成ではその 8 本も)と最終予測が提出版とビット一致。
  - 段 2:日ごとに、X1 の学習窓(前の日まで)が保存版とビット一致。
  - 段 3:CPU で全部品を再計算(LightGBM・物理・a・底上げの統計はビット一致、NN は GPU との差と予測セルの差)。
  - 段 4:X1 をその場で微調整して保存版と比較。
  - 仮置きの A(finA2pre、c24f 以外は最終 A と同じ部品)で全段合格(10/23 07:13)。切った写しは窓の分割の全路線(物理の a と X1 は月全体の T までを読む)、window_index は T 以前の窓を残す。
- TTA の揺らぎは暦の格子に固定した乱数で、切った写しでも同じ値になる。1 窓では前の月・路線の分を `repro/tta_shapes.json` の形で読み飛ばして同じ乱数列を再現する(最終 A・B は TTA を使わない)。

## 2026-10-24:run_final.sh の可搬化
- 根は台本の置き場所から(`TFB_ROOT` で上書き可)、作業の置き場は `TFB_WORK`、網は `TFB_NN`。t2aux は `t2_lgbnm_apply.py`、t2x1 は月の窓が無ければ作る。
- 確認:`STAGES="t2aux t2final verify" FINAL_A_REF=finA4 FINAL_B_REF=finB4 bash repro/run_final.sh` で A・B の T2、T1、T4 が提出版と完全一致(159 秒)。
- 10/24 06:10(038cd47):既定を速い経路(cache → t2aux → t2final → verify)に。`FINAL=B` か引数で B、`PY`(既定 python3、Windows は PY=py)。verify は submission.csv の sha256 を提出版と比べる(どの機械でも可)。旧 verify(提出記録との部品ごとの比較)は `verifylocal`。`PY=py FINAL="A B" bash repro/run_final.sh` で両方とも一致(64 秒、キャッシュあり)。英語の入口は根の README.md。

## 2026-10-23 夜:選んだ最終提出 1 本の作り直し(主催者の再現の対象は選ばれた 1 本)
- 最終 2 本:**A = finA4**(`work/subs/finA4_final.zip`)/ **B = finB4**(`work/subs/finB4_final.zip`)。中身は DECISION_LEDGER.md の「最終 A / B の構成」、コードは `src/t2_final_spec.py` の FINAL["A"] / FINAL["B"]。
- **A を作り直す**(保存済みの網の確率から、CPU、約 10 分):
  `cd src && py t2_lgbnm_apply.py && py t2_phys_onset.py apply && py t2_phys_onset.py adaptive && py t2_build_final.py A repro_finA`(10/24:`t2_lgbnm_apply.py` は 3 GB の学習キャッシュを読まない版、出力は `t2_lgb_nomem_final.py apply` とビット一致)
- **B を作り直す**(LightGBM・物理 onset を使わないので後段だけ):
  `cd src && py t2_build_final.py B repro_finB`
- (11/01:X1 は 100 網 16,000 窓を全数比較して不一致 0)X1 から作り直す場合は先に `STAGES="t2x1" bash repro/run_final.sh`(約 6〜8 時間、CPU、決定的。月の微調整用の窓 nn_data_march / nn_data_april が無ければ `twin/build_march.py` でリリースから作る)。**10/24 06:03:別名(X1_ARM=rep)で全 80 網の X1 を作り直し、12,800 の窓予測すべてが保存版とビット一致(work/x1rep_check.sh)**。T1 / T4 も含めた全体は `bash repro/run_final.sh`(既定:t1 t4 t2aux t2final verify)。
- 照合:`STAGES="verify" FINAL_A_REF=finA4 FINAL_B_REF=finB4 bash repro/run_final.sh`(T2 の表が提出版と完全一致するか)。10/23 21:46 に A・B とも完全一致を確認。
- 1 窓ずつの因果律の検査(T より後を消した写し):`py t2_trunc_physical.py A ../work/t2/queue_finA4.parquet --cpu 4` と `--only 4 <window>`(B も同様)。10/23 に A・B とも全段合格(160 / 160 窓、X1 の学習窓 22 / 22 日、CPU 再計算でセル差 0、X1 のその場の微調整で差 0)。

## 2026-10-23 深夜:別のフォルダ・配る入力だけでの作り直し(監査 1 日目 (2))
- `bash repro/clean_folder_check.sh`:コード(src・twin/build_march.py・repro)を新しいフォルダ(C:/tfb_clean)に写し、配る入力だけ(T1 の状態・T4、最終の網の保存確率と X1 の確率、LightGBM・物理 onset の模型、学習月の珍しさの表 work/t2/rarity_table.pkl、公式窓の特徴 nn_test_afdem、特徴の名前)を置いて、キャッシュを公開データから作るところから最終 A・B を組む。**提出した finA4 / finB4 と submission.csv がバイト単位で一致(10/23 23:20、所要 180 秒)**。
- 環境変数:`TFB_REL`(公開データの場所。既定はリポジトリの data/kaggle_public)、`TFB_WORK`(作業フォルダ。既定はリポジトリの work)、`TFB_NN`(網の重み・特徴の置き場)、`TFB_CACHE`(既定 TFB_WORK/cache)。コードの中の機械に依存するパスは 10/23 に除いた(common.ROOT は src/.. から自動)。
- 推論の経路は 3 GB の学習キャッシュを使わない:LightGBM は `t2_lgbnm_apply.py`(列名は repro/featcache_names.json)、物理 onset の a は rarity_table.pkl(どちらも学習月だけから作る静的な成果物)。
- **T1 も別のフォルダで**(`bash repro/clean_t1_check.sh`、clean_folder_check.sh の後):配る T1 の入力(LightGBM 3 組 6 ファイル、境界の模型 2 つ、空白専用の模型の学習窓の一覧 featcache2_noT.npz 9 MB)から、キャッシュ(全 10 路線 × 3 分割)の作成 → 3 組の推論 → 混合 → 空白専用(その場で学び直す、決定的)→ 抑え → FD → 境界 = **提出した T1 と完全一致(6,735,795 行、10/24 00:35、所要 60 分)**。T4 も別の作業フォルダで作り直して提出版と一致(10/23)。→ 提出物の 3 成分(T1・T2・T4)すべてが、配る入力と公開データだけから作り直せる。

## 2026-10-31:B7・B8
- B7 = B6 の 4 月 ongoing から X1 を外したもの、B8 = finB4 の 3 月 + B7 の 4 月(#63 で finA5 が残るときの B、未提出)。どちらも資産の追加なしで `tfb_final_assets_A6B6.zip` から組める。
- 確認:`bash repro/recheck_1031.sh`(新しい clone、FINAL="A6 B6 B7 B8")。Windows 134 秒・Linux 139 秒で 4 本ともバイト一致、ログは `repro/logs/recheck_1031b.txt`。
