@echo off
python C:\Users\user\tfb\src\guard_mem.py 8000 || exit /b 1
python t1_ens.py _ad2 C:/Users/user/tfb/work/t1/state_f3ens3c_ad2.parquet > E:\tfb_nn\s26.log 2>&1 || exit /b 1
python t1_gapml_apply.py 3000 1900 0.1 C:/Users/user/tfb/work/t1/state_f3ens3c_ad2.parquet C:/Users/user/tfb/work/t1/state_f3ens3c_ad2_gapml.parquet >> E:\tfb_nn\s26.log 2>&1 || exit /b 1
python -W ignore t1_clip.py C:/Users/user/tfb/work/t1/state_f3ens3c_ad2_gapml.parquet C:/Users/user/tfb/work/t1/state_f3ens3c_ad2_gapml_clip.parquet >> E:\tfb_nn\s26.log 2>&1 || exit /b 1
python -W ignore t1_fd.py C:/Users/user/tfb/work/t1/state_f3ens3c_ad2_gapml_clip.parquet C:/Users/user/tfb/work/t1/state_f3ens3c_ad2_gapml_clip_fd.parquet >> E:\tfb_nn\s26.log 2>&1 || exit /b 1
echo BUILT >> E:\tfb_nn\s26.log
