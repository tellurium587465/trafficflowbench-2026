@echo off
python C:\Users\user\tfb\src\guard_mem.py 9000 || exit /b 1
set TFB_T1_ADAPT=1
set TFB_T1_FEAT_TRAINSTAT=1
set TFB_T1_OUTSUF=_ad2
for %%m in (f3big f3all f3all2) do python t1_infer2.py 3 %%m > E:\tfb_nn\t1ad2_%%m.log 2>&1 || (echo FAILED %%m >> E:\tfb_nn\t1ad2_done.log & exit /b 1)
echo DONE > E:\tfb_nn\t1ad2_done.log
