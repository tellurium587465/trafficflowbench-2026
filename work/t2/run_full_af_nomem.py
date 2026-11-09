import os, sys
# 10/10: LightGBM without location/time memory (CAUSAL_MAP P1/P7): diversity for the no-memory NN blend
os.environ.update(TFB_T2_NOT="1", TFB_T2CACHE="2_noT_af", TFB_T2_ARR="1", TFB_T2_DEM="1",
                  TFB_T2_DROP="prior_on,prior_og,pos,vf,capl,lanes,length,w,tod,dow")
sys.path.insert(0, r"C:/Users/user/tfb/src")
import t2_fast as T
T.cv(params=dict(num_leaves=127, num_threads=14), rounds=(600, 1500), tag="l127_full_af_nomem", save=True)
