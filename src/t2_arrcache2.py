import os, time, numpy as np
from cachelib import load
from t2_model import PanelInputs
from t2_arrival2 import arrival_features2
SUF=os.environ.get("TFB_T2CACHE","")
Z=np.load(rf"C:/Users/user/tfb/work/t2/featcache{SUF}.npz",allow_pickle=True)
off=Z["off"]; panel=Z["panel"]; t=Z["t"]; PIs={}; out=None; t0=time.time()
for k in range(len(off)-1):
    p=str(panel[k])
    if p not in PIs: PIs[p]=PanelInputs(load(p,"train"))
    PI=PIs[p]; V,Q,tl=PI.window(int(t[k])); X,names=arrival_features2(PI.P,V,Q,tl,PI.x)
    if out is None: out=np.zeros((off[-1],X.shape[1]),np.float32)
    out[off[k]:off[k+1]]=X
np.save(rf"C:/Users/user/tfb/work/t2/arrcache{SUF}_v2.npy",out); print("done",out.shape,names,f"{time.time()-t0:.0f}s")
