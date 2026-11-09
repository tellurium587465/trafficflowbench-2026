"""Append month-adaptive prior features for every cached window (aligned with featcache rows)."""
import os, time, numpy as np
from t2_monthprior import features
SUF=os.environ.get("TFB_T2CACHE","")
Z=np.load(rf"C:/Users/user/tfb/work/t2/featcache{SUF}.npz",allow_pickle=True)
off=Z["off"]; panel=Z["panel"]; t=Z["t"]
out=np.zeros((off[-1],4),np.float32); t0=time.time()
for k in range(len(off)-1):
    out[off[k]:off[k+1]]=features(str(panel[k]),"train",int(t[k]))
    if k%10000==0: print(k,f"{time.time()-t0:.0f}s",flush=True)
np.save(rf"C:/Users/user/tfb/work/t2/mpcache{SUF}.npy",out); print("done",out.shape,f"{time.time()-t0:.0f}s")
