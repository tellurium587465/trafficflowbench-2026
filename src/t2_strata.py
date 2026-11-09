"""OOF IoU on selector-like windows, split by whether the window's queue is at a recurrent location."""
import sys, pickle, numpy as np
import t2_fast as T
from cachelib import load, T2_PANELS
from t2_windows import queue_matrix
from t2_model import best_set
tag=sys.argv[1]
oof=pickle.load(open(rf"C:/Users/user/tfb/work/t2/oof_{tag}.pkl","rb"))
freq={}
for p in T2_PANELS:
    P=load(p,"train"); q=queue_matrix(P); freq[p]=q.mean(0)
res={}
for k,o in oof.items():
    if not T.META["is_sel"][k]: continue
    lab,el=T.LAB[k]; p=T.META["panel"][k]; cond=T.META["cond"][k]
    links=np.flatnonzero(lab.any(axis=0))
    rare = len(links)>0 and freq[p][links].max()<0.01
    s=best_set(o.ravel(),np.ones(o.size,bool)).reshape(o.shape)
    iou=((s&lab&el).sum())/max(((s|lab)&el).sum(),1)
    res.setdefault((cond,"rare" if rare else "recurrent"),[]).append(iou)
for key,v in sorted(res.items()): print(tag,key,"n",len(v),"mean IoU %.3f"%np.mean(v))
