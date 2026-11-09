import sys, pickle, numpy as np
import t2_fast as T
from cachelib import load, T2_PANELS
from t2_windows import queue_matrix
from t2_model import best_set
oof=pickle.load(open(r"C:/Users/user/tfb/work/t2/oof_l127_noT.pkl","rb"))
freq={p:queue_matrix(load(p,"train")).mean(0) for p in T2_PANELS}
# rare-location windows among ALL cached windows (eval+extra), using history queue location
I_NQ=T.NAMES.index("last_queue")
cnt={"queue_onset":[0,0],"queue_ongoing":[0,0]}; diag=[]
for k in range(T.NWIN):
    if T.META["is_sel"][k]: continue
    lab,el=T.LAB[k]; p=T.META["panel"][k]; cond=T.META["cond"][k]
    links=np.flatnonzero(lab.any(axis=0))
    rare=len(links)>0 and freq[p][links].max()<0.01
    cnt[cond][int(rare)]+=1
    if rare and cond=="queue_ongoing" and k in oof:
        o=oof[k]; s=best_set(o.ravel(),np.ones(o.size,bool)).reshape(o.shape)
        tp=(s&lab&el).sum(); fp=(s&~lab&el).sum(); fn=(~s&lab&el).sum()
        diag.append((tp,fp,fn,(lab&el).sum(axis=1).tolist(),s.sum(axis=1).tolist()))
print("windows (recurrent, rare):",cnt)
d=np.array([x[:3] for x in diag]); print("rare ongoing OOF: n",len(d),"TP",d[:,0].sum(),"FP",d[:,1].sum(),"FN",d[:,2].sum())
for x in diag[:12]: print("  true per step",x[3]," pred per step",x[4])
