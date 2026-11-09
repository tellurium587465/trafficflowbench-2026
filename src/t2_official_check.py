"""Score our Task 2 pipeline on the organizer's official TRAIN windows (June), trained without June."""
import os, numpy as np, pandas as pd, lightgbm as lgb
import t2_fast as T
from cachelib import load, T2_PANELS
from common import REL
from t2_windows import F, queue_matrix, score
from t2_model import PanelInputs, best_set, PARAMS
from t2_cache import window_X
T.PRIOR_LOMO=True
EXC={"2030-06"}
pr=T.priors(EXC)
mon=np.array([d[:7] for d in T.META["date"]])
tr=np.flatnonzero(~np.isin(mon,list(EXC)) & ~T.META["is_sel"])
prm=dict(PARAMS); prm.update(num_leaves=127)
models={}
for cond,nr in (("queue_onset",600),("queue_ongoing",1500)):
    X,y,w=T.design(tr,pr,cond,exclude=EXC); models[cond]=lgb.train(prm,lgb.Dataset(X,y,weight=w),nr)
W=[];preds=[];pers=[];details=[]
for panel in T2_PANELS:
    P=load(panel,"train"); PI=PanelInputs(P); Qm=queue_matrix(P); ti={s:i for i,s in enumerate(P.times)}
    wi=pd.read_csv(REL/"task2"/panel/"train"/"window_index.csv")
    for r in wi.itertuples():
        t=ti[r.forecast_origin]; X,_=window_X(PI,t,r.condition); X[:,T.I_PON]=pr[panel][0]; X[:,T.I_POG]=pr[panel][1]; n=len(X)
        if r.condition=="queue_onset":
            p=np.zeros((F,n)); p[-1]=models[r.condition].predict(X,num_threads=16)
        else:
            p=models[r.condition].predict(np.vstack([T.step_block(X,s+1) for s in range(F)]),num_threads=16).reshape(F,n)
        lab=Qm[t+1:t+F+1]; el=P.E[t+1:t+F+1]
        sel=best_set(p.ravel(),np.ones(p.size,bool)).reshape(p.shape)
        W.append(dict(panel=panel,cond=r.condition,label=lab,elig=el)); preds.append(sel)
        pers.append(np.broadcast_to(Qm[t-1],(F,n)))
        inter=(sel&lab&el).sum(); uni=((sel|lab)&el).sum()
        details.append((panel,r.condition,r.forecast_origin,int((lab&el).sum()),int(sel.sum()),round(inter/max(uni,1),3)))
print("NO_T env:",os.environ.get("TFB_T2_NOT"))
print("model  S_queue(all,onset,ongoing):",np.round(score(W,preds),4))
print("persist S_queue(all,onset,ongoing):",np.round(score(W,pers),4))
d=pd.DataFrame(details,columns=["panel","cond","origin","true_cells","pred_cells","iou"])
print(d.groupby("cond").iou.describe().round(3))
print(d[d.iou<0.5].to_string())
