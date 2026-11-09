"""Task 2 final: train on all cached train windows (featcache, no fold), predict validation + private."""
import sys, json, pickle, numpy as np, pandas as pd, lightgbm as lgb
import t2_fast as T
from cachelib import load, T2_PANELS
from common import REL
from t2_windows import F
from t2_model import PanelInputs, best_set, PARAMS, OUT
from t2_cache import window_X
tag=sys.argv[1]; prm=dict(PARAMS); prm.update(json.loads(sys.argv[2])); rounds=tuple(int(x) for x in sys.argv[3].split(","))
import os
T.PRIOR_LOMO = os.environ.get("TFB_T2_LOMO","0")=="1"
pr=T.priors(set())
tr=np.flatnonzero((T.META["is_eval"] | True) & ~T.META["is_sel"])
models={}
for cond,nr in zip(("queue_onset","queue_ongoing"),rounds):
    X,y,w=T.design(tr,pr,cond); models[cond]=lgb.train(prm,lgb.Dataset(X,y,weight=w),nr); models[cond].save_model(str(OUT/f"{cond}_{tag}.txt"))
    print(cond,"trained",X.shape,flush=True)
def feats(PI,t,cond,panel):
    X,_=window_X(PI,t,cond); X[:,T.I_PON]=pr[panel][0]; X[:,T.I_POG]=pr[panel][1]; return X
out=[]
for split in ("validation","private"):
    for panel in T2_PANELS:
        P=load(panel,split); PI=PanelInputs(P); ti={s:i for i,s in enumerate(P.times)}
        wi=pd.read_csv(REL/"task2"/panel/split/"window_index.csv")
        for r in wi.itertuples():
            t=ti[r.forecast_origin]; X=feats(PI,t,r.condition,panel); n=len(X)
            if r.condition=="queue_onset":
                p=np.zeros((F,n)); p[-1]=models[r.condition].predict(X,num_threads=16)
            else:
                p=models[r.condition].predict(np.vstack([T.step_block(X,s+1) for s in range(F)]),num_threads=16).reshape(F,n)
            sel=best_set(p.ravel(),np.ones(p.size,bool)).reshape(p.shape)
            stamps=[P.times[t+s+1] for s in range(F)]
            out.append(pd.DataFrame({"window_id":r.window_id,"timestamp":np.repeat(stamps,n),"link_id":np.tile(P.links,F),"queue_pred":sel.ravel().astype(int)}))
q=pd.concat(out,ignore_index=True); q.to_parquet(OUT/f"queue_{tag}.parquet",index=False); print("rows",len(q),"pos",int(q.queue_pred.sum()))
