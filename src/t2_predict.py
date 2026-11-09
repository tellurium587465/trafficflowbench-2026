"""Predict Task 2 probabilities with saved models (one or more tags), average, select, write queue parquet."""
import sys, pickle, numpy as np, pandas as pd, lightgbm as lgb
import t2_fast as T
from cachelib import load, T2_PANELS
from common import REL
from t2_windows import F
from t2_model import PanelInputs, best_set, OUT
from t2_cache import window_X
tags=sys.argv[1].split(","); outtag=sys.argv[2]
pr=T.priors(set())
models={tag:{c:lgb.Booster(model_file=str(OUT/f"{c}_{tag}.txt")) for c in ("queue_onset","queue_ongoing")} for tag in tags}
out=[]; probs={}
for split in ("validation","private"):
    for panel in T2_PANELS:
        P=load(panel,split); PI=PanelInputs(P); ti={s:i for i,s in enumerate(P.times)}
        wi=pd.read_csv(REL/"task2"/panel/split/"window_index.csv")
        for r in wi.itertuples():
            t=ti[r.forecast_origin]; X,_=window_X(PI,t,r.condition)
            if T.USE_MP:
                from t2_monthprior import features as mpf
                X=np.column_stack([X,mpf(panel,split,t)])
            if T.USE_ARR:
                from t2_arrival import arrival_features
                from t2_arrival2 import arrival_features2
                af=arrival_features2 if T.ARR_V=="_v2" else arrival_features
                Vw,Qw,tlw=PI.window(t); Xa,_=af(P,Vw,Qw,tlw,PI.x); X=np.column_stack([X,Xa])
            if T.USE_DEM:                                               # weekday x ToD demand forecast, profile from all train months
                from t2_demand import demand_features, profile
                from t2_arrival import _dmin
                Vw,Qw,tlw=PI.window(t); X=np.column_stack([X,demand_features(P,Qw,tlw,t,P.times[t][:10],profile(load(panel,"train")),_dmin(P,PI.x))])
            X[:,T.I_PON]=pr[panel][0]; X[:,T.I_POG]=pr[panel][1]; X=T.apply_drop(X); n=len(X)
            ps=[]
            for tag in tags:
                m=models[tag][r.condition]
                if r.condition=="queue_onset":
                    p=np.zeros((F,n)); p[-1]=m.predict(X,num_threads=16)
                else:
                    p=m.predict(np.vstack([T.step_block(X,s+1) for s in range(F)]),num_threads=16).reshape(F,n)
                ps.append(p)
            p=np.mean(ps,axis=0); probs[r.window_id]=p
            sel=best_set(p.ravel(),np.ones(p.size,bool)).reshape(p.shape)
            stamps=[P.times[t+s+1] for s in range(F)]
            out.append(pd.DataFrame({"window_id":r.window_id,"timestamp":np.repeat(stamps,n),"link_id":np.tile(P.links,F),"queue_pred":sel.ravel().astype(int)}))
q=pd.concat(out,ignore_index=True); q.to_parquet(OUT/f"queue_{outtag}.parquet",index=False); pickle.dump(probs,open(OUT/f"probs_{outtag}.pkl","wb"))
print("rows",len(q),"pos",int(q.queue_pred.sum()))
