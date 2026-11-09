"""Task 2 v3 inference: v2 features, train windows + extra (1h, 20min spacing), size weighting, negative downsampling."""
import pickle, sys, numpy as np, pandas as pd, lightgbm as lgb
import t2_exp
from cachelib import load, T2_PANELS
from common import REL
from t2_windows import F
from t2_model import PanelInputs, priors, best_set, PARAMS, OUT
from t2_model2 import predict_window
from t2_exp import design_w
tag=sys.argv[1] if len(sys.argv)>1 else "v3"
t2_exp.NEG_KEEP=0.15
W=pickle.load(open(r"C:/Users/user/tfb/work/t2_windows_train.pkl","rb"))
E=pickle.load(open(r"C:/Users/user/tfb/work/t2_windows_extra12.pkl","rb"))+pickle.load(open(r"C:/Users/user/tfb/work/t2_windows_extra4.pkl","rb"))
Ps={p:load(p,"train") for p in T2_PANELS}; PIs={p:PanelInputs(Ps[p]) for p in T2_PANELS}
pr=priors(W,{p:Ps[p].n for p in T2_PANELS},set())
models={}
for cond,nr in (("queue_onset",600),("queue_ongoing",800)):
    X,y,g,wt=design_w(W+E,PIs,pr,cond,"size"); models[cond]=lgb.train(PARAMS,lgb.Dataset(X[g],y[g],weight=wt[g]),nr)
    models[cond].save_model(str(OUT/f"{cond}_{tag}.txt"))
pickle.dump(pr,open(OUT/f"priors_{tag}.pkl","wb"))
out=[]
for split in ("validation","private"):
    for panel in T2_PANELS:
        P=load(panel,split); PI=PanelInputs(P); ti={s:i for i,s in enumerate(P.times)}
        wi=pd.read_csv(REL/"task2"/panel/split/"window_index.csv")
        for r in wi.itertuples():
            t=ti[r.forecast_origin]; p=predict_window(models[r.condition],PI,t,r.condition,pr[panel])
            sel=best_set(p.ravel(),np.ones(p.size,bool)).reshape(p.shape)
            stamps=[P.times[t+s+1] for s in range(F)]
            out.append(pd.DataFrame({"window_id":r.window_id,"timestamp":np.repeat(stamps,P.n),"link_id":np.tile(P.links,F),"queue_pred":sel.ravel().astype(int)}))
q=pd.concat(out,ignore_index=True); q.to_parquet(OUT/f"queue_{tag}.parquet",index=False); print("rows",len(q),"pos",int(q.queue_pred.sum()))
