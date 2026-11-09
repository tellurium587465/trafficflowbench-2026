"""Adversarial validation: link-level features of selector-like TRAIN windows vs VALIDATION+PRIVATE windows.
High AUC => covariate shift; top features => what differs."""
import numpy as np, pandas as pd, lightgbm as lgb, pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import t2_fast as T
from cachelib import load, T2_PANELS
from common import REL
from t2_model import PanelInputs
from t2_cache import window_X
from t2_arrival import arrival_features
names=list(T.NAMES)+["arr10_c","arr20_c","arr30_c","arr35_c","bnd_trend","cap_dn_min_rel","arr30_over_dnmin"]
pr=T.priors(set())
def feats(PI,panel,t,cond):
    X,_=window_X(PI,t,cond); X[:,T.I_PON]=pr[panel][0]; X[:,T.I_POG]=pr[panel][1]
    V,Q,tl=PI.window(t); Xa,_=arrival_features(PI.P,V,Q,tl,PI.x); return np.column_stack([X,Xa])
rows={c:[] for c in ("queue_onset","queue_ongoing")}
for split in ("validation","private"):
    for p in T2_PANELS:
        P=load(p,split); PI=PanelInputs(P); ti={s:i for i,s in enumerate(P.times)}
        for r in pd.read_csv(REL/"task2"/p/split/"window_index.csv").itertuples():
            rows[r.condition].append((feats(PI,p,ti[r.forecast_origin],r.condition),1,p))
PIs={p:PanelInputs(load(p,"train")) for p in T2_PANELS}
sel=np.flatnonzero(T.META["is_sel"])
rng=np.random.default_rng(0)
for c in rows:
    ks=[k for k in sel if T.META["cond"][k]==c]; ks=rng.choice(ks,size=min(400,len(ks)),replace=False)
    for k in ks:
        p=str(T.META["panel"][k]); rows[c].append((feats(PIs[p],p,int(T.META["t"][k]),c),0,p))
for c,lst in rows.items():
    # aggregate to window level: statistics over links (mean, max, min of each feature) -> window-level classifier
    X=np.vstack([np.concatenate([np.nanmean(x,0),np.nanmax(np.where(np.isfinite(x),x,-1e9),0),np.nanmin(np.where(np.isfinite(x),x,1e9),0)]) for x,_,_ in lst])
    y=np.array([lab for _,lab,_ in lst]); fn=[f"{s}_{n}" for s in ("mean","max","min") for n in names]
    oof=np.zeros(len(y)); imp=np.zeros(X.shape[1])
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(X,y):
        m=lgb.train(dict(objective="binary",num_leaves=15,learning_rate=0.05,min_data_in_leaf=10,feature_fraction=0.7,verbose=-1,num_threads=16),lgb.Dataset(X[tr],y[tr]),300)
        oof[te]=m.predict(X[te]); imp+=m.feature_importance("gain")
    print(f"== {c}: n train {int((y==0).sum())} vs val/priv {int(y.sum())}  adversarial AUC {roc_auc_score(y,oof):.3f}")
    top=np.argsort(-imp)[:12]
    for i in top:
        a=X[y==0,i]; b=X[y==1,i]
        print(f"   {fn[i]:28s} gain {imp[i]:8.0f}  train median {np.nanmedian(a):9.3f}  val/priv median {np.nanmedian(b):9.3f}")
