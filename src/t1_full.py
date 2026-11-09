"""Task 1 final model on all train months (no holdout), fixed rounds from the f3big early-stopping run."""
import os, sys, time, numpy as np, lightgbm as lgb
from pathlib import Path
from cachelib import load
from common import PANELS, WORK
from t1_exp import featfns
OUT=Path(WORK)/"t1"
tag=sys.argv[1]; ntrain=int(sys.argv[2]); rv,rq=int(sys.argv[3]),int(sys.argv[4]); seed=int(sys.argv[5]) if len(sys.argv)>5 else 1
gf,rf,bf=featfns(3); rng=np.random.default_rng(seed); t0=time.time()
Xs,yv,yq,ws=[],[],[],[]
for pi,panel in enumerate(PANELS):
    P=load(panel,"train"); F,_,_,_=gf(P); bv,bq=bf(F)
    r,c=np.nonzero(P.target&P.E&np.isfinite(P.V))
    hard=np.nan_to_num(F["min_vr_nb"][r,c],nan=1.0)<0.9
    pr=np.where(hard,10.0,1.0); pr=pr/pr.sum()
    idx=rng.choice(len(r),size=min(ntrain,len(r)),replace=False,p=pr)
    X,_=rf(P,F,r[idx],c[idx]); Xs.append(np.column_stack([np.full(len(idx),0 if os.environ.get('TFB_T1_NOMEM','0')=='1' else pi,np.float32),X]))
    yv.append(P.V[r[idx],c[idx]]/P.vf[c[idx]]-bv[r[idx],c[idx]]); yq.append(P.Q[r[idx],c[idx]]/P.lanes[c[idx]]-bq[r[idx],c[idx]])
    ws.append(1.0/(pr[idx]*len(r)))
    print(panel,len(idx),f"{time.time()-t0:.0f}s",flush=True)
X=np.vstack(Xs); w=np.concatenate(ws)
prm=dict(objective="regression",learning_rate=0.05,num_leaves=255,min_data_in_leaf=100,feature_fraction=float(sys.argv[6]) if len(sys.argv)>6 else 0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,num_threads=16,verbose=-1,seed=seed,force_row_wise=True,deterministic=True)
for nm,y,nr in (("speed",np.concatenate(yv),rv),("flow",np.concatenate(yq),rq)):
    m=lgb.train(prm,lgb.Dataset(X,y,weight=w,categorical_feature=[0]),nr); m.save_model(str(OUT/f"{nm}_{tag}.txt")); print(nm,"done",f"{time.time()-t0:.0f}s",flush=True)
