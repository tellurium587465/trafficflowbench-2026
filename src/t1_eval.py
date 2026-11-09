"""Feb-only evaluation of a trained Task 1 model pair: S_state and physics proxy restricted to Feb days."""
import sys, numpy as np, lightgbm as lgb
from pathlib import Path
from cachelib import load
from common import PANELS
from t1_features import grid_features, rows
from t1_train import base_of
from metrics import state_score, physics_score, family_mean
MD=Path(r"C:/Users/user/tfb/work/t1")
def feb_eval(mv,mq,post=None):
    res={}
    for pi,panel in enumerate(PANELS):
        P=load(panel,"train"); F,_,_,_=grid_features(P); bv,bq=base_of(F)
        feb=np.array([s.startswith("2031-02") for s in P.times])
        r,c=np.nonzero(P.target&feb[:,None])
        X,_=rows(P,F,r,c); X=np.column_stack([np.full(len(r),pi,np.float32),X])
        Vp=np.full((P.T,P.n),np.nan); Qp=Vp.copy()
        Vp[r,c]=(bv[r,c]+mv.predict(X,num_threads=16))*P.vf[c]; Qp[r,c]=(bq[r,c]+mq.predict(X,num_threads=16))*P.lanes[c]
        if post: Vp,Qp=post(P,Vp,Qp,r,c)
        tg,rg=P.target.copy(),P.regime.copy()
        P.target=P.target&feb[:,None]; P.regime=np.where(feb,P.regime,255).astype(np.uint8)
        s=state_score(P,Vp,Qp); ph,det=physics_score(P,Vp,Qp,detail=True)
        P.target,P.regime=tg,rg
        res[panel]=(s,ph)
        print(f"{panel}: S_state {s:.4f} phys(FD,LWRp,S) {np.round(ph,4)} LWR by regime {[round(b,3) for a,b in det]}",flush=True)
    print("family mean: S_state %.5f  S_phys_proxy %.5f  S_LWR_proxy %.5f"%(family_mean({p:v[0] for p,v in res.items()}),family_mean({p:v[1][2] for p,v in res.items()}),family_mean({p:v[1][1] for p,v in res.items()})))
    return res
if __name__=="__main__":
    mv=lgb.Booster(model_file=str(MD/"speed.txt")); mq=lgb.Booster(model_file=str(MD/"flow.txt"))
    feb_eval(mv,mq)
