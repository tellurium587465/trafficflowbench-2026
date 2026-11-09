"""Feb evaluation of post-processing variants on f3big predictions (S_state + physics proxy)."""
import numpy as np, lightgbm as lgb
from pathlib import Path
from cachelib import load
from common import PANELS
from t1_exp import featfns
from metrics import state_score, physics_score
MD=Path(r"C:/Users/user/tfb/work/t1")
gf,rf,bf=featfns(3)
mv=lgb.Booster(model_file=str(MD/"speed_f3big.txt")); mq=lgb.Booster(model_file=str(MD/"flow_f3big.txt"))
def variants(P,V,Q,c):
    out={"raw":(V,Q)}
    vf=P.vf[c]; w=P.w[c]; kj=P.kj[c]
    qfd=w*kj*V/(V+w)                       # congested-branch flow implied by speed
    cong=V<0.6*vf
    for a in (0.5,1.0):
        out[f"fdq{a}"]=(V,np.where(cong,(1-a)*Q+a*qfd,Q))
    # speed implied by flow on congested branch: k=kj-q/w, v=q/k
    k=np.maximum(kj-Q/w,1e-3); vfd=Q/k
    for a in (0.5,0.75,1.0):
        out[f"fdv{a}"]=(np.where(cong,(1-a)*V+a*vfd,V),Q)
    for th in (0.5,0.7,0.8):
        cg=V<th*vf; out[f"fdv0.75_th{th}"]=(np.where(cg,0.25*V+0.75*vfd,V),Q)
    return out
tot={}
for pi,panel in enumerate(PANELS):
    P=load(panel,"train"); F,_,_,_=gf(P); bv,bq=bf(F)
    feb=np.array([s.startswith("2031-02") for s in P.times]); r,c=np.nonzero(P.target&feb[:,None])
    X,_=rf(P,F,r,c); X=np.column_stack([np.full(len(r),pi,np.float32),X])
    V=(bv[r,c]+mv.predict(X,num_threads=16))*P.vf[c]; Q=(bq[r,c]+mq.predict(X,num_threads=16))*P.lanes[c]
    tg,rg=P.target.copy(),P.regime.copy(); P.target=P.target&feb[:,None]; P.regime=np.where(feb,P.regime,255).astype(np.uint8)
    for nm,(v,q) in variants(P,V,Q,c).items():
        Vp=np.full((P.T,P.n),np.nan); Qp=Vp.copy(); Vp[r,c]=v; Qp[r,c]=q
        s=state_score(P,Vp,Qp); ph=physics_score(P,Vp,Qp)
        tot.setdefault(nm,[]).append((s,ph[2],ph[1]))
    P.target,P.regime=tg,rg
    print(panel,{k:np.round(v[-1],4).tolist() for k,v in tot.items()},flush=True)
for k,v in tot.items():
    a=np.array(v); print(f"{k:8s} S_state {a[:,0].mean():.5f}  S_phys_proxy {a[:,1].mean():.5f}  S_LWR_proxy {a[:,2].mean():.5f}  total-equivalent {0.35*a[:,0].mean()+0.15*a[:,1].mean():.5f}")
