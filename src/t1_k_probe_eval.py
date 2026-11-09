"""Proxy effect (Feb) of diagnostic T1 transforms: k-noise injection and temporal smoothing of k at target cells."""
import numpy as np, lightgbm as lgb
from pathlib import Path
from cachelib import load
from common import PANELS
from t1_exp import featfns
from metrics import state_score, physics_score
MD=Path(r"C:/Users/user/tfb/work/t1"); gf,rf,bf=featfns(3)
mv=lgb.Booster(model_file=str(MD/"speed_f3big.txt")); mq=lgb.Booster(model_file=str(MD/"flow_f3big.txt"))
rng=np.random.default_rng(7); tot={}
def smooth_k(P,Vp,Qp,tgt):
    """3-point temporal average of k (pred at targets, obs elsewhere), applied only at target cells by rescaling q."""
    K=np.where(tgt,Qp/np.maximum(Vp,1),P.Q/np.maximum(P.V,1))
    Ks=np.copy(K); num=np.nan_to_num(K); den=np.isfinite(K).astype(float)
    for s in (1,-1):
        sh=np.roll(K,s,axis=0); m=np.isfinite(sh); num+=np.where(m,sh,0); den+=m
    Ks=num/np.maximum(den,1)
    Q2=np.where(tgt,Ks*np.maximum(Vp,1),Qp); return Vp,Q2
for pi,panel in enumerate(PANELS):
    P=load(panel,"train"); F,_,_,_=gf(P); bv,bq=bf(F)
    feb=np.array([s.startswith("2031-02") for s in P.times]); r,c=np.nonzero(P.target&feb[:,None])
    X,_=rf(P,F,r,c); X=np.column_stack([np.full(len(r),pi,np.float32),X])
    V=(bv[r,c]+mv.predict(X,num_threads=16))*P.vf[c]; Q=(bq[r,c]+mq.predict(X,num_threads=16))*P.lanes[c]
    tg,rg=P.target.copy(),P.regime.copy(); P.target=P.target&feb[:,None]; P.regime=np.where(feb,P.regime,255).astype(np.uint8)
    Vp=np.full((P.T,P.n),np.nan); Qp=Vp.copy(); Vp[r,c]=V; Qp[r,c]=Q
    var={"raw":(Vp,Qp)}
    Qn=Qp.copy(); Qn[r,c]=Q*(1+0.024*rng.standard_normal(len(r))); var["knoise2.4"]=(Vp,Qn)
    var["ksmooth3"]=smooth_k(P,Vp,Qp,P.target)
    for nm,(a,b) in var.items():
        s=state_score(P,a,b); ph=physics_score(P,a,b); tot.setdefault(nm,[]).append((s,ph[2],ph[1]))
    P.target,P.regime=tg,rg
    print(panel,{k:np.round(v[-1],4).tolist() for k,v in tot.items()},flush=True)
for k,v in tot.items():
    a=np.array(v); print(f"{k:10s} S_state {a[:,0].mean():.5f}  S_phys_proxy {a[:,1].mean():.5f}  S_LWR {a[:,2].mean():.5f}")
