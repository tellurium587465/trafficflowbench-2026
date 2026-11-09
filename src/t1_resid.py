"""Where does the S_LWR proxy residual come from? Split by traffic state of the transition."""
import numpy as np, lightgbm as lgb, sys
from pathlib import Path
from cachelib import load
from common import PANELS
from t1_exp import featfns
MD=Path(r"C:/Users/user/tfb/work/t1")
def decompose(panel,mv,mq,fv=2):
    gf,rf,bf=featfns(fv); pi=PANELS.index(panel)
    P=load(panel,"train"); F,_,_,_=gf(P); bv,bq=bf(F)
    feb=np.array([s.startswith("2031-02") for s in P.times])
    r,c=np.nonzero(P.target&feb[:,None])
    X,_=rf(P,F,r,c); X=np.column_stack([np.full(len(r),pi,np.float32),X])
    Vp=P.V.copy(); Qp=P.Q.copy()
    Vp[r,c]=(bv[r,c]+mv.predict(X,num_threads=16))*P.vf[c]; Qp[r,c]=(bq[r,c]+mq.predict(X,num_threads=16))*P.lanes[c]
    on_v,off_v,_,_=P.ramp_validity()
    field=P.E&np.isfinite(P.V)&feb[:,None]
    N=np.where(field,Qp/np.maximum(Vp,1)*P.length[None,:],np.nan); Nt=np.where(field,P.Q/np.maximum(P.V,1)*P.length[None,:],np.nan)
    dN=N[1:]-N[:-1]; dNt=Nt[1:]-Nt[:-1]
    sameday=np.array([P.times[i][:10]==P.times[i+1][:10] for i in range(P.T-1)])
    tr=np.isfinite(dN)&np.isfinite(dNt)&on_v[:-1]&off_v[:-1]
    res=np.abs(dN-dNt); den=np.abs(dNt)
    vr=np.minimum(P.V[1:],P.V[:-1])/P.vf[None,:]
    cong=vr<0.9
    tgt=P.target[1:]|P.target[:-1]
    for reg in (0,1,2):
        m=tr&(P.regime[1:]==reg)[:,None]
        tot=res[m].sum(); D=den[m].sum()
        print(f"{panel} R{reg+1}: E={tot/D:.3f}  residual share from congested transitions {res[m&cong].sum()/tot:.2f} (they are {(m&cong).sum()/m.sum():.3f} of transitions, {den[m&cong].sum()/D:.2f} of denominator); free-transition E={res[m&~cong].sum()/D:.3f}")
if __name__=="__main__":
    tag=sys.argv[1]
    mv=lgb.Booster(model_file=str(MD/f"speed_{tag}.txt")); mq=lgb.Booster(model_file=str(MD/f"flow_{tag}.txt"))
    for p in sys.argv[2].split(","): decompose(p,mv,mq)
