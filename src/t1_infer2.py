"""Predict Task 1 (validation + private) with a saved model pair from t1_exp (feature version fv)."""
import os, sys, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from cachelib import load
from common import PANELS, WORK
from t1_exp import featfns
MD=Path(WORK) / "t1"
fv=int(sys.argv[1]); tag=sys.argv[2]
gf,rf,bf=featfns(fv)
mv=lgb.Booster(model_file=str(MD/f"speed_{tag}.txt")); mq=lgb.Booster(model_file=str(MD/f"flow_{tag}.txt"))
parts=[]
for split in ("validation","private"):
    for pi,panel in enumerate(PANELS):
        P=load(panel,split); F,_,_,_=gf(P); bv,bq=bf(F)
        r,c=np.nonzero(P.target)
        X,_=rf(P,F,r,c); X=np.column_stack([np.full(len(r),0 if os.environ.get('TFB_T1_NOMEM','0')=='1' else pi,np.float32),X])
        v=(bv[r,c]+mv.predict(X,num_threads=16))*P.vf[c]; q=(bq[r,c]+mq.predict(X,num_threads=16))*P.lanes[c]
        v=np.clip(v,3.0,1.15*P.vf[c]); q=np.clip(q,60.0,None)
        parts.append(pd.DataFrame({"panel":panel,"timestamp":P.times[r],"link_id":P.links[c],"mask_regime":np.array(["R1","R2","R3"])[P.regime[r]],"speed_kmh":v,"flow_vph":q}))
        print(panel,split,len(r),flush=True)
out=pd.concat(parts,ignore_index=True); import os; out.to_parquet(MD/f"state_{tag}{os.environ.get('TFB_T1_OUTSUF','')}.parquet",index=False); print("rows",len(out))
