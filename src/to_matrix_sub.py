"""Convert a per-task state submission CSV into (Vp, Qp) matrices on a cached panel grid."""
import numpy as np, pandas as pd
def sub_to_mats(P, df):
    ti={t:i for i,t in enumerate(P.times)}; li={l:i for i,l in enumerate(P.links)}
    d=df[df.panel==P.panel]
    r=d.timestamp.map(ti).to_numpy(); c=d.link_id.astype(str).map(li).to_numpy()
    Vp=np.full((P.T,P.n),np.nan); Qp=Vp.copy()
    Vp[r,c]=d.speed_kmh.to_numpy(); Qp[r,c]=d.flow_vph.to_numpy()
    return Vp,Qp
