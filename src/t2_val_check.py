"""Compare our val/private queue predictions with persistence and window-history facts."""
import numpy as np, pandas as pd
from cachelib import load, T2_PANELS
from common import REL
from t2_windows import F, queue_matrix
q=pd.read_parquet(r"C:/Users/user/tfb/work/t2/queue_v8.parquet")
rows=[]
for split in ("train","validation","private"):
    for panel in T2_PANELS:
        P=load(panel,split); ti={s:i for i,s in enumerate(P.times)}; li={l:i for i,l in enumerate(P.links)}
        wi=pd.read_csv(REL/"task2"/panel/split/"window_index.csv")
        h=pd.read_parquet(REL/"task2"/panel/split/"window_history.parquet")
        for r in wi.itertuples():
            t=ti[r.forecast_origin]
            hw=h[h.window_id==r.window_id]
            last=hw[hw.timestamp==hw.timestamp.max()]
            lastq=np.zeros(P.n,bool); m=last.is_score_eligible.astype(bool)&(last.speed_kmh<=P.vcut[last.link_id.map(li).values])
            lastq[last.link_id.map(li).values[m.values]]=True
            hq=np.zeros((12,P.n),bool)
            for k,(ts,g) in enumerate(hw.groupby("timestamp")):
                mm=g.is_score_eligible.astype(bool)&(g.speed_kmh<=P.vcut[g.link_id.map(li).values])
                hq[k,g.link_id.map(li).values[mm.values]]=True
            if split=="train":
                pred=None
            else:
                x=q[q.window_id==r.window_id]
                pred=np.zeros((F,P.n),bool)
                for s,(ts,g) in enumerate(x.groupby("timestamp")): pred[s,g.link_id.map(li).values]=g.queue_pred.values.astype(bool)
            pers=np.broadcast_to(lastq,(F,P.n))
            if pred is not None:
                inter=(pred&pers).sum(); uni=(pred|pers).sum(); ov=inter/max(uni,1)
                npred=pred.sum(); npred6=pred[-1].sum()
            else: ov=npred=npred6=np.nan
            rows.append((split,panel,r.condition,r.forecast_origin[11:16],int(hq.sum()),int(hq.any(axis=0).sum()),int(lastq.sum()),npred,npred6,ov,round(r.history_coverage,3)))
d=pd.DataFrame(rows,columns=["split","panel","cond","T","hist_q_cells","hist_q_links","last_q_links","pred_cells","pred_T30","IoU(pred,persist)","hist_cov"])
pd.set_option("display.width",200)
print(d.groupby(["split","cond"])[["hist_q_cells","hist_q_links","last_q_links","pred_cells","pred_T30","IoU(pred,persist)","hist_cov"]].median())
print(d[(d.split=="validation")].head(20).to_string())
