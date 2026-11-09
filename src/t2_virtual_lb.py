"""Virtual leaderboards: rerun the selector replica from many start days; each run = one LB-like
month (5 onset + 5 ongoing per panel). Score OOF predictions on each -> distribution of S_queue."""
import sys, pickle, numpy as np
import t2_fast as T
from cachelib import load, T2_PANELS
from t2_selector import scan
from t2_windows import queue_matrix, score
from t2_model import best_set
tag=sys.argv[1]
oof=pickle.load(open(rf"C:/Users/user/tfb/work/t2/oof_{tag}.pkl","rb"))
key={(T.META["panel"][k],int(T.META["t"][k])):k for k in oof}
runs={}
for panel in T2_PANELS:
    P=load(panel,"train"); Qm=queue_matrix(P)
    for d0 in range(0,P.T//288,7):
        on,og=scan(P,Qm,start=d0*288+12)
        runs.setdefault(d0,[]).extend([(panel,t,"queue_onset") for t in on]+[(panel,t,"queue_ongoing") for t in og])
scores=[]
for d0,ws in sorted(runs.items()):
    W=[];preds=[]
    for panel,t,cond in ws:
        k=key.get((panel,t))
        if k is None: continue
        lab,el=T.LAB[k]; o=oof[k]
        W.append(dict(panel=panel,cond=cond,label=lab,elig=el)); preds.append(best_set(o.ravel(),np.ones(o.size,bool)).reshape(o.shape))
    if len(W)>=60: scores.append(score(W,preds))
s=np.array(scores)
print(f"{tag}: virtual LBs n={len(s)}  S_queue mean {s[:,0].mean():.4f}  sd {s[:,0].std():.4f}  [p10 {np.percentile(s[:,0],10):.4f}, p90 {np.percentile(s[:,0],90):.4f}]  onset {s[:,1].mean():.4f}±{s[:,1].std():.4f}  ongoing {s[:,2].mean():.4f}±{s[:,2].std():.4f}")
