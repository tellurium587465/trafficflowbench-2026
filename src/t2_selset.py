"""Selector-like evaluation windows: run the chronological selector replica from the start of every
7th day of train, union the windows. Approximates the LB/private window distribution."""
import pickle, numpy as np
from cachelib import load, T2_PANELS
from t2_selector import scan
from t2_windows import queue_matrix, F
out=[]; seen=set()
for panel in T2_PANELS:
    P=load(panel,"train"); Qm=queue_matrix(P)
    for d0 in range(0, P.T//288, 7):
        on,og=scan(P,Qm,start=d0*288+12)
        for cond,ts in (("queue_onset",on),("queue_ongoing",og)):
            for t in ts:
                if (panel,t) in seen: continue
                seen.add((panel,t))
                out.append(dict(panel=panel,t=t,cond=cond,date=P.times[t][:10],label=Qm[t+1:t+F+1].copy(),elig=P.E[t+1:t+F+1].copy()))
    print(panel,len(out),flush=True)
pickle.dump(out,open(r"C:/Users/user/tfb/work/t2_windows_sel.pkl","wb"))
print("selector-like windows",len(out),sum(w["cond"]=="queue_onset" for w in out))
