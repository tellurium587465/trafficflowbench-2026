"""Extend the Task 2 feature cache with selector-like windows (flag is_sel)."""
import pickle, time, os, numpy as np
SUF = "_noT" if os.environ.get("TFB_T2_NOT", "0") == "1" else ""
from cachelib import load, T2_PANELS
from t2_model import PanelInputs
from t2_cache import window_X
Z=np.load(rf"C:/Users/user/tfb/work/t2/featcache{SUF}.npz",allow_pickle=True)
LAB=pickle.load(open(rf"C:/Users/user/tfb/work/t2/labels_all{SUF}.pkl","rb"))
S=pickle.load(open(r"C:/Users/user/tfb/work/t2_windows_sel.pkl","rb"))
PIs={p:PanelInputs(load(p,"train")) for p in T2_PANELS}
Xs=[Z["X"]]; off=list(Z["off"]); 
for w in S:
    X,_=window_X(PIs[w["panel"]],w["t"],w["cond"]); Xs.append(X); off.append(off[-1]+len(X))
n0=len(Z["off"])-1
meta={k:np.concatenate([Z[k],np.array([w[k if k!="is_eval" else "cond"] for w in S]) if k!="is_eval" else np.zeros(len(S),bool)]) for k in ("panel","t","cond","date","is_eval")}
meta["is_sel"]=np.concatenate([np.zeros(n0,bool),np.ones(len(S),bool)])
np.savez(rf"C:/Users/user/tfb/work/t2/featcache2{SUF}.npz",X=np.vstack(Xs),off=np.array(off),names=Z["names"],**meta)
pickle.dump(LAB+[(w["label"],w["elig"]) for w in S],open(rf"C:/Users/user/tfb/work/t2/labels_all2{SUF}.pkl","wb"))
print("windows",len(off)-1,"sel",len(S))
