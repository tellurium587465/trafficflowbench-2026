import pickle, numpy as np
import t2_fast as T
from t2_windows import score
from t2_model import best_set
s,oof=T.cv(params=dict(num_leaves=255,min_data_in_leaf=20,feature_fraction=0.6,seed=11,bagging_seed=11),rounds=(600,2000),tag="l255ff06",save=True)
o1=pickle.load(open(r"C:/Users/user/tfb/work/t2/oof_l255r2000.pkl","rb"))
for key in ("is_eval","is_sel"):
    idx=[k for k in o1 if T.META[key][k]]
    Wv=[dict(panel=T.META["panel"][k],cond=T.META["cond"][k],label=T.LAB[k][0],elig=T.LAB[k][1]) for k in idx]
    preds=[best_set(((o1[k]+oof[k])/2).ravel(),T.LAB[k][1].ravel()).reshape(o1[k].shape) for k in idx]
    print("ENS2",key,np.round(score(Wv,preds),4),flush=True)
