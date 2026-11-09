"""Onset-only CV experiments on the feature cache (fast)."""
import sys, numpy as np, lightgbm as lgb, pickle
import t2_fast as T
from cachelib import load, T2_PANELS
from t2_windows import score, F
from t2_model import best_set, FOLDS, PARAMS
from functools import lru_cache

@lru_cache(None)
def profile(panel):
    P=load(panel,"train"); tod=(np.arange(P.T)%288)//6  # 30-min bins
    qc=np.where(P.E,P.Q/P.cap[None,:],np.nan); vr=np.where(P.E,P.V/P.vf[None,:],np.nan)
    pq=np.stack([np.nanmean(qc[tod==b],axis=0) for b in range(48)]); pv=np.stack([np.nanmean(vr[tod==b],axis=0) for b in range(48)])
    # probability a link is queued at this tod
    qq=np.where(P.E,(np.nan_to_num(P.V,nan=1e9)<=P.vcut[None,:]).astype(float),np.nan)
    pz=np.stack([np.nanmean(qq[tod==b],axis=0) for b in range(48)])
    return pq,pv,pz

I_QT=T.NAMES.index("q_T"); I_QM3=T.NAMES.index("q_m3")
def extra(k,X):
    panel=T.META["panel"][k]; t=T.META["t"][k]; pq,pv,pz=profile(panel)
    b=(t%288)//6; b6=((t+6)%288)//6
    with np.errstate(all="ignore"):
        return np.column_stack([X, pq[b], X[:,I_QT]/pq[b], X[:,I_QM3]/pq[b], pz[b6], pq[b6]]).astype(np.float32)

def run(params, rounds, use_extra_feats, tag):
    prm=dict(PARAMS); prm.update(params)
    eval_idx=np.flatnonzero((T.META["is_eval"]|T.META["is_sel"])&(T.META["cond"]=="queue_onset"))
    mon=np.array([d[:7] for d in T.META["date"]]); oof={}
    for months in FOLDS:
        pr=T.priors(set(months))
        tr=np.flatnonzero(~np.isin(mon,months)&~T.META["is_sel"]&(T.META["cond"]=="queue_onset"))
        Xs,ys,ws=[],[],[]
        for k in tr:
            X=T.win_X(k,pr); lab,el=T.LAB[k]
            if use_extra_feats: X=extra(k,X)
            m=el[-1]; Xs.append(X[m]); ys.append(lab[-1][m]); ws.append(np.full(m.sum(),100.0/max((lab&el).sum(),3)))
        mdl=lgb.train(prm,lgb.Dataset(np.vstack(Xs),np.concatenate(ys).astype(np.float32),weight=np.concatenate(ws)),rounds)
        for k in eval_idx:
            if mon[k] in months:
                X=T.win_X(k,pr); X=extra(k,X) if use_extra_feats else X
                p=np.zeros((F,len(X))); p[-1]=mdl.predict(X,num_threads=16); oof[k]=p
    for key in ("is_eval","is_sel"):
        idx=[k for k in eval_idx if T.META[key][k]]
        Wv=[dict(panel=T.META["panel"][k],cond="queue_onset",label=T.LAB[k][0],elig=T.LAB[k][1]) for k in idx]
        preds=[best_set(oof[k].ravel(),T.LAB[k][1].ravel()).reshape(oof[k].shape) for k in idx]
        print(tag,key,"onset IoU:",round(score(Wv,preds)[1],4),flush=True)

if __name__=="__main__":
    run(dict(num_leaves=63),600,False,"base")
    run(dict(num_leaves=63,min_data_in_leaf=20),600,False,"mdl20")
    run(dict(num_leaves=63,min_data_in_leaf=20),600,True,"mdl20+prof")
    run(dict(num_leaves=31,min_data_in_leaf=20,learning_rate=0.03),1200,True,"l31+prof")
