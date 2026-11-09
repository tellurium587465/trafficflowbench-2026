"""Shift-weighted evaluation: score OOF predictions per (panel, month) and weight months by how far the
month's bottleneck mix (first-queue-of-day cluster distribution) is from the panel's train average.
Reports plain mean and the mean over the most-shifted (panel, month) cells."""
import sys, pickle, numpy as np, pandas as pd
from t2_eval_oof import T, load_oof   # metadata + labels only; blend specs allowed
from cachelib import load, T2_PANELS
from t2_windows import queue_matrix, score
from t2_model import best_set
tags=sys.argv[1].split(",")
# month shift statistic per panel-month: total-variation distance of first-queue cluster distribution vs panel mean
shift={}
for p in T2_PANELS:
    P=load(p,"train"); q=queue_matrix(P); D=P.T//288; rec=[]
    for d in range(D):
        sl=q[d*288:(d+1)*288]; rows=np.flatnonzero(sl.any(axis=1))
        if len(rows): rec.append((P.day[d][:7],int(np.median(np.flatnonzero(sl[rows[0]])))//10))
    df=pd.DataFrame(rec,columns=["m","b"]); ct=pd.crosstab(df.m,df.b,normalize="index"); mean=ct.mean()
    for m,row in ct.iterrows(): shift[(p,m)]=0.5*np.abs(row-mean).sum()
cut=np.quantile(list(shift.values()),0.75)
for tag in tags:
    oof=load_oof(tag)
    per={}
    for k,o in oof.items():
        if not T.META["is_sel"][k]: continue
        p=T.META["panel"][k]; m=T.META["date"][k][:7]; lab,el=T.LAB[k]
        s=best_set(o.ravel(),np.ones(o.size,bool)).reshape(o.shape)
        iou=((s&lab&el).sum())/max(((s|lab)&el).sum(),1)
        per.setdefault((p,m,T.META["cond"][k]),[]).append(iou)
    rows=[(p,m,c,np.mean(v),shift[(p,m)]) for (p,m,c),v in per.items()]
    d=pd.DataFrame(rows,columns=["p","m","c","iou","shift"])
    hi=d[d["shift"]>=cut]
    print(f"{tag:16s} all: onset {d[d.c=='queue_onset'].iou.mean():.4f} ongoing {d[d.c=='queue_ongoing'].iou.mean():.4f} | most-shifted 25% months: onset {hi[hi.c=='queue_onset'].iou.mean():.4f} ongoing {hi[hi.c=='queue_ongoing'].iou.mean():.4f} (n panel-months {hi[['p','m']].drop_duplicates().shape[0]})")
