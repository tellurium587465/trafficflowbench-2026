"""Evaluate OOF pickles (and blends) on random eval windows, selector-like windows, virtual LBs and shift months.

Reads only window metadata + labels (not the 3.4 GB feature matrix) and caches the virtual-LB selector scans.
"""
import os, sys, pickle, types, numpy as np
from cachelib import load, T2_PANELS
from t2_windows import score, queue_matrix
from t2_model import best_set
from common import NN, WORK                                        # 10/22: data root off the failing E: (TFB_NN)
_C = os.environ.get("TFB_T2CACHE", "2_noT")
_Z = np.load(f"{WORK}/t2/featcache{_C}.npz", allow_pickle=True)
T = types.SimpleNamespace(META={k: _Z[k] for k in ("panel", "t", "cond", "date", "is_eval")},
                          LAB=pickle.load(open(f"{WORK}/t2/labels_all{_C}.pkl", "rb")))
T.NWIN = len(T.META["panel"])
T.META["is_sel"] = _Z["is_sel"] if "is_sel" in _Z.files else np.zeros(T.NWIN, bool)
_RUNS = f"{WORK}/t2/vlb_runs{_C}.pkl"
KEEP = set()                                    # tags kept in memory across specs (e.g. the LightGBM reference)
_OOF = {}
FAR = int(os.environ.get("TFB_T2_FAR", "0"))    # 10/09: score ongoing after dropping predictions > FAR links from a queue at T
_FARM = {}


def far_mask(k):
    """bool (n,) of links farther than FAR from any queue visible at T, for ongoing windows (None otherwise);
    the same rule as t2_far_suppress.py on the submission."""
    if not FAR or T.META["cond"][k] != "queue_ongoing":
        return None
    if not _FARM:
        f = f"{NN}/nn_data_afdem_lastq.npz"
        if not os.path.exists(f):
            names = list(np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True))
            X = np.load(f"{NN}/nn_data_afdem/X.npy", mmap_mode="r")
            np.savez(f, lq=np.asarray(X[:, names.index("last_queue")]) == 1, off=np.load(f"{NN}/nn_data_afdem/off.npy"))
        z = np.load(f); _FARM["lq"], _FARM["off"] = z["lq"], z["off"]
        assert len(_FARM["off"]) - 1 == T.NWIN, "nn_data windows differ from the evaluator windows"
    lq = _FARM["lq"][_FARM["off"][k]:_FARM["off"][k + 1]]; near = np.zeros(len(lq), bool)
    for j in np.flatnonzero(lq):
        near[max(0, j - FAR):j + FAR + 1] = True
    return ~near


def oof_of(t):
    if t in _OOF:
        return _OOF[t]
    o = pickle.load(open(f"{WORK}/t2/oof_{t}.pkl", "rb"))
    if t in KEEP:
        _OOF[t] = o
    return o


def load_oof(spec):
    parts=[(float(w),oof_of(t)) for t,w in (x.split(":") if ":" in x else (x,"1") for x in spec.split("+"))]
    keys=set.intersection(*[set(o) for _,o in parts]); tw=sum(w for w,_ in parts)
    return {k:sum(w*o[k] for w,o in parts)/tw for k in keys}


if os.path.exists(_RUNS):
    runs=pickle.load(open(_RUNS,"rb"))
else:
    from t2_selector import scan
    runs={}
    for p in T2_PANELS:
        P=load(p,"train"); Qm=queue_matrix(P)
        for d0 in range(0,P.T//288,7):
            on,og=scan(P,Qm,start=d0*288+12); runs.setdefault(d0,[]).extend([(p,t) for t in on+og])
    pickle.dump(runs,open(_RUNS,"wb"))
key={(T.META["panel"][k],int(T.META["t"][k])):k for k in range(T.NWIN) if T.META["is_sel"][k]}


def _W(k):
    return dict(panel=T.META["panel"][k],cond=T.META["cond"][k],label=T.LAB[k][0],elig=T.LAB[k][1])


def evaluate(spec):
    """-> eval[3], sel[3] (all/onset/ongoing), virtual LB mean (all, onset, ongoing)."""
    oof=load_oof(spec)
    bs={k:best_set(o.ravel(),np.ones(o.size,bool)).reshape(o.shape) for k,o in oof.items()}   # once per window
    if FAR:
        for k in bs:
            m=far_mask(k)
            if m is not None: bs[k]=bs[k]&~m[None,:]
    out=[]
    for kk in ("is_eval","is_sel"):
        idx=[k for k in oof if T.META[kk][k]]
        out.append(np.round(score([_W(k) for k in idx],[bs[k] for k in idx]),4))
    vs=[]
    for d0,ws in runs.items():
        ks=[key[x] for x in ws if x in key and key[x] in oof]
        if len(ks)<60: continue
        vs.append(score([_W(k) for k in ks],[bs[k] for k in ks]))
    vs=np.array(vs)
    return out[0], out[1], (vs.mean(0) if len(vs) else np.full(3, np.nan))   # condition-only OOFs have no full weeks


def fmt(spec, r):
    e, s, v = r
    return f"{spec:28s} eval {e}  sel {s}  virtualLB {v[0]:.4f} (onset {v[1]:.4f} ongoing {v[2]:.4f})"


if __name__ == "__main__":
    for spec in sys.argv[1:]:
        print(fmt(spec, evaluate(spec)), flush=True)