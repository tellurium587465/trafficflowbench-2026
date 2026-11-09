"""Far ongoing predictions gated by same-month causal evidence (10/10, knowledge-map gap T2 x pre-T same-month data).
t2_far_suppress drops every ongoing prediction more than D links from a queue visible at T (LB +0.0031, but -0.061 on
train weekdays, and the private month is weekday-heavy). Here a far prediction is kept when a queue was observed at that
link (+-2) earlier the same day (05:00 .. T-5) or in the same time slot (T-90 .. T+60 min) on up to 3 earlier days of the
same month; only the far predictions without such evidence are dropped. Uses observations before T only (causal);
hidden cells (Task 1 masks, Task 2 blanks) simply give no evidence. On the train months this keeps 70% of the far
positives and loses -0.011 instead of -0.044 (tfb/review/g3_sameday_gate.py).

usage: python t2_far_gate.py <in_queue_parquet> <out_queue_parquet> [D=3] [CONTIG=0]
       CONTIG=1 (10/11, G26): a far positive whose run of consecutive predicted links (same timestamp) touches the
       near region is contiguous growth of the visible queue (LWR: the congested region grows as one block), not a
       detached new queue, and is kept without evidence. Locally the gate removes 780 such cells, 88% right.
       env TFB_T2_TEST=<nn_test dir> (10/22 E6, t2_window_entry.py): the windows' features (last_queue) come from there
       env TFB_T2_LQT=1 (11/03 ledger #197): a link observed at the row T takes its queue state there as the last queue (the
       near region); the networks' inputs are unchanged (745700: the row at T may be read)
       env TFB_T2_GATE2=1 (11/01 ledger #146, needs CONTIG=1): the dropped cells get p = 0 in the probabilities best_set saw (pbog_<name>.pkl from t2_onset_evidence.py), best_set runs again and the gate again
"""
import os, sys, pickle, numpy as np, pandas as pd
from cachelib import load, asof
from t2_model import best_set
from t2_dex import DEX; OGDEX = os.environ.get("TFB_T2_OGDEX", "0") == "1"   # 11/04 ledger #216: the re-decode after the gate by the exact expected IoU (B12)
from common import NN                                        # 10/22: data root off the failing E: (TFB_NN)

src, dst = sys.argv[1], sys.argv[2]; D = int(sys.argv[3]) if len(sys.argv) > 3 else 3
CONTIG = int(sys.argv[4]) if len(sys.argv) > 4 else 0
LQT = os.environ.get("TFB_T2_LQT") == "1"                   # 11/03 ledger #197
names = list(np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)); ci = names.index("last_queue")
d = os.environ.get("TFB_T2_TEST", f"{NN}/nn_test_afdem").rstrip("/") + "/"
X = np.load(d + "X.npy", mmap_mode="r"); off = np.load(d + "off.npy")
wid = np.load(d + "window_id.npy", allow_pickle=True).astype(str); cond = np.load(d + "cond.npy", allow_pickle=True).astype(str)
q = pd.read_parquet(src); g = dict(tuple(q.groupby("window_id", sort=False)))
assert set(g) == set(wid), "queue windows differ from the test windows"
PC = {}; drop = np.zeros(len(q), bool); n_far = n_kept = n_contig = 0; PBG = None; n_re = 0
if os.environ.get("TFB_T2_GATE2") == "1":
    PBG = pickle.load(open(src.replace("queue_", "pbog_").replace(".parquet", ".pkl"), "rb"))


def contig(M, near):
    C = np.zeros_like(M)
    for r_ in range(M.shape[0]):
        j = 0
        while j < M.shape[1]:
            if M[r_, j]:
                e = j
                while e + 1 < M.shape[1] and M[r_, e + 1]:
                    e += 1
                if near[j:e + 1].any():
                    C[r_, j:e + 1] = True
                j = e + 1
            else:
                j += 1
    return C


for i, w in enumerate(wid):
    if cond[i] != "queue_ongoing":
        continue
    split = "validation" if "_validation_" in w else "private"
    panel = w[3:w.index("_" + split + "_")]
    if (panel, split) not in PC:
        P = load(panel, split); PC[(panel, split)] = (P, {s: j for j, s in enumerate(P.times)})   # 10/29 audit: no month-wide qobs
    P, tix = PC[(panel, split)]
    sub = g[w]; ts = sorted(sub.timestamp.unique()); links = sub[sub.timestamp == ts[0]].link_id.to_numpy()
    assert len(links) == off[i + 1] - off[i] == P.n and (links == np.asarray(P.links)).all(), ("link order", w)
    t = tix[(pd.Timestamp(ts[0]) - pd.Timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")]   # index of T
    Pt = asof(P, t); qobs = (Pt.V <= Pt.vcut[None, :]) & np.isfinite(Pt.V)   # 10/22 E6: read boundary, rows < T only
    d0 = t - (t % 288) + 60
    seen = qobs[d0:t].any(axis=0) if t > d0 else np.zeros(P.n, bool)
    for back in (1, 2, 3):
        dd = t - back * 288
        if dd - 18 >= 0:                                           # the split starts on the 1st: earlier days are this month
            seen = seen | qobs[dd - 18:dd + 12].any(axis=0)
    ev = seen.copy()
    for o in (1, 2):
        ev[o:] |= seen[:-o]; ev[:-o] |= seen[o:]
    lq = np.asarray(X[off[i]:off[i + 1], ci], dtype=float) == 1; near = np.zeros(len(links), bool)
    if LQT:                                                    # 11/03 ledger #197: the row at T where it is observed
        vt = np.asarray(P.V[t], float); lq = np.where(np.isfinite(vt), vt <= P.vcut, lq)
    for j in np.flatnonzero(lq):
        near[max(0, j - D):j + D + 1] = True
    far = ~near
    pos = sub.queue_pred.to_numpy() == 1; lk = pd.Index(links).get_indexer(sub.link_id)
    fp = pos & far[lk]; n_far += int(fp.sum()); keep = fp & ev[lk]; n_kept += int(keep.sum())
    if CONTIG:
        ti = pd.Index(ts).get_indexer(sub.timestamp); M = np.zeros((len(ts), len(links)), bool); M[ti[pos], lk[pos]] = True
        C = contig(M, near)
        cg = fp & ~ev[lk] & C[ti, lk]; n_contig += int(cg.sum()); fp = fp & ~cg
    drop[sub.index[fp & ~ev[lk]]] = True
    if PBG is not None and (fp & ~ev[lk]).any():
        gone = np.zeros(M.shape, bool); g_ = fp & ~ev[lk]; gone[ti[g_], lk[g_]] = True; pb = np.where(gone, 0.0, PBG[w]); M2 = DEX(pb) if OGDEX else best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape)
        M2 = M2 & ~(M2 & far[None, :] & ~ev[None, :] & ~contig(M2, near)); q.loc[sub.index, "queue_pred"] = M2[ti, lk].astype(q.queue_pred.dtype); drop[sub.index] = False; n_re += 1
q.loc[drop, "queue_pred"] = 0
q.to_parquet(dst, index=False)
print(f"gate before best_set: {n_re} windows decoded again" if PBG is not None else "", flush=True)
print(f"D={D} CONTIG={CONTIG}: far ongoing positives {n_far}, kept by same-month evidence {n_kept}, kept as contiguous growth {n_contig}, removed {int(drop.sum())} -> {dst}")
