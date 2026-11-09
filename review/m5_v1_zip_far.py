"""M5 audit V1/V2: submission zips vs queue parquets, independent far-suppression re-implementation,
link order of nn_test X vs parquet vs links.csv, weekday of windows. Read-only; low memory."""
import sys, zipfile, hashlib, numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, "C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from common import REL

W = "C:/Users/user/tfb/work/"
SUBS = {
    "s32": ("s32_t1Adapt2GapMLClipFdBnd_t2s31_t4l2.zip", "queue_s31_t1AdaptGapMLClipFd_t2v14afAttnR13Drop_t4l2.parquet"),
    "s33": ("s33_t1Adapt2GapMLClipFdBnd_t2s31og0_t4l2.zip", "queue_s31og0_t1AdaptGapMLClipFd_t2v14afAttnR13Drop_t4l2.parquet"),
    "s34": ("s34_t1Adapt2GapMLClipFdBnd_t2s31far0_t4l2.zip", "queue_s31far0_t1AdaptGapMLClipFd_t2v14afAttnR13Drop_t4l2.parquet"),
    "s35": ("s35_t1Adapt2GapMLClipFdBnd_t2nomemFar_t4l2.zip", "queue_s35_t1Adapt2GapMLClipFdBnd_t2nomemFar_t4l2.parquet"),
    "s37": ("s37_t1Adapt2GapMLClipFdBnd_t2nomemNNonlyFar_t4l2.zip", "queue_s37_t1Adapt2GapMLClipFdBnd_t2nomemNNonlyFar_t4l2.parquet"),
    "s38": ("s38_t1Adapt2GapMLClipFdBnd_t2s31farWeekdayOnly_t4l2.zip", "queue_s31farWD_t2.parquet"),
}
key = pq.read_table(W + "submission_key.parquet", columns=["submission_id", "task", "window_id", "timestamp", "link_id"],
                    filters=[("task", "==", "queue")]).to_pandas()
print("key queue rows", len(key), "windows", key.window_id.nunique())

# ---------- V1: zip contents ----------
nonq_hash = {}
for s, (zf, qf) in SUBS.items():
    z = zipfile.ZipFile(W + "subs/" + zf); f = z.open("submission.csv")
    h = hashlib.sha1(); qrows = []; n = 0
    for ch in pd.read_csv(f, chunksize=1_000_000, dtype={"submission_id": np.int64, "task": str}):
        n += len(ch); m = ch.task == "queue"
        qrows.append(ch.loc[m, ["submission_id", "queue_pred"]])
        nq = ch.loc[~m, ["submission_id", "speed_kmh", "flow_vph", "path_flow"]]
        h.update(pd.util.hash_pandas_object(nq, index=False).to_numpy().tobytes())
    qz = pd.concat(qrows); nonq_hash[s] = h.hexdigest()[:12]
    j = key.merge(qz, on="submission_id", how="left")
    exp = pd.read_parquet(W + "t2/" + qf)
    j = j.merge(exp.rename(columns={"queue_pred": "exp"}), on=["window_id", "timestamp", "link_id"], how="left")
    print(f"{s}: rows {n}, queue rows {len(qz)}, missing {j.queue_pred.isna().sum()}, parquet-unmatched {j.exp.isna().sum()}, "
          f"mismatch {(j.queue_pred != j.exp).sum()}, positives zip {int(qz.queue_pred.sum())} parquet {int(exp.queue_pred.sum())}, "
          f"T1+T4 hash {nonq_hash[s]}")

# ---------- V2: independent far rule ----------
d = "E:/tfb_nn/nn_test_afdem/"
names = list(np.load("E:/tfb_nn/nn_data_afdem/names.npy", allow_pickle=True))
Xt = np.load(d + "X.npy", mmap_mode="r"); off = np.load(d + "off.npy")
wid = np.load(d + "window_id.npy", allow_pickle=True).astype(str); cond = np.load(d + "cond.npy", allow_pickle=True).astype(str)
ci, cvf, cpos, cdow, ctod, ccond = (names.index(c) for c in ("last_queue", "vf", "pos", "dow", "tod", "cond"))

# window index (forecast origin) and panel links
wi = {}
Pl = {}
for split in ("validation", "private"):
    for p in T2_PANELS:
        P = load(p, split); Pl[(p, split)] = (P.links, P.vf)
        for r in pd.read_csv(REL / "task2" / p / split / "window_index.csv").itertuples():
            wi[r.window_id] = (p, split, r.forecast_origin, r.condition)
print("window_index windows", len(wi), "nn_test windows", len(wid), "same set", set(wi) == set(wid))


def far_links(i, D=3):
    a, b = off[i], off[i + 1]
    lq = np.asarray(Xt[a:b, ci], dtype=np.float64) == 1
    near = np.zeros(b - a, bool)
    for j in np.flatnonzero(lq):
        near[max(0, j - D):j + D + 1] = True
    return ~near


bad_order = bad_vf = bad_cond = 0; nwk = {}; day_diff = 0; dow_diff = 0; noq = 0
info = {}
for i, w in enumerate(wid):
    p, split, fo, cnd = wi[w]
    links, vf = Pl[(p, split)]
    a, b = off[i], off[i + 1]
    if b - a != len(links):
        bad_order += 1
    if not np.allclose(np.asarray(Xt[a:b, cvf], float), vf, rtol=1e-5, equal_nan=True):
        bad_vf += 1
    if (np.asarray(Xt[a:b, ccond]) == 1.0).all() != (cnd == "queue_ongoing") or cnd != cond[i]:
        bad_cond += 1
    t5 = pd.Timestamp(fo) + pd.Timedelta(minutes=5)
    if t5.date() != pd.Timestamp(fo).date():
        day_diff += 1
    pd_dow = pd.Timestamp(fo).dayofweek
    if int(Xt[a, cdow]) != pd_dow:
        dow_diff += 1
    wk = pd_dow >= 5
    nwk.setdefault((split, cnd), [0, 0])[int(wk)] += 1
    if cnd == "queue_ongoing" and not (np.asarray(Xt[a:b, ci]) == 1).any():
        noq += 1
    info[w] = (i, links, wk, cnd)
print(f"link-count mismatches {bad_order}, vf-column vs links.csv mismatches {bad_vf}, cond mismatches {bad_cond}")
print(f"windows whose T+5 is on another date than T: {day_diff}; X dow column != pandas dayofweek(T): {dow_diff}")
print("weekday/weekend windows by (split, cond):", {k: f"{v[0]}/{v[1]}" for k, v in nwk.items()})
print("ongoing test windows with no queue visible at T (all links far):", noq)


def suppress(q, days="all"):
    q = q.copy(); drop = np.zeros(len(q), bool); order_bad = 0
    for w, sub in q.groupby("window_id", sort=False):
        i, links, wk, cnd = info[w]
        ts = sorted(sub.timestamp.unique())
        if len(ts) != 6:
            print("window", w, "has", len(ts), "timestamps")
        for t in ts:                                   # link order must be the P.links order at every step
            if not (sub[sub.timestamp == t].link_id.to_numpy() == links).all():
                order_bad += 1
        if cnd != "queue_ongoing" or (days == "weekday" and wk) or (days == "weekend" and not wk):
            continue
        far = far_links(i)
        fl = set(links[far])
        drop[sub.index[sub.link_id.isin(fl).to_numpy() & (sub.queue_pred.to_numpy() == 1)]] = True
    q.loc[drop, "queue_pred"] = 0
    return q, int(drop.sum()), order_bad


pairs = [("queue_s31_t1AdaptGapMLClipFd_t2v14afAttnR13Drop_t4l2.parquet", "queue_s31far0_t1AdaptGapMLClipFd_t2v14afAttnR13Drop_t4l2.parquet", "all"),
         ("queue_s31_t1AdaptGapMLClipFd_t2v14afAttnR13Drop_t4l2.parquet", "queue_s31farWD_t2.parquet", "weekday"),
         ("queue_s35raw_t1Adapt2GapMLClipFdBnd_t2nomem_t4l2.parquet", "queue_s35_t1Adapt2GapMLClipFdBnd_t2nomemFar_t4l2.parquet", "all"),
         ("queue_s37raw_t1Adapt2GapMLClipFdBnd_t2nomemNNonly_t4l2.parquet", "queue_s37_t1Adapt2GapMLClipFdBnd_t2nomemNNonlyFar_t4l2.parquet", "all")]
for src, dst, days in pairs:
    q = pd.read_parquet(W + "t2/" + src); mine, nd, ob = suppress(q, days); ref = pd.read_parquet(W + "t2/" + dst)
    same = (mine.reset_index(drop=True)[["window_id", "timestamp", "link_id"]] == ref[["window_id", "timestamp", "link_id"]]).all().all()
    print(f"{dst[:60]:60s} days={days}: my removal {nd}, rows aligned {same}, cell mismatches {(mine.queue_pred.to_numpy() != ref.queue_pred.to_numpy()).sum()}, "
          f"link-order violations {ob}, positives src {int(q.queue_pred.sum())} -> {int(ref.queue_pred.sum())}")
