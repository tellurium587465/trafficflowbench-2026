"""M5 audit V5: rebuild the T2 blends of s31/s35/s36/s37 from the component probs with the recorded specs, compare with
probs_<name>.pkl and queue_<name>.parquet; component sanity (finite, shape, onset steps 0-4, weights); s36 zip/far."""
import sys, pickle, zipfile, numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, "C:/Users/user/tfb/src")
W = "C:/Users/user/tfb/work/"


def best_set(p):
    pe = p; order = np.argsort(-pe); cs = np.cumsum(pe[order]); tot = pe.sum(); k = np.arange(1, len(pe) + 1)
    kb = int(np.argmax(cs / (k + tot - cs))) + 1; s = np.zeros(len(p), bool); s[order[:kb]] = True; return s


att = lambda pre, seeds: "+".join(f"{pre}_attnglob_s{s}:2+{pre}_attnwp05_s{s}:2+{pre}_attn1_s{s}:2" for s in seeds)
S31 = "v14af:8+" + "+".join(f"q15f_attnglob_s{s}:2+q15f_attnwp05_s{s}:2+q15f_attn1_s{s}:2" for s in range(4)) + "+" + "+".join(f"q19f_r13_s{s}+q19f_drop1_s{s}" for s in range(4))
S35 = "v14af:8+" + att("q24f", range(4)) + "+" + "+".join(f"q24f_r13_s{s}+q24f_drop1_s{s}" for s in range(4))   # = scratchpad/s35_spec.txt (component set and weights)
S37 = S35.replace("v14af:8+", "", 1)
S36 = "v14af:8+" + "+".join(f"q26f_attnglob_s{s}:2+q26f_attnwp05_s{s}:2+q26f_attn1_s{s}:2+q26f_drop1_s{s}" for s in range(4)) + "+" + "+".join(f"q26f_r13_s{s}:1.3333333" for s in (1, 2, 3))
BUILDS = {"s31_t1AdaptGapMLClipFd_t2v14afAttnR13Drop_t4l2": S31, "s35raw_t1Adapt2GapMLClipFdBnd_t2nomem_t4l2": S35,
          "s37raw_t1Adapt2GapMLClipFdBnd_t2nomemNNonly_t4l2": S37, "s36raw_t1Adapt2GapMLClipFdBnd_t2rstat_t4l2": S36}
cache = {}
cond = dict(zip(np.load("E:/tfb_nn/nn_test_afdem/window_id.npy", allow_pickle=True).astype(str), np.load("E:/tfb_nn/nn_test_afdem/cond.npy", allow_pickle=True).astype(str)))
for name, spec in BUILDS.items():
    parts = [(t, float(w)) for t, w in (x.split(":") if ":" in x else (x, "1") for x in spec.split("+"))]
    tw = sum(w for _, w in parts); grp = {}
    for t, w in parts:
        g = "lgbm" if t.startswith("v14") else ("r13" if "_r13_" in t else ("drop1" if "_drop1_" in t else "attn"))
        grp[g] = grp.get(g, 0) + w / tw
    pr = {}
    for t, _ in parts:
        if t not in cache:
            cache[t] = pickle.load(open(W + f"t2/probs_{t}.pkl", "rb"))
        pr[t] = cache[t]
    keys = set.intersection(*[set(p) for p in pr.values()])
    nonfin = {t: sum((~np.isfinite(v)).any() for v in p.values()) for t, p in pr.items()}
    on_early = {t: max(float(np.abs(p[k][:5]).max()) for k in p if cond[k] == "queue_onset") for t, p in pr.items()}
    blend = {k: sum(w * pr[t][k] for t, w in parts) / tw for k in keys}
    ref = pickle.load(open(W + f"t2/probs_{name}.pkl", "rb"))
    dmax = max(float(np.abs(blend[k] - ref[k]).max()) for k in keys)
    q = pd.read_parquet(W + f"t2/queue_{name}.parquet"); mism = 0; npos = 0
    for w, sub in q.groupby("window_id", sort=False):
        b = blend[w]; sel = best_set(b.ravel()).reshape(b.shape)
        mism += int((sel.ravel().astype(int) != sub.queue_pred.to_numpy()).sum()); npos += int(sel.sum())
    print(f"{name}\n   components {len(parts)}, total weight {tw:g}, weight shares {{{', '.join(f'{g}: {v:.3f}' for g, v in grp.items())}}}, windows {len(keys)}")
    print(f"   non-finite windows per component: {sum(nonfin.values())}; max |p| at onset steps 0-4: {max(on_early.values()):.3g} "
          f"({', '.join(t for t, v in on_early.items() if v > 0) or 'all zero'})")
    print(f"   my blend vs probs pkl max|diff| {dmax:.3g}; queue cells differing from my best_set {mism}; positives {npos} (parquet {int(q.queue_pred.sum())})")

# s36: zip T2 rows vs parquet and far rule
key = pq.read_table(W + "submission_key.parquet", columns=["submission_id", "task", "window_id", "timestamp", "link_id"], filters=[("task", "==", "queue")]).to_pandas()
qrows = []
for ch in pd.read_csv(zipfile.ZipFile(W + "subs/s36_t1Adapt2GapMLClipFdBnd_t2rstatFar_t4l2.zip").open("submission.csv"), chunksize=1_000_000, dtype={"task": str}):
    qrows.append(ch.loc[ch.task == "queue", ["submission_id", "queue_pred"]])
j = key.merge(pd.concat(qrows), on="submission_id").merge(pd.read_parquet(W + "t2/queue_s36_t1Adapt2GapMLClipFdBnd_t2rstatFar_t4l2.parquet").rename(columns={"queue_pred": "exp"}),
                                                          on=["window_id", "timestamp", "link_id"])
print("s36 zip vs parquet: rows", len(j), "mismatch", int((j.queue_pred != j.exp).sum()))
