"""G281 (11/05, ledger #232): A16t against A15t, by split and condition. A16 changes the rank-input onset network to seeds 0-3 only, so Task 1, 3, 4 and the ongoing cells
must be identical and the queue cells may differ on April's windows only.
usage: py g281_a16_diff.py"""
import zipfile, hashlib, pandas as pd, sys
sys.path.insert(0, r"C:/Users/user/tfb/src")
from common import REL
from cachelib import T2_PANELS
R = r"C:/Users/user/tfb/work/"; meta = {}
for sp in ("validation", "private"):
    for p in T2_PANELS:
        for r in pd.read_csv(REL / "task2" / p / sp / "window_index.csv").itertuples():
            meta[r.window_id] = (sp, r.condition)
a13 = zipfile.ZipFile(R + "subs/repro_finA16t_final.zip").read("submission.csv"); print("sha256 A16t", hashlib.sha256(a13).hexdigest())
a = a13.split(b"\n"); b = zipfile.ZipFile(R + "subs/repro_finA15t_final.zip").read("submission.csv").split(b"\n"); assert len(a) == len(b), "row count"
dr = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]; print("differing rows", len(dr), "of", len(a) - 1, "| tasks", sorted({x.split(b",")[1].decode() for x in (a[i] for i in dr)}))
q16 = pd.read_parquet(R + "t2/queue_repro_finA16t.parquet"); q15 = pd.read_parquet(R + "t2/queue_repro_finA15t.parquet"); assert (q16.window_id.values == q15.window_id.values).all()
d = q16[q16.queue_pred.to_numpy().astype(bool) != q15.queue_pred.to_numpy().astype(bool)]
print("queue cells differing:", len(d), "in", d.window_id.nunique(), "windows"); print(d.groupby([d.window_id.map(lambda w: meta[w][0]), d.window_id.map(lambda w: meta[w][1])]).size().to_string())
ok = set(meta[w][1] for w in d.window_id) <= {"queue_onset"} and len(dr) == len(d); print("A16t = A15t on Tasks 1, 3, 4 and ongoing; differences on onset queue cells only:", ok)
