"""G279 (11/05, ledgers #230, strategy 11/05 00:3x item 3): A15t against A13t, by split and condition. A15 changes April's ongoing family to seeds 0-15 only, so March, Task 1, 3 and 4
must be identical and the queue cells may differ on April's windows only.
usage: py g279_a15_diff.py"""
import zipfile, hashlib, pandas as pd, sys
sys.path.insert(0, r"C:/Users/user/tfb/src")
from common import REL
from cachelib import T2_PANELS
R = r"C:/Users/user/tfb/work/"; meta = {}
for sp in ("validation", "private"):
    for p in T2_PANELS:
        for r in pd.read_csv(REL / "task2" / p / sp / "window_index.csv").itertuples():
            meta[r.window_id] = (sp, r.condition)
a13 = zipfile.ZipFile(R + "subs/repro_finA15t_final.zip").read("submission.csv"); print("sha256 A15t", hashlib.sha256(a13).hexdigest())
a = a13.split(b"\n"); b = zipfile.ZipFile(R + "subs/repro_finA13t_final.zip").read("submission.csv").split(b"\n"); assert len(a) == len(b), "row count"
dr = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]; print("differing rows", len(dr), "of", len(a) - 1, "| tasks", sorted({x.split(b",")[1].decode() for x in (a[i] for i in dr)}))
q15 = pd.read_parquet(R + "t2/queue_repro_finA15t.parquet"); q13 = pd.read_parquet(R + "t2/queue_repro_finA13t.parquet"); assert (q15.window_id.values == q13.window_id.values).all()
d = q15[q15.queue_pred.to_numpy().astype(bool) != q13.queue_pred.to_numpy().astype(bool)]
print("queue cells differing:", len(d), "in", d.window_id.nunique(), "windows"); print(d.groupby([d.window_id.map(lambda w: meta[w][0]), d.window_id.map(lambda w: meta[w][1])]).size().to_string())
ok = {meta[w][0] for w in d.window_id} <= {"private"} and len(dr) == len(d); print("A15t = A13t on March and Tasks 1, 3, 4; differences on April's queue only:", ok)
