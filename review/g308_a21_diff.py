"""G308 (11/05, ledger #245): A21t against A17t, by split and condition. A21 leaves the 2D U-Net out of the ongoing mix only, so Task 1, 3, 4 and the onset cells
must be identical and the queue cells may differ on April's windows only.
usage: py g308_a21_diff.py"""
import zipfile, hashlib, pandas as pd, sys
sys.path.insert(0, r"C:/Users/user/tfb/src")
from common import REL
from cachelib import T2_PANELS
R = r"C:/Users/user/tfb/work/"; meta = {}
for sp in ("validation", "private"):
    for p in T2_PANELS:
        for r in pd.read_csv(REL / "task2" / p / sp / "window_index.csv").itertuples():
            meta[r.window_id] = (sp, r.condition)
a13 = zipfile.ZipFile(R + "subs/repro_finA21t_final.zip").read("submission.csv"); print("sha256 A21t", hashlib.sha256(a13).hexdigest())
a = a13.split(b"\n"); b = zipfile.ZipFile(R + "subs/repro_finA17t_final.zip").read("submission.csv").split(b"\n"); assert len(a) == len(b), "row count"
dr = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]; print("differing rows", len(dr), "of", len(a) - 1, "| tasks", sorted({x.split(b",")[1].decode() for x in (a[i] for i in dr)}))
q16 = pd.read_parquet(R + "t2/queue_repro_finA21t.parquet"); q15 = pd.read_parquet(R + "t2/queue_repro_finA17t.parquet"); assert (q16.window_id.values == q15.window_id.values).all()
d = q16[q16.queue_pred.to_numpy().astype(bool) != q15.queue_pred.to_numpy().astype(bool)]
print("queue cells differing:", len(d), "in", d.window_id.nunique(), "windows"); print(d.groupby([d.window_id.map(lambda w: meta[w][0]), d.window_id.map(lambda w: meta[w][1])]).size().to_string())
ok = set(meta[w][1] for w in d.window_id) <= {"queue_ongoing"} and len(dr) == len(d); print("A21t = A17t on Tasks 1, 3, 4 and onset; differences on ongoing queue cells only:", ok)
