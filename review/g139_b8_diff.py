"""G139 (10/31, ledger #97): B8 (the B if #63 keeps finA5) against finB7 and finB4, by split and condition: B8 must equal finB7 on April and finB4 on March.
usage: py g139_b8_diff.py"""
import zipfile, hashlib, pandas as pd, sys
sys.path.insert(0, r"C:/Users/user/tfb/src")
from common import REL
from cachelib import T2_PANELS
R = r"C:/Users/user/tfb/work/"; meta = {}
for sp in ("validation", "private"):
    for p in T2_PANELS:
        for r in pd.read_csv(REL / "task2" / p / sp / "window_index.csv").itertuples():
            meta[r.window_id] = (sp, r.condition)
b8 = zipfile.ZipFile(R + "subs/repro_finB8_final.zip").read("submission.csv"); print("sha256 B8", hashlib.sha256(b8).hexdigest()); a = b8.split(b"\n"); q8 = pd.read_parquet(R + "t2/queue_repro_finB8.parquet")
for nm, zp, qp, want in (("finB7", "finB7_final.zip", "queue_finB7.parquet", "validation"), ("finB4", "finB4_final.zip", "queue_finB4.parquet", "private")):
    b = zipfile.ZipFile(R + "subs/" + zp).read("submission.csv").split(b"\n"); assert len(a) == len(b), "row count"
    print(nm, "differing rows", sum(x != y for x, y in zip(a, b)), "of", len(a) - 1)
    q = pd.read_parquet(R + "t2/" + qp); assert (q.window_id.values == q8.window_id.values).all()
    d = q8[q8.queue_pred.to_numpy().astype(bool) != q.queue_pred.to_numpy().astype(bool)]
    print(d.groupby([d.window_id.map(lambda w: meta[w][0]), d.window_id.map(lambda w: meta[w][1])]).size().to_string()); assert {meta[w][0] for w in d.window_id} <= {want}, nm
print("B8 = finB7 on April, finB4 on March: True")
