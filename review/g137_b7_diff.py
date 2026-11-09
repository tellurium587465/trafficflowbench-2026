"""G137 (10/31, ledger #97): finB7 against finB6, row by row: every differing submission row must be a Task 2 row of an April ongoing window; the T2 queue tables say how much moved.
usage: py g137_b7_diff.py"""
import zipfile, hashlib, numpy as np, pandas as pd, sys
sys.path.insert(0, r"C:/Users/user/tfb/src")
from common import REL
from cachelib import T2_PANELS
R = r"C:/Users/user/tfb/work/"; zb6 = zipfile.ZipFile(R + "subs/finB6_final.zip").read("submission.csv"); zb7 = zipfile.ZipFile(R + "subs/finB7_final.zip").read("submission.csv")
print("sha256 B6", hashlib.sha256(zb6).hexdigest()[:16], "B7", hashlib.sha256(zb7).hexdigest()[:16], "bytes", len(zb6), len(zb7))
a = zb6.split(b"\n"); b = zb7.split(b"\n"); assert len(a) == len(b), "row count"; print("rows", len(a) - 1, "header same", a[0] == b[0], a[0][:80])
dif = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]; print("differing rows", len(dif)); print("first", a[dif[0]][:120], "|", b[dif[0]][:120]) if dif else None
meta = {}
for sp in ("validation", "private"):
    for p in T2_PANELS:
        for r in pd.read_csv(REL / "task2" / p / sp / "window_index.csv").itertuples():
            meta[r.window_id] = (sp, r.condition)
tok = [a[i].split(b",")[0].decode() for i in dif]; print("sample ids", tok[:3])
q6 = pd.read_parquet(R + "t2/queue_repro_finB6.parquet"); q7 = pd.read_parquet(R + "t2/queue_finB7.parquet"); assert (q6.window_id.values == q7.window_id.values).all()
d = q6.queue_pred.to_numpy().astype(bool) != q7.queue_pred.to_numpy().astype(bool); q6["sp"] = [meta[w][0] for w in q6.window_id]; q6["c"] = [meta[w][1] for w in q6.window_id]; q6["d"] = d; q6["p6"] = q6.queue_pred.astype(bool); q6["p7"] = q7.queue_pred.astype(bool).values
g = q6.groupby(["sp", "c"]).agg(cells=("d", "size"), differ=("d", "sum"), b6=("p6", "sum"), b7=("p7", "sum")); print(g.to_string())
w = q6[q6.d].groupby("window_id").size(); print("windows that differ", len(w), "of", q6[(q6.sp == "private") & (q6.c == "queue_ongoing")].window_id.nunique(), "April ongoing")
assert set(q6[q6.d].sp) <= {"private"} and set(q6[q6.d].c) <= {"queue_ongoing"}; print("only April ongoing windows differ: True")
