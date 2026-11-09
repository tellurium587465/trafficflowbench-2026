"""G311 (11/05, ledger #246, the strategy's request 2): how much A17's (or another final's) Task 2 sets move when the q24f networks' saved
GPU (bf16) probabilities are replaced by the CPU fp32 ones of the new stage (t2_nn_cpu_infer.py on nn_test_afdem, prefix cpuq). The final's
t2_build_variant.py runs here unchanged (runpy) with only the reads of work/t2/probs_q24f_*.pkl (the base networks; not the X1 files)
redirected to probs_cpuq__q24f_*.pkl; the queue is compared with the built final's queue by split and condition.
usage: py -B -X utf8 g311_cpu_q24f_build.py <final A17|A21|B7|B11|...>"""
import os, sys, builtins, runpy
cand = sys.argv[1]; SRC = r"C:/Users/user/tfb_a8/src"; sys.path.insert(0, SRC); os.chdir(SRC)
import t2_final_spec as FS
env, choice = FS.FINAL[cand]; c = choice.split(",")
for k in [k for k in os.environ if k.startswith("TFB_T2_")]:
    del os.environ[k]
os.environ.update(env); W = os.environ["TFB_WORK"] + "/"
_open = builtins.open; n_red = [0]


def red(f, *a, **k):
    s = str(f).replace("\\", "/")
    if "/t2/probs_q24f_" in s and s.endswith(".pkl"):
        f = s.replace("/t2/probs_q24f_", "/t2/probs_cpuq__q24f_"); n_red[0] += 1
    return _open(f, *a, **k)


builtins.open = red
name = f"cpuq{cand}"; sys.argv = ["t2_build_variant.py", name, c[0], c[1], c[2], c[3], c[4], W + "t1/state_t1final.parquet", W + "t4/odme_l2.parquet"]
runpy.run_path(SRC + "/t2_build_variant.py", run_name="__main__")
builtins.open = _open
import pandas as pd, numpy as np
from cachelib import T2_PANELS
from common import REL
meta = {}
for sp in ("validation", "private"):
    for p in T2_PANELS:
        for r in pd.read_csv(REL / "task2" / p / sp / "window_index.csv").itertuples():
            meta[r.window_id] = (sp, r.condition, p)
a = pd.read_parquet(W + f"t2/queue_{name}.parquet"); b = pd.read_parquet(W + f"t2/queue_repro_fin{cand}t.parquet"); assert (a.window_id.values == b.window_id.values).all()
d = a[a.queue_pred.to_numpy().astype(bool) != b.queue_pred.to_numpy().astype(bool)]
print(f"{cand}: q24f files redirected {n_red[0]} | cells differing {len(d)} of {len(a)} in {d.window_id.nunique()} windows (of {a.window_id.nunique()})", flush=True)
if len(d):
    g = d.groupby([d.window_id.map(lambda w: meta[w][0]), d.window_id.map(lambda w: meta[w][1])]).window_id.agg(["size", "nunique"]); print(g.to_string(), flush=True)
    p1 = d[d.window_id.map(lambda w: meta[w][2]) == "D7_I10_E"]; print(f"panel D7_I10_E: cells {len(p1)}, windows {p1.window_id.nunique()}", flush=True)
