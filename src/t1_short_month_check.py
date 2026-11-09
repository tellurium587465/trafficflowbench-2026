"""E6 (INSTRUCTIONS_WORKER: "the T1 stage must not crash on a short month"): the Task 1 model stage of run_final.sh
(t1_infer2.py with TFB_T1_ADAPT=1 TFB_T1_FEAT_TRAINSTAT=1, the three LightGBM pairs) run on the caches that
t2_trunc_physical.py built from release copies cut at a window's T (one panel x split each). Checks: no exception,
predictions for every remaining Task 1 target cell, all finite. Task 1 is not part of the organisers' truncation check
(749601 concerns Task 2); this only shows that a truncated release does not break the T1 code path.
usage: python t1_short_month_check.py <window_id> [<window_id> ...]   (cut copies under C:/tfb_data/e6_trunc/<window_id>)"""
import os, sys, subprocess

CHILD = r'''
import os, sys, numpy as np, lightgbm as lgb
from pathlib import Path
from cachelib import load
from common import PANELS
from t1_exp import featfns
panel, split = sys.argv[1], sys.argv[2]; MD = Path(r"C:/Users/user/tfb/work/t1"); gf, rf, bf = featfns(3)
P = load(panel, split); pi = PANELS.index(panel); F, _, _, _ = gf(P); bv, bq = bf(F); r, c = np.nonzero(P.target)
X, _ = rf(P, F, r, c); X = np.column_stack([np.full(len(r), pi, np.float32), X])
for tag in ("f3big", "f3all", "f3all2"):
    mv = lgb.Booster(model_file=str(MD / f"speed_{tag}.txt")); mq = lgb.Booster(model_file=str(MD / f"flow_{tag}.txt"))
    v = (bv[r, c] + mv.predict(X, num_threads=16)) * P.vf[c]; q = (bq[r, c] + mq.predict(X, num_threads=16)) * P.lanes[c]
    print(panel, split, tag, "target cells", len(r), "finite", bool(np.isfinite(v).all() and np.isfinite(q).all()), "last target", P.times[r.max()] if len(r) else "-")
'''
ok = True
for w in sys.argv[1:]:
    split = "validation" if "_validation_" in w else "private"; panel = w[3:w.index(f"_{split}_")]
    env = dict(os.environ, TFB_CACHE=f"C:/tfb_data/e6_trunc/{w}/cache", TFB_REL=f"C:/tfb_data/e6_trunc/{w}/rel",
               TFB_T1_ADAPT="1", TFB_T1_FEAT_TRAINSTAT="1")
    p = subprocess.run([sys.executable, "-c", CHILD, panel, split], cwd=os.path.dirname(os.path.abspath(__file__)), env=env, capture_output=True, text=True)
    print(w, "exit", p.returncode, "|", p.stdout.strip().replace("\n", " | ") if p.returncode == 0 else p.stderr[-800:])
    ok &= p.returncode == 0 and "finite False" not in p.stdout
print("T1 short-month check:", "PASS" if ok else "FAIL")
