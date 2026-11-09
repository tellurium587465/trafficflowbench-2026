"""Single entry point that rebuilds the final submission from the released data.

Steps (each skips work whose output already exists unless --force):
  1. matrix caches            build_cache.py
  2. Task 1 model + inference t1_big.py / t1_infer2.py      (feature set v3)
  3. Task 2 windows, caches, model + inference              t2_windows / t2_cache / t2_selset / t2_cache2 / t2_final
  4. Task 4 exact L2 projection                             task4.solve_l2_exact
  5. assemble + validate                                    submit.assemble

Usage: python run_all.py --release-root <kaggle_public> [--force]
Environment: Python 3.14, numpy 2.5, pandas 3.0, scipy 1.18, lightgbm (see requirements), CPU only.
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path(r"C:/Users/user/tfb/work")


def sh(args):
    print(">>", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-W", "ignore", *args], cwd=HERE, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--t1-tag", default="f3big")
    ap.add_argument("--t2-tag", default="v4")
    a = ap.parse_args()
    os.environ["TFB_REL"] = a.release_root
    if a.force or len(list((WORK / "cache").glob("*.npz"))) < 30:
        sh(["build_cache.py"])
    if a.force or not (WORK / "t1" / f"state_{a.t1_tag}.parquet").exists():
        sh(["t1_big.py", "2000000", a.t1_tag, "0.05"])
        sh(["t1_infer2.py", "3", a.t1_tag])
    if a.force or not (WORK / "t2" / f"queue_{a.t2_tag}.parquet").exists():
        sh(["-c", "import pickle;from t2_windows import build;pickle.dump(build(),open(r'C:/Users/user/tfb/work/t2_windows_train.pkl','wb'))"])
        sh(["-c", "import pickle;from t2_windows import build;W=pickle.load(open(r'C:/Users/user/tfb/work/t2_windows_train.pkl','rb'));h={(w['panel'],w['t']) for w in W};E=[w for w in build(spacing=12,seed=1) if (w['panel'],w['t']) not in h];pickle.dump(E,open(r'C:/Users/user/tfb/work/t2_windows_extra12.pkl','wb'));h|={(w['panel'],w['t']) for w in E};E4=[w for w in build(spacing=4,seed=2) if (w['panel'],w['t']) not in h];pickle.dump(E4,open(r'C:/Users/user/tfb/work/t2_windows_extra4.pkl','wb'))"])
        sh(["t2_cache.py"]); sh(["t2_selset.py"]); sh(["t2_cache2.py"])
        os.environ["TFB_T2CACHE"] = "2"
        sh(["t2_final.py", a.t2_tag, '{"num_leaves":127}', "600,1500"])
    sh(["-c", "import pandas as pd;from common import PANELS;from task4 import load_problem,distinct_rows,solve_l2_exact,frame;"
              "rows=[frame(p,solve_l2_exact(*distinct_rows(p['A'],p['c'])[:2],p['b'])[0]) for s in ('validation','private') for p in [load_problem(x,s) for x in PANELS]];"
              "pd.concat(rows,ignore_index=True).to_parquet(r'C:/Users/user/tfb/work/t4/odme_final.parquet',index=False)"])
    sh(["-c", f"import pandas as pd;from submit import assemble;W=r'C:/Users/user/tfb/work/';"
              f"assemble(pd.read_parquet(W+'t1/state_{a.t1_tag}.parquet'),pd.read_parquet(W+'t2/queue_{a.t2_tag}.parquet'),pd.read_parquet(W+'t4/odme_final.parquet'),'final_{a.t1_tag}_{a.t2_tag}')"])


if __name__ == "__main__":
    main()
