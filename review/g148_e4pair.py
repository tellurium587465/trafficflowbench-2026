"""G148 (10/31, ledger #109, strategy candidate 7): the E4 premise on the real March (public LB, diagnostic only, never used to pick A / B).
Two submissions on finB7's base, only the 80 March windows of Task 2 differ: P0 = q24f_attnglob seeds 0-3 (nine months), P1 = b0L seeds
0-3 (fold 2, six months, without 2030-12 / 2031-01 / 2031-02). Onset and ongoing use the same raw blend, no X1, no attention network;
cone, evidence rule and far gate as in B7. Nothing in src changes: t2_build_variant.py runs here with FS.config patched.
  infer : CPU probabilities of the eight networks on nn_test_afdem (work/t2/probs_e4p__<tag>.pkl), and the M3 copies g130 reads
  build <0|1> : t2_build_variant.py as e4p<arm>
  check : both zips against finB7_final.zip, row by row (April Task 2, Tasks 1, 3, 4 must be identical)
usage: py g148_e4pair.py infer | build 0 | build 1 | check"""
import os, sys, runpy, shutil, subprocess, zipfile
R = r"C:/Users/user/tfb/"; SRC = R + "src/"; W = R + "work/"; MODE = sys.argv[1]
TAGS = {0: [f"q24f_attnglob_s{s}" for s in range(4)], 1: [f"b0L_s{s}_fold2030-12" for s in range(4)]}


def infer():
    subprocess.run([sys.executable, "t2_nn_cpu_infer.py", ",".join(TAGS[0] + TAGS[1]), "nn_test_afdem", "e4p"], cwd=SRC, check=True)
    for s, t in enumerate(TAGS[0]):                             # g130 m3 reads the fold-2 name pattern; the probabilities are already CPU ones
        shutil.copyfile(W + f"t2/probs_m3v3__{t}.pkl", W + f"t2/probs_m3v3cap__q24fa_s{s}_fold2030-12.pkl")


def build(arm):
    sys.path.insert(0, SRC); os.chdir(SRC)
    import t2_final_spec as FS
    env, _ = FS.FINAL["B7"]; os.environ.update({k: v for k, v in env.items() if not k.startswith("TFB_T2_TFM")})
    cfg0 = FS.config

    def cfg():
        c = cfg0(); s = "+".join(f"e4p__{t}" for t in TAGS[arm]); c["spec"]["validation"] = dict(queue_onset=s, queue_ongoing=s); c["tfm"] = {}
        return c
    FS.config = cfg
    sys.argv = ["t2_build_variant.py", f"e4p{arm}", "base", "0", "base", "0", "1", W + "t1/state_f3ens3c_ad2_gapml_clip_fd_bnd.parquet", W + "t4/odme_l2.parquet"]
    runpy.run_path("t2_build_variant.py", run_name="__main__")


def check():
    import numpy as np, pandas as pd
    sys.path.insert(0, SRC)
    from cachelib import T2_PANELS
    from common import REL
    rd = lambda n: pd.read_csv(zipfile.ZipFile(W + f"subs/{n}.zip").open("submission.csv"), dtype={"submission_id": str})
    B = rd("finB7_final"); q = pd.read_parquet(W + "t2/queue_finB7.parquet"); bq = B[B.task == "queue"]; print("rows", len(B), "queue", len(bq), flush=True)
    val = set()
    for p in T2_PANELS:
        val |= set(pd.read_csv(REL / "task2" / p / "validation" / "window_index.csv").window_id)
    for a in (0, 1):
        Z = rd(f"e4p{a}_final"); assert (Z.submission_id == B.submission_id).all()
        nq = Z.task != "queue"; same = all(np.array_equal(Z.loc[nq, c].to_numpy(), B.loc[nq, c].to_numpy()) for c in ("speed_kmh", "flow_vph", "path_flow"))
        Q = pd.read_parquet(W + f"t2/queue_e4p{a}.parquet"); assert len(Q) == len(q) and (Q.window_id == q.window_id).all()
        d = Q.queue_pred.to_numpy() != q.queue_pred.to_numpy(); isv = q.window_id.isin(val).to_numpy()
        print(f"e4p{a}: Tasks 1, 3, 4 identical {same} | queue cells changed March {int(d[isv].sum())} of {int(isv.sum())}, April {int(d[~isv].sum())} of {int((~isv).sum())} | windows changed {q.window_id[d].nunique()} | set share {Q.queue_pred.mean():.4f} vs {q.queue_pred.mean():.4f}", flush=True)
        assert same and not d[~isv].any()


{"infer": infer, "build": lambda: build(int(sys.argv[2])), "check": check}[MODE]()
