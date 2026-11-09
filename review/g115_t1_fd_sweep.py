"""G115 (10/29, ledger #65, strategy J4): LB sweep of the T1 FD blend strength a (t1_fd.py; production 0.75).
From the production T1 chain's clip output: t1_fd.py (a) -> t1_bnd_apply.py bnd3 0.75, then the submission with the T2 / T4
parts of finA5 (queue_finA5, odme_l2), so the LB difference is T1 (and T3) only.
Identity first: a = 0.75 must give the production T1 state bit for bit and finA5's submission.csv (sha256).
usage: py g115_t1_fd_sweep.py 0.75,0.5,1.0"""
import os, sys, subprocess, zipfile, hashlib, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from submit import assemble

S = r"C:/Users/user/tfb/src"; W = r"C:/Users/user/tfb/work/"
CLIP = W + "t1/state_f3ens3c_ad2_gapml_clip.parquet"; PROD = W + "t1/state_f3ens3c_ad2_gapml_clip_fd_bnd.parquet"
SHA_A5 = "7f6d3e983ed69c429692f92650778a7c0f29a6058ea22486a8f581121dcce84e"
Q = pd.read_parquet(W + "t2/queue_finA5.parquet"); O = pd.read_parquet(W + "t4/odme_l2.parquet")
for a in sys.argv[1].split(","):
    tag = f"fd{int(round(float(a) * 100)):03d}"; fd = W + f"t1/state_sweep_{tag}.parquet"; bnd = W + f"t1/state_sweep_{tag}_bnd.parquet"
    for cmd in (["py", "-W", "ignore", "t1_fd.py", CLIP, fd, a], ["py", "-W", "ignore", "t1_bnd_apply.py", "bnd3", "0.75", fd, bnd]):
        print(subprocess.run(cmd, cwd=S, capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1], flush=True)
    st = pd.read_parquet(bnd)
    if a == "0.75":
        print("identity: a = 0.75 T1 state equals production:", st.equals(pd.read_parquet(PROD)), flush=True)
    z = assemble(st, Q, O, f"t1{tag}_finA5")
    h = hashlib.sha256(zipfile.ZipFile(z).read("submission.csv")).hexdigest()
    print(a, z, "submission.csv sha256", h[:16], "(= finA5)" if h == SHA_A5 else "(differs from finA5)", flush=True)
