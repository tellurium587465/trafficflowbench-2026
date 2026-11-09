"""Build a named final candidate (t2_final_spec.FINAL: its env options and per-condition choice) with t2_build_variant.py,
10/23. The same names drive the one-window entry check (t2_trunc_physical.py).
usage: python t2_build_final.py <candidate A|B|...> <submission name> [state_parquet] [odme_parquet]"""
import os, sys, subprocess
import t2_final_spec as FS
from common import WORK

cand, name = sys.argv[1], sys.argv[2]
state = sys.argv[3] if len(sys.argv) > 3 else WORK + "/t1/state_f3ens3c_ad2_gapml_clip_fd_bnd.parquet"
odme = sys.argv[4] if len(sys.argv) > 4 else WORK + "/t4/odme_l2.parquet"
env, choice = FS.FINAL[cand]; c = choice.split(",")
print(f"{cand} -> {name}: {env}, choice {choice}", flush=True)
subprocess.run([sys.executable, "t2_build_variant.py", name, c[0], c[1], c[2], c[3], c[4], state, odme], cwd=os.path.dirname(os.path.abspath(__file__)),
               check=True, env=dict({k: v for k, v in os.environ.items() if not k.startswith("TFB_T2_")}, **env))   # only the candidate's options
