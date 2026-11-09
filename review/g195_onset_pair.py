"""G195 (11/01, ledgers #159, #173b): an LB pair that changes only March's onset rows of finA8. <queue_b> = t2_onset_evidence.py's output for the variant; the rows of the validation split's onset windows are taken from it, everything else stays finA8 (same T1, T4). Prints the windows changed and the IoU of each with finA8's set.
usage: py -B -X utf8 g195_onset_pair.py <queue_b name> <submission name>"""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.dont_write_bytecode = True; sys.path.insert(0, r"C:/Users/user/tfb_a8/src")
from common import WORK, REL
from cachelib import T2_PANELS
from submit import assemble
W = WORK + "/t2/"; A = pd.read_parquet(W + "queue_finA8.parquet"); B = pd.read_parquet(W + f"queue_{sys.argv[1]}.parquet"); print(list(A.columns)[:8], len(A), len(B))
on = {r.window_id for p in T2_PANELS for r in pd.read_csv(REL / "task2" / p / "validation" / "window_index.csv").itertuples() if r.condition == "queue_onset"}; print("March onset windows", len(on))
k = [c for c in A.columns if c != "queue_pred"]; assert A[k].equals(B[k]), "keys differ"; m = A.window_id.isin(on).to_numpy(); C = A.copy(); C.loc[m, "queue_pred"] = B.queue_pred.to_numpy()[m]; d = A[m & (A.queue_pred.to_numpy() != B.queue_pred.to_numpy())]; print("windows changed", d.window_id.nunique(), "rows changed", len(d), "positives", int(A.queue_pred[m].sum()), "->", int(B.queue_pred[m].sum()))
C.to_parquet(W + f"queue_{sys.argv[2]}.parquet", index=False); print(assemble(pd.read_parquet(WORK + "/t1/state_f3ens3c_ad2_gapml_clip_fd_bnd.parquet"), C, pd.read_parquet(WORK + "/t4/odme_l2.parquet"), sys.argv[2] + "_final"))
