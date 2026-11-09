"""11/03 ledger #189 (speculative cache, discarded unless the January / February read passes): review/t2_x1_sameday_dense.py =
tfb_a8/src/t2_x1_sameday.py with the private month's pool directory (X1_POOL_PRIV, default nn_data_april) and the output arm
(X1_OUTARM, default "": probs_x1offsd<arm>__<tag>.pkl) from the environment; src is not changed. usage: py mk_x1sd_dense.py"""
import hashlib
s = open(r"C:/Users/user/tfb_a8/src/t2_x1_sameday.py", encoding="utf-8").read()


def sub(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)


sub('DATA = {"validation": "nn_data_march", "private": "nn_data_april"}', 'DATA = {"validation": "nn_data_march", "private": os.environ.get("X1_POOL_PRIV", "nn_data_april")}')
sub('open(W + f"probs_x1offsd__{tag}.pkl", "wb")', 'open(W + f"probs_x1offsd{os.environ.get(\'X1_OUTARM\', \'\')}__{tag}.pkl", "wb")')
sub('"""Same-day X1 for the official ongoing windows', '"""(11/03 #189 speculative copy: X1_POOL_PRIV, X1_OUTARM.) Same-day X1 for the official ongoing windows')
open(r"C:/Users/user/tfb/review/t2_x1_sameday_dense.py", "w", encoding="utf-8").write(s); print("written", hashlib.sha256(s.encode()).hexdigest()[:8])
