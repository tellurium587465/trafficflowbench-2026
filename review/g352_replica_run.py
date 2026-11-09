"""G352 (11/05, ledger #279 record): G321 on the June replica selector windows with other cvf probabilities: junmh0 -> $TFB_REP_PROBS
(e.g. junmh0ft, G351's fine-tuned nets); outputs g321_jun_* -> g352_<probs>_*. usage: TFB_REP_PROBS=junmh0ft py -B -X utf8 g352_replica_run.py"""
import os, sys
P_ = os.environ["TFB_REP_PROBS"]; s = open(r"C:/Users/user/tfb/review/g321_june_signs.py", encoding="utf-8").read()
s = s.replace("junmh0", P_).replace("g321_jun_", f"g352_{P_}_"); exec(compile(s, "g321_g352", "exec"))
