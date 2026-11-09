"""G237 (11/03, ledgers #189 / #179, A13t): G174's same-day X1 on the hold-out ongoing windows with the March pool on origins every
3 rows (nn_data_march_sp3 = twin/build_march.py 3 validation; G236's rule: the greedy spacing 3 of nn_data_march_sp1, bit for bit),
probabilities only, no score. G141's MP / MT (the pool's panel and row, read from nn_data_march there) are taken from the dense pool.
Writes work/t2/x1_arms/parts_holdout_d3/sdd3__<tag>__<D0>.pkl per hold-out day, restart-safe; the read is #179's, once.
usage: TFB_M3VER=v3 py -B -X utf8 g237_sdd3_holdout.py <tag>"""
import os, sys
os.environ["G141_MD"] = "nn_data_march_sp3"
R_ = r"C:/Users/user/tfb/review/"; sys.dont_write_bytecode = True
s = open(R_ + "g174_sd_holdout.py", encoding="utf-8").read()


def sub(a, b):
    global s
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)


sub('O2 = OUT + "parts_holdout/"', 'MP = np.load(f"{NN}/{MDIR}/panel.npy", allow_pickle=True); MT = np.load(f"{NN}/{MDIR}/t.npy"); assert len(MT) == len(MD["day"]) == 4763\nO2 = OUT + "parts_holdout_d3/"')
sub('f = O2 + f"sd__{tag}__{D0}.pkl"', 'f = O2 + f"sdd3__{tag}__{D0}.pkl"')
sub('log = open(OUT + f"sd_hold__{tag}.log", "a")', 'log = open(OUT + f"sdd3_hold__{tag}.log", "a")')
sys.argv = [R_ + "g174_sd_holdout.py"] + sys.argv[1:]
exec(compile(s, "g174d3", "exec"))
