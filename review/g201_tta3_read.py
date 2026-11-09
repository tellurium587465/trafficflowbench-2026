"""G201 (11/01, ledger #103b): G142's score on the new seeds: c24f_attnglob s2, s3, scale 1 of the same fit as the base, tta3 against it, A6's flow. G142 and its csv are not changed. Dev windows only.
usage: TFB_M3VER=v3 py -B -X utf8 g201_tta3_read.py"""
import sys
s = open(r"C:/Users/user/tfb/review/g142_x1_arms_score.py", encoding="utf-8").read()


def sub(a, b):
    global s
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)


sub('TAGS = [f"c24f_attnglob_s{s}" for s in (0, 1)]', 'TAGS = [f"c24f_attnglob_s{s}" for s in (2, 3)]')
sub('    print("identity tta scale 1 against the saved X1: max |dp|", max(float(np.abs(TT[(w, k, 1.0)] - X1[(w, k)]).max()) for (w, k, s) in TT if s == 1.0), flush=True)\n', '    X1 = {(w, k): TT[(w, k, 1.0)] for (w, k, s) in TT if s == 1.0}\n')
sub('TN = TN if len(TN) == len(ON) else {}', 'TN = {}')
sub('D.to_csv(OUT + "g142.csv", index=False)', 'D.to_csv(OUT + "g142b.csv", index=False)')
sys.argv = [sys.argv[0]]; exec(compile(s, "g142b", "exec"))
