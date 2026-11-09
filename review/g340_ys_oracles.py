"""G340 (11/05, ledger #271, measurement only): the #262 ongoing oracles (O_box / O_tail / O_new / O_box+tail) within the young / small box
stratum (G337 / G338's ys from R, known at T). Runs G328 unchanged except that each table's per-window rows are saved
(work/t2/g340_<instrument>.csv); the split by ys is read by g340_report.py. usage: TFB_M3VER=v3 py -B -X utf8 g340_ys_oracles.py"""
import sys
s = open(r"C:/Users/user/tfb/review/g328_oracle_map.py", encoding="utf-8").read()
a = "    D = pd.DataFrame(rows); b = D.base;"
assert s.count(a) == 1
s = s.replace(a, "    D = pd.DataFrame(rows); D.to_csv(W + 'g340_' + lab.split(' (')[0].replace(' / ', '_').replace(' ', '_') + '.csv', index=False); b = D.base;")
exec(compile(s, "g328p340", "exec"))
