"""G224 (11/03, ledger #189, January / February arm): twin/build_march.py's pool, run on the train split for 2031-01 and
2031-02 only, origins every row (SP = 1) -> NN/nn_data_jf_sp1. The build's source is executed with three textual changes:
the split, the month filter on the origin and no release blanks (the train split has none). check: the spacing-6
subset against the January / February windows of nn_data_afdem (window sets and features).
usage: py g224_build_jf_dense.py build | check"""
import os, sys, numpy as np
sys.path.insert(0, r"C:/Users/user/tfb/src")
MODE = sys.argv[1]
if MODE == "build":
    s = open(r"C:/Users/user/tfb/twin/build_march.py", encoding="utf-8").read()

    def sub(a, b):
        global s
        assert s.count(a) == 1, a[:60]
        s = s.replace(a, b)

    sub('SPLIT = sys.argv[2] if len(sys.argv) > 2 else "validation"; OUTN = os.environ.get("TFB_MONTH_OUT") or {"validation": "nn_data_march", "private": "nn_data_april"}[SPLIT]',
        'SPLIT = "train"; OUTN = "nn_data_jf_sp1"')
    sub('    for r in pd.read_csv(REL / "task2" / p / SPLIT / "window_index.csv").itertuples():\n        b0 = ti[r.forecast_origin]; blank[b0 + 1:b0 + 19] = True\n', '')
    sub('        if (t - H) // 288 != (t + F) // 288 or blank[t - H:t + F + 1].any():',
        '        if str(P.times[t])[:7] not in ("2031-01", "2031-02") or (t - H) // 288 != (t + F) // 288 or blank[t - H:t + F + 1].any():')
    sys.argv = ["build_march.py", "1"]; sys.path.insert(0, r"C:/Users/user/tfb/twin")
    exec(compile(s, "build_march_jf", "exec"))
    sys.exit()

from common import NN
J = {f: np.load(f"{NN}/nn_data_jf_sp1/{f}.npy", allow_pickle=True) for f in ("panel", "t", "cond", "off", "date")}
A = {f: np.load(f"{NN}/nn_data_afdem/{f}.npy", allow_pickle=True) for f in ("panel", "t", "cond", "off", "date", "is_sel", "is_eval")}
out, last = [], {}
for i, (p, t, c) in enumerate(zip(J["panel"], J["t"].astype(int), J["cond"])):
    if t - last.get((p, c), -10 ** 9) >= 6:
        out.append(i); last[(p, c)] = t
k6 = np.array(out); mon = np.array([str(d)[:7] for d in A["date"]]); jf = np.flatnonzero(np.isin(mon, ("2031-01", "2031-02")))
SJ = {(str(J["panel"][i]), int(J["t"][i]), str(J["cond"][i])) for i in k6}; SA = {(str(A["panel"][i]), int(A["t"][i]), str(A["cond"][i])) for i in jf}
print(f"dense pool {len(J['t'])} windows; spacing 6 {len(k6)}; afdem January / February {len(jf)}; common {len(SJ & SA)}, only dense {len(SJ - SA)}, only afdem {len(SA - SJ)}")
XA = np.load(f"{NN}/nn_data_afdem/X.npy", mmap_mode="r"); XJ = np.load(f"{NN}/nn_data_jf_sp1/X.npy", mmap_mode="r"); ia = {(str(A["panel"][i]), int(A["t"][i]), str(A["cond"][i])): i for i in jf}; dm = []
for i in k6[:300]:
    key = (str(J["panel"][i]), int(J["t"][i]), str(J["cond"][i]))
    if key in ia:
        a = np.asarray(XJ[J["off"][i]:J["off"][i + 1]]); b = np.asarray(XA[A["off"][ia[key]]:A["off"][ia[key] + 1]])
        dm.append(float(np.nanmax(np.abs(np.nan_to_num(a) - np.nan_to_num(b)))) if a.shape == b.shape else np.inf)
print(f"features on {len(dm)} common windows: max |d| {max(dm) if dm else float('nan'):.3g}, share exact {np.mean(np.array(dm) == 0) if dm else float('nan'):.3f}")
