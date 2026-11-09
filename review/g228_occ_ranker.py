"""G228 (11/03, ledger #200): occupancy, which no Task 2 feature reads, tried on the physics onset ranker as G214 (#180) did for ramps:
G214's source with its extra features replaced by two occupancy columns, the within-window percentile of O / k_c at row T-1 and over
rows T-3 .. T-1 (the row at T is not read). Training months read occupancy under the Task 1 blank (P.blank), March as released.
Arm oc against b0, the current ranker retrained the same way.
usage: TFB_M3VER=v3 py -B -X utf8 g228_occ_ranker.py"""
s = open(r"C:/Users/user/tfb/review/g214_ramp_cap.py", encoding="utf-8").read()
a0, a1 = s.index("def extra(p, sp, t, kk, t0):"), s.index("COLS_ = ")
NEW = '''def extra(p, sp, t, kk, t0):
    if (p, sp) not in RC:
        P = load(p, sp); O = np.where(P.blank, np.nan, P.O) if sp == "train" else np.asarray(P.O, float); RC[(p, sp)] = (O, P.kc.copy())
    O, kc = RC[(p, sp)]
    with np.errstate(all="ignore"):
        o1 = O[t - 1] / kc; o3 = np.nanmean(O[t - 3:t], axis=0) / kc
    return (np.column_stack([pct(o1), pct(o3)]),)


'''
s = s[:a0] + NEW + s[a1:]
for a, b, n in (('COLS_ = {"base": (), "rp": (0,), "cp": (1,), "rc": (0, 1)}', 'COLS_ = {"base": (), "oc": (0,)}', 1),
                ('"b0", "rp", "cp", "rc"', '"b0", "oc"', None), ('"g214_m3.csv"', '"g228_m3.csv"', 1), ('"g214_train.csv"', '"g228_train.csv"', 1)):
    assert (s.count(a) == n) if n else s.count(a) >= 1, (a, s.count(a))
    s = s.replace(a, b)
exec(compile(s, "g214o", "exec"))
