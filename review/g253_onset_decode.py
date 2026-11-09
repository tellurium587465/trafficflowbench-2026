"""G253 (11/04, ledger #216, the strategy's (N) on onset): G170's onset instrument with the decode variants of dex.py on the final A's
onset flow (the evidence rule as the finals, B = 0.25): ex = exact expected IoU (DEX), nz = the T+30 row rescaled on a non-empty truth
(NZ) then best_set, exnz = both. Readings as G170: M3 dev (100 windows, hold-out days not read) by judge; January / February L / F / M.
usage: TFB_M3VER=v3 py -B -X utf8 g253_onset_decode.py [time | run]"""
import sys
sys.path.insert(0, r"C:/Users/user/tfb/review"); import dex as _dx
s = open(r"C:/Users/user/tfb/review/g170_onset_instrument.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


sub('"hid", "hidf", "ac"); ND = 8', '"hid", "hidf", "ac", "ex", "nz", "exnz"); ND = 8')
sub('    return {v: decode(x)[-1] for v, x in pm.items()}\n',
    '    out = {v: decode(x)[-1] for v, x in pm.items()}; out["ex"] = _dx.DEX(b)[-1]; out["nz"] = decode(_dx.NZ(b))[-1]; out["exnz"] = _dx.DEX(_dx.NZ(b))[-1]\n    return out\n')
sub('"g170_m3.csv"', '"g253_m3.csv"'); sub('"g170_train.csv"', '"g253_train.csv"')
exec(compile(s, "g170x", "exec"))
