"""11/03 ledger #196: review/t2_nn3a.py = review/t2_nn3.py + the config key AW (AW=1: the ongoing training windows weighted by their jam-age
bin, C:/tfb_data/tfb_nn/age196/aw.npy from g235, path in NN_AGEFILE; the ongoing windows' mean training weight is kept, onset unchanged).
AW=0 is t2_nn3.py.
usage: py make_t2_nn3a.py"""
s = open(r"C:/Users/user/tfb/review/t2_nn3.py", encoding="utf-8").read()


def sub(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)


sub('        GWt = torch.from_numpy(GW).to(dev)\n',
    '        if AW_:                                                 # 11/03 ledger #196: jam-age weights of the ongoing training windows (g235)\n'
    '            aw = np.load(os.environ["NN_AGEFILE"]); assert len(aw) == len(GW)\n'
    '            og_ = tr[np.array([str(c) == "queue_ongoing" for c in cond[tr]])]; s0_ = GW[og_].sum()\n'
    '            GW[og_] = GW[og_] * aw[og_]; GW[og_] = GW[og_] * (s0_ / GW[og_].sum())   # the ongoing mean weight stays\n'
    '            print(f"AW: ongoing training windows {len(og_)}, weight min {GW[og_].min():.3f} max {GW[og_].max():.3f}, onset unchanged", flush=True)\n'
    '        GWt = torch.from_numpy(GW).to(dev)\n')
sub('    global LOV, TAILN, AUXD; AUXD = int(kv.get("AUXD", 0));', '    global LOV, TAILN, AUXD, AW_; AW_ = int(kv.get("AW", 0)); AUXD = int(kv.get("AUXD", 0));')
sub('"SIMW", "DOM", "MW", "EMA"}', '"SIMW", "DOM", "MW", "EMA", "AW"}')
sub('import os, sys, time, pickle, numpy as np, torch, torch.nn as nn\n', 'import os, sys, time, pickle, numpy as np, torch, torch.nn as nn\nAW_ = 0\n')
open(r"C:/Users/user/tfb/review/t2_nn3a.py", "w", encoding="utf-8").write(s); print("t2_nn3a.py written")
