"""11/03 ledger #193: review/t2_nn3r.py = review/t2_nn3.py + the config key RW (loss weight of the rare-birth training windows,
C:/tfb_data/tfb_nn/rare193/rare.npy from g226, path in NN_RAREFILE; mean training weight kept at 1 as MW does). RW=1 is t2_nn3.py.
usage: py make_t2_nn3r.py"""
s = open(r"C:/Users/user/tfb/review/t2_nn3.py", encoding="utf-8").read()


def sub(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)


sub('        GWt = torch.from_numpy(GW).to(dev)\n',
    '        if RW_ != 1.0:                                          # 11/03 ledger #193: weight of the rare-birth training windows (g226)\n'
    '            rr = np.load(os.environ["NN_RAREFILE"]); assert len(rr) == len(GW)\n'
    '            GW[tr] = GW[tr] * np.where(rr[tr], RW_, 1.0)\n'
    '            GW[tr] = GW[tr] * (len(tr) / GW[tr].sum())        # mean training weight stays 1\n'
    '            on_ = np.array([str(c) == "queue_onset" for c in cond])[tr]\n'
    '            print(f"RW={RW_}: rare-birth training windows {int(rr[tr].sum())} of {int(on_.sum())} onset, onset weight share {GW[tr][rr[tr]].sum() / GW[tr][on_].sum():.3f}", flush=True)\n'
    '        GWt = torch.from_numpy(GW).to(dev)\n')
sub('    global LOV, TAILN, AUXD; AUXD = int(kv.get("AUXD", 0));', '    global LOV, TAILN, AUXD, RW_; RW_ = float(kv.get("RW", 1)); AUXD = int(kv.get("AUXD", 0));')
sub('"SIMW", "DOM", "MW", "EMA"}', '"SIMW", "DOM", "MW", "EMA", "RW"}')
sub('import os, sys, time, pickle, numpy as np, torch, torch.nn as nn\n', 'import os, sys, time, pickle, numpy as np, torch, torch.nn as nn\nRW_ = 1.0\n')
open(r"C:/Users/user/tfb/review/t2_nn3r.py", "w", encoding="utf-8").write(s); print("t2_nn3r.py written")
