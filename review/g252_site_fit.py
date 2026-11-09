"""G252 fit (11/04, ledger #217, the strategy's (M), reopening #105): G144's onset site model (3 dilated Conv1d blocks over the links,
listwise softmax CE on the eligible T+30 links, 3 seeds x 20 epochs, inputs <= T), refitted and saved so that its link scores can be
mixed into the onset flow: <hold> = 2031-01 or 2031-02 (that month left out, for the January / February read), all (the nine months,
for the M3 dev read; M3 is March). Writes work/t2/onset_place/g252_<hold>.pt (the three state dicts).
usage: TFB_M3VER=v3 py -B -X utf8 g252_site_fit.py <2031-01 | 2031-02 | all>"""
import sys, torch
R_ = r"C:/Users/user/tfb/review/"; HOLD = sys.argv[1]; sys.argv = [R_ + "g144_onset_place.py", "run"]
src = open(R_ + "g144_onset_place.py", encoding="utf-8").read(); cut = "rows = load_train()"; assert src.count(cut) == 1
exec(compile(src.split(cut)[0], "g144", "exec"))
rows = load_train(); tr = [r for r in rows if r["month"] != HOLD] if HOLD != "all" else rows
print(HOLD, "train windows", len(tr), "of", len(rows), flush=True)
nets = [fit(tr, s) for s in range(SEEDS)]
torch.save([n.state_dict() for n in nets], OUT + f"g252_{HOLD}.pt"); print("saved", OUT + f"g252_{HOLD}.pt", flush=True)
